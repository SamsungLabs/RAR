#!/usr/bin/env bash
# run_gpu_queue.sh

# bash scripts/test_autocomparison.sh 0,1,2,3,4,5,6,7 configs/infer_cfg.yaml checkpoints/RAR_modelzoo/RAR/checkpoints/epoch_5_step_60506_weight.pth output_composite/results/ 4 4 256 online d2c multi_c2c_d2c_online_ep5_compare_s4r4 bf16

# 動態排程：每個實驗占 1 張 GPU，同時最多 N 張；任務完成即將下一個排上該 GPU。
# testing configuration
set -euo pipefail
mkdir -p logs
config=$2
model_path=$3
work_dir=$4
step=$5
num_rounds=$6
resolution=$7
mode=$8
flow_type=$9
log_tag=${10}
sampling_algo=flow_euler   # flow_dpm-solver or flow_euler
cfg_scale=1.0
pag_scale=1.0
save_nums=300
sample_nums=300
suffix=_SD35M_ep1_wstatus
num_workers=3
weight_type=${11}
src_dir=./dataset/PIR_tar


cleanup() {
  echo ""
  echo "Interrupt All Process."
  pkill -P $$
  exit 1
}
trap cleanup SIGINT SIGTERM

# === 可調區 ===
# 從命令列覆蓋要用的 GPU ---
# 用法：bash run_gpu_queue.sh 0,2,4,6
if [[ $# -ge 1 && -n "$1" ]]; then
  # 去除空白並以逗號分割
  gpu_csv="${1//[[:space:]]/}"
  IFS=',' read -r -a GPUS <<< "$gpu_csv"

  # 驗證都是非負整數，順便去重
  declare -A _seen=()
  _tmp=()
  for g in "${GPUS[@]}"; do
    if [[ ! "$g" =~ ^[0-9]+$ ]]; then
      echo "❌ 無效的 GPU ID: '$g'（必須是非負整數，以逗號分隔）" >&2
      exit 2
    fi
    if [[ -z "${_seen[$g]:-}" ]]; then
      _tmp+=("$g"); _seen[$g]=1
    fi
  done
  if (( ${#_tmp[@]} == 0 )); then
    echo "❌ 未提供有效的 GPU 列表。" >&2
    exit 2
  fi
  GPUS=("${_tmp[@]}")
fi
echo "🧩 使用的 GPU 清單：${GPUS[*]}"

# define dataset
ITEMS=(
    # single distortion
    # "Denoise/SIDD/metas/test_iqa_A_brief$suffix.json|SIDD"
    "Deblur/GoPro/metas/test_iqa_A_brief$suffix.json|GoPro"
    "Dehaze/SOTS/metas/test_iqa_A_brief$suffix.json|SOTS"
    "SuperResolution/DIV2K/metas/DIV2K_valid_pair_SR_iqa_A_brief$suffix.json|DIV2K"
    "LowLight/LOL/metas/test_iqa_A_brief$suffix.json|LOL"
    "Derain/Rain100L/metas/test_iqa_A_brief$suffix.json|Rain100L"
    "Derain/RainDrop/metas/Raindrop_test_b_iqa_A_brief$suffix.json|RainDrop"
    "Denoise/Kodak/metas/Kodak_Noise_L3_iqa_A_brief$suffix.json|Kodak"
    "Other/UDC/metas/test_iqa_A_brief$suffix.json|UDC"
    "Other/EUVP/metas/test_iqa_A_brief$suffix.json|EUVP"

    # "Desnow/Snow100k/metas/test_M_iqa_A_brief$suffix.json|Snow100k"

    # "Composite/CDD11/metas/test_haze_rain_iqa_A_brief$suffix.json|CDD11_haze_rain"
    # "Composite/CDD11/metas/test_low_haze_rain_iqa_A_brief$suffix.json|CDD11_low_haze_rain"
    # "Composite/CDD11/metas/test_haze_iqa_A_brief$suffix.json|CDD11_haze"
    # "Composite/CDD11/metas/test_haze_snow_iqa_A_brief$suffix.json|CDD11_haze_snow"
    # "Composite/CDD11/metas/test_low_haze_iqa_A_brief$suffix.json|CDD11_low_haze"
    # "Composite/CDD11/metas/test_low_haze_snow_iqa_A_brief$suffix.json|CDD11_low_haze_snow"
    # "Composite/CDD11/metas/test_low_iqa_A_brief$suffix.json|CDD11_low"
    # "Composite/CDD11/metas/test_low_rain_iqa_A_brief$suffix.json|CDD11_low_rain"
    # "Composite/CDD11/metas/test_low_snow_iqa_A_brief$suffix.json|CDD11_low_snow"
    # "Composite/CDD11/metas/test_rain_iqa_A_brief$suffix.json|CDD11_rain"
    # "Composite/CDD11/metas/test_snow_iqa_A_brief$suffix.json|CDD11_snow"

    "Composite/MiO100/metas/test-GroupA-Blur+Compression_iqa_A_brief$suffix.json|MiO100_GroupA-Blur+Compression"
    "Composite/MiO100/metas/test-GroupA-Blur+Fog_iqa_A_brief$suffix.json|MiO100_GroupA-Blur+Fog"
    "Composite/MiO100/metas/test-GroupA-Blur+Low-light_iqa_A_brief$suffix.json|MiO100_GroupA-Blur+Low-light"
    "Composite/MiO100/metas/test-GroupA-Blur+Low-Resolution_iqa_A_brief$suffix.json|MiO100_GroupA-Blur+Low-Resolution"
    "Composite/MiO100/metas/test-GroupA-Low-light+Noise_iqa_A_brief$suffix.json|MiO100_GroupA-Low-light+Noise"
    "Composite/MiO100/metas/test-GroupA-Noise+Compression_iqa_A_brief$suffix.json|MiO100_GroupA-Noise+Compression"
    "Composite/MiO100/metas/test-GroupA-Rain+Fog_iqa_A_brief$suffix.json|MiO100_GroupA-Rain+Fog"
    "Composite/MiO100/metas/test-GroupA-Rain+Low-Resolution_iqa_A_brief$suffix.json|MiO100_GroupA-Rain+Low-Resolution"
    "Composite/MiO100/metas/test-GroupB-Blur+Compression_iqa_A_brief$suffix.json|MiO100_GroupB-Blur+Compression"
    "Composite/MiO100/metas/test-GroupB-Blur+Low-Resolution_iqa_A_brief$suffix.json|MiO100_GroupB-Blur+Low-Resolution"
    "Composite/MiO100/metas/test-GroupB-Fog+Noise_iqa_A_brief$suffix.json|MiO100_GroupB-Fog+Noise"
    "Composite/MiO100/metas/test-GroupB-Rain+Low-light_iqa_A_brief$suffix.json|MiO100_GroupB-Rain+Low-light"
    "Composite/MiO100/metas/test-GroupC-Blur+Blur+Noise_iqa_A_brief$suffix.json|MiO100_GroupC-Blur+Blur+Noise"
    "Composite/MiO100/metas/test-GroupC-Fog+Blur+Low-Resolution_iqa_A_brief$suffix.json|MiO100_GroupC-Fog+Blur+Low-Resolution"
    "Composite/MiO100/metas/test-GroupC-Low-light+Blur+Compression_iqa_A_brief$suffix.json|MiO100_GroupC-Low-light+Blur+Compression"
    "Composite/MiO100/metas/test-GroupC-Rain+Noise+Low-Resolution_iqa_A_brief$suffix.json|MiO100_GroupC-Rain+Noise+Low-Resolution"

    "Composite/MiO100/metas/test-GroupA-Blur+Compression_iqa_A_brief_single$suffix.json|MiO100_GroupA-Blur+Compression"
    "Composite/MiO100/metas/test-GroupA-Blur+Fog_iqa_A_brief_single$suffix.json|MiO100_GroupA-Blur+Fog"
    "Composite/MiO100/metas/test-GroupA-Blur+Low-light_iqa_A_brief_single$suffix.json|MiO100_GroupA-Blur+Low-light"
    "Composite/MiO100/metas/test-GroupA-Blur+Low-Resolution_iqa_A_brief_single$suffix.json|MiO100_GroupA-Blur+Low-Resolution"
    "Composite/MiO100/metas/test-GroupA-Low-light+Noise_iqa_A_brief_single$suffix.json|MiO100_GroupA-Low-light+Noise"
    "Composite/MiO100/metas/test-GroupA-Noise+Compression_iqa_A_brief_single$suffix.json|MiO100_GroupA-Noise+Compression"
    "Composite/MiO100/metas/test-GroupA-Rain+Fog_iqa_A_brief_single$suffix.json|MiO100_GroupA-Rain+Fog"
    "Composite/MiO100/metas/test-GroupA-Rain+Low-Resolution_iqa_A_brief_single$suffix.json|MiO100_GroupA-Rain+Low-Resolution"
    "Composite/MiO100/metas/test-GroupB-Blur+Compression_iqa_A_brief_single$suffix.json|MiO100_GroupB-Blur+Compression"
    "Composite/MiO100/metas/test-GroupB-Blur+Low-Resolution_iqa_A_brief_single$suffix.json|MiO100_GroupB-Blur+Low-Resolution"
    "Composite/MiO100/metas/test-GroupB-Fog+Noise_iqa_A_brief_single$suffix.json|MiO100_GroupB-Fog+Noise"
    "Composite/MiO100/metas/test-GroupB-Rain+Low-light_iqa_A_brief_single$suffix.json|MiO100_GroupB-Rain+Low-light"
    "Composite/MiO100/metas/test-GroupC-Blur+Blur+Noise_iqa_A_brief_single$suffix.json|MiO100_GroupC-Blur+Blur+Noise"
    "Composite/MiO100/metas/test-GroupC-Fog+Blur+Low-Resolution_iqa_A_brief_single$suffix.json|MiO100_GroupC-Fog+Blur+Low-Resolution"
    "Composite/MiO100/metas/test-GroupC-Low-light+Blur+Compression_iqa_A_brief_single$suffix.json|MiO100_GroupC-Low-light+Blur+Compression"
    "Composite/MiO100/metas/test-GroupC-Rain+Noise+Low-Resolution_iqa_A_brief_single$suffix.json|MiO100_GroupC-Rain+Noise+Low-Resolution"
)

# 你的 16 個實驗指令（每個只用 1 張卡；程式需尊重 CUDA_VISIBLE_DEVICES）
EXPS=()
for item in "${ITEMS[@]}"; do
    IFS='|' read -r ds tag <<< "$item"
    EXPS+=("python scripts/test_autocomparison.py --meta_file=$ds --tag=$tag --config=$config --model_path=$model_path --work_dir=$work_dir --data_dir=$src_dir --sample_nums=$sample_nums --resolution=$resolution --bs=1 --num_workers=$num_workers --weight_type=$weight_type --cfg_scale=$cfg_scale --pag_scale=$pag_scale --step=$step --num_rounds=$num_rounds --save_nums=$save_nums --mode=$mode --flow_type=$flow_type --sampling_algo=$sampling_algo --save_result=True")
done


LOGDIR="logs"            # 日誌資料夾
mkdir -p "$LOGDIR"
LOGDIR="logs/$log_tag"
mkdir -p "$LOGDIR"

total=${#EXPS[@]}
ngpu=${#GPUS[@]}
next=0
# 每張 GPU 目前執行的 PID（0 代表空閒）
GPU_PIDS=()
for ((gi=0; gi<ngpu; gi++)); do 
  GPU_PIDS[gi]=0; 
  echo "GPU: $gi"
done

while (( next < total )); do
  # 若已達併發上限，就等到有背景作業結束
  while (( $(jobs -pr | wc -l) >= ngpu )); do
    sleep 1
  done
  # Submit job to empty gpus
  for ((gi=0; gi<ngpu; gi++)); do
    pid="${GPU_PIDS[$gi]}"
    if (( next < total )); then
      # 空閒（pid=0）或該 pid 已結束（kill -0 失敗）就派新任務
      if [[ "$pid" -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
        echo "Run Job $next / $total, Running GPUs: $(jobs -pr | wc -l) / $ngpu, Current GPU: $gi, Pid: $pid."
        gpu="${GPUS[$gi]}"
        cmd="${EXPS[next]}"
        IFS='|' read -r ds tag <<< "${ITEMS[$next]}"
        name=$tag
        echo "▶️ Launch $name on GPU $gpu: $cmd"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
          nohup bash -lc "$cmd" >"${LOGDIR}/${name}_${flow_type}_${mode}.out" 2>"${LOGDIR}/${name}_${flow_type}_${mode}.err" &
        GPU_PIDS[$gi]=$!
        next=$((next+1))
      fi
    fi
  done
done

# 等全部作業結束
wait
echo "🎉 All done. Logs in ./logs/"
