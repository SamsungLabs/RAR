#/bin/bash
set -e

np=$1
work_dir=$2

while [[ $# -gt 0 ]]; do
    case $3 in
        --np=*)
            np="${1#*=}"
            shift
            ;;
        *.yaml)
            config=$3
            shift
            ;;
        *)
            other_args+=("$3")
            shift
            ;;
    esac
done


if [[ -z "$config" ]]; then
    config="configs/sd35_imgflow_config/ir_sd35m_r512_multi.yaml"
    echo "No yaml file specified. Set to --config_path=$config"
fi

cmd="TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=$((RANDOM % 10000 + 20000))  \
        train_scripts_imgflow/train_imgflow_e2e_multi.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --name=tmp \
        --report_to=tensorboard \
        --debug=true \
        ${other_args[@]}"

echo $cmd
eval $cmd


