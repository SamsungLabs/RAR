import os, sys 
import numpy as np 
import pandas as pd 
from dataclasses import make_dataclass 

root = r'$output_filder'
models = os.listdir(root)

# dataset:
dataset_list = ["SIDD", "GoPro", "DIV2K", "LOL", "SOTS", "Rain100L", "RainDrop", "UDC", "EUVP", "Kodak", "CDD11_haze_rain", "CDD11_low_haze_rain", 
                "MiO100_GroupA-Blur+Compression", "MiO100_GroupA-Blur+Fog", "MiO100_GroupA-Blur+Low-light", "MiO100_GroupA-Blur+Low-Resolution", "MiO100_GroupA-Low-light+Noise", "MiO100_GroupA-Noise+Compression", "MiO100_GroupA-Rain+Fog", "MiO100_GroupA-Rain+Low-Resolution", 
                "MiO100_GroupB-Blur+Compression", "MiO100_GroupB-Blur+Low-Resolution", "MiO100_GroupB-Fog+Noise", "MiO100_GroupB-Rain+Low-light", "MiO100_GroupC-Blur+Blur+Noise", "MiO100_GroupC-Fog+Blur+Low-Resolution", "MiO100_GroupC-Low-light+Blur+Compression", "MiO100_GroupC-Rain+Noise+Low-Resolution",]

# data format:
"""
index: [dataset, global_step, psnr ssim lpips clipiqa musiq maniqa fid niqe nima]}
"""
Point = make_dataclass("Point", [("dataset", str), ("global_step", float), ("psnr", float), ("ssim", float), ("lpips", float), ("clipiqa", float), ("musiq", float), ("maniqa", float), ("fid", float), ("niqe", float), ("nima", float)])

final_line = 1
print("="*100)
for mm in models:
    result_log = []
    path = os.path.join(root, mm)
    if not os.path.isdir(path):
        continue
    dataset = os.listdir(path)
    print(mm)
    print("="*100)
    n_round = 4
    if "n16" in mm:
        n_round = 16
    for data in dataset_list:
        if data in dataset:
            # read file
            logfile = os.path.join(os.path.join(path, data), "log.txt")
            if os.path.isfile(logfile): # file exists
                meta_info = []
                with open(logfile, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        meta_info.append(line.strip())
                
                # find time idx
                final_index = 0
                for idx, line in enumerate(meta_info):
                    if "INFO: Final Results:" in line:
                        final_index = idx
                        break
                try:
                    if final_index == 0: # didn't find
                        result_log.append(Point(data, -1, None, None, None, None, None, None, None, None, None))
                    else:
                        # 2025-10-28 14:31:31,579 SD35M INFO: Final Results:
                        # 2025-10-28 14:31:31,579 SD35M INFO: ====================================================================================================
                        # 2025-10-28 14:31:31,580 SD35M INFO: 22.269239482879637 0.6676999037917253 0.18334856338799 0.6240313529968262 55.956473655700684 0.4290472193062305 nan 
                        # 2025-10-28 14:31:31,580 SD35M INFO: Total Rounds: 223, Average Rounds: 2.23
                        # 2025-10-28 14:31:31,580 SD35M INFO: ====================================================================================================
                        round_data = meta_info[final_index+3].split(" ")[9]
                        result_data = meta_info[final_index+2].split(" ")
                        psnr =      float(result_data[4])
                        ssim =      float(result_data[5])
                        lpips =     float(result_data[6])
                        clipiqa =   float(result_data[7])
                        musiq =     float(result_data[8])
                        maniqa =    float(result_data[9])
                        fid =       float(result_data[10])
                        niqe   =    float(result_data[11])
                        nima   =    float(result_data[12])
                        print(f"{data}: {round_data} {psnr} {ssim} {lpips} {clipiqa} {musiq} {maniqa} None {niqe} {nima}")
                        result_log.append(Point(data, round_data, round(psnr, 2), round(ssim, 4), round(lpips, 4), round(clipiqa, 4), round(musiq, 2), round(maniqa, 4), None, round(niqe, 4), round(nima, 4)))
                except:
                    result_log.append(Point(data, -1, None, None, None, None, None, None, None, None, None))

                # find round idx
                round_idx = 0
                result_index = 0
                for idx, line in enumerate(meta_info):
                    if "DiT Results:" in line:
                        result_index = idx
                        break
                try: 
                    if result_index == 0:
                        while round_idx < (n_round + 1):
                            result_log.append(Point(data, round_idx, None, None, None, None, None, None, None, None, None))
                    else:
                        while round_idx < (n_round+1):
                            idx = result_index + 3 + round_idx
                            result = meta_info[idx].split(" ")  
                            psnr =      float(result[4])
                            ssim =      float(result[5])
                            lpips =     float(result[6])
                            clipiqa =   float(result[7])
                            musiq =     float(result[8])
                            maniqa =    float(result[9])
                            fid =       float(result[10])
                            niqe   =    float(result_data[11])
                            nima   =    float(result_data[12])
                            print(f"{data}: {round_data} {psnr} {ssim} {lpips} {clipiqa} {musiq} {maniqa} {fid} {niqe} {nima}")
                            result_log.append(Point(data, round_idx, round(psnr, 2), round(ssim, 4), round(lpips, 4), round(clipiqa, 4), round(musiq, 2), round(maniqa, 4), round(fid, 4), round(niqe, 4), round(nima, 4)))
                            round_idx += 1
                except:
                    while round_idx < (n_round+1):
                        result_log.append(Point(data, round_idx, None, None, None, None, None, None, None, None, None))
                        round_idx += 1
        else:
            result_log.append(Point(data, -1, None, None, None, None, None, None, None, None, None))
            write_line = 0
            while write_line < (n_round+1):
                result_log.append(Point(data, write_line, None, None, None, None, None, None, None, None, None))
                write_line += 1

    print("")
    print("="*100)

    df = pd.DataFrame(result_log)
    df.to_csv(os.path.join(path, "summary.csv"), index=False)