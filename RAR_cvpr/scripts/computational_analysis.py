# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
--meta_file=Composite/CDD11/metas/test_haze_rain_iqa_A_brief.json \
--tag=CDD11_haze_rain \

CUDA_VISIBLE_DEVICES=0 python scripts/computational_analysis.py \
      --config=output_composite/sd35m_d2c_multi_bf16_lr2e6/config.yaml \
      --model_path=output_composite/sd35m_d2c_multi_bf16_lr2e6/checkpoints/epoch_5_step_60506_weight.pth \
      --work_dir=output_composite/computational_analysis/ \
      --data_dir=/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/Datasets/IR \
      --meta_file=Other/UDC/metas/test_iqa_A_brief_SD35M_ep1_wstatus.json \
      --tag=UDC \
      --sample_nums=300 \
      --resolution=256 \
      --bs=1 \
      --cfg_scale=1.0 \
      --pag_scale=1.0 \
      --sampling_algo=flow_euler \
      --save_result=True \
      --save_nums=1 \
      --step=4 \
      --num_rounds=1 \
      --flow_type=d2c \
      --mode=online  

"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os, sys
import re
import subprocess
import tarfile
import time
import warnings
import random 
import numpy as np 
from einops import rearrange
from PIL import Image
from dataclasses import dataclass, field
from torch.utils import data
import math 
import yaml
from easydict import EasyDict

# from datetime import datetime
from typing import List, Optional

import pyrallis
import torch
import torch.nn as nn 
from termcolor import colored
from torchvision.utils import save_image
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning
import time
from diffusion import DPMS, FlowEuler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_encode, vae_decode
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar
from diffusion.utils.config import SanaConfig, model_init_config
# from tools.download import find_model
import logging 
import json
import torchvision.transforms as T
import pyiqa
from diffusion.model.sd35 import load_scheduler, load_vae, load_mmdit, load_text_encoder
import torchvision.transforms.functional as F 
from torchvision.transforms import InterpolationMode
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusion.model.utils import set_fp32_attention, set_grad_checkpoint

# from fvcore.nn import FlopCountAnalysis
from torch.profiler import profile, ProfilerActivity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# question dictionary:
question_dict = {
    "Quality Comparison": [
        "Make a judgment on which image, Image A or Image B, you consider to be of better quality. Answer the question using a single word or phrase.",
        "Assess the quality of Image A and Image B, and indicate which one you find to be better. Answer the question using a single word or phrase.",
        "Which image do you believe has better overall quality: Image A or Image B? Answer the question using a single word or phrase.",
        "Evaluate Image A and Image B, and select the one that you feel has better quality. Answer the question using a single word or phrase.",
        "Determine which image, Image A or Image B, you perceive to have better quality. Answer the question using a single word or phrase.",
        "Compare the quality of Image A and Image B, and determine which one you prefer. Answer the question using a single word or phrase.",
        "Between Image A and Image B, which image do you perceive to have better quality overall? Answer the question using a single word or phrase.",
        "In your opinion, which image demonstrates superior quality: Image A or Image B? Answer the question using a single word or phrase.",
        "Determine which image exhibits higher quality between Image A and Image B. Answer the question using a single word or phrase.",
        "Which of the two images, Image A or Image B, appears to have superior quality to you? Answer the question using a single word or phrase.",
        "Which image, Image A or Image B, do you think displays better quality when compared? Answer the question using a single word or phrase.",
        "Differentiate between Image A and Image B in terms of overall quality and decide which one is superior. Answer the question using a single word or phrase.",
        "Can you compare the quality of Image A and Image B and decide which one is better? Answer the question using a single word or phrase.",
        "Compare the general quality of Image A and Image B, and state your preference. Answer the question using a single word or phrase.",
        "Decide which image, Image A or Image B, you think possesses higher quality. Answer the question using a single word or phrase.",
        "Between Image A and Image B, which image do you think has better quality overall? Answer the question using a single word or phrase.",
        "Assess the quality of Image A and Image B, and choose the one you believe is superior. Answer the question using a single word or phrase.",
        "Which of the two images, Image A or Image B, do you consider to be of better quality? Answer the question using a single word or phrase.",
        "Evaluate the quality of Image A and Image B, and decide which one is superior. Answer the question using a single word or phrase.",
        "Which image stands out to you as having better quality: Image A or Image B? Answer the question using a single word or phrase.",
    ],
    "Distortion Identification": [
        "Determine the leading ONE degradation in the evaluated image.",
        "Determine the most impactful ONE distortion in the evaluated image.",
        "Highlight the most significant ONE distortion in the evaluated image.",
        "Identify the chief ONE degradation in the evaluated image.",
        "Identify the most critical ONE distortion in the evaluated image.",
        "Identify the most notable ONE distortion in the evaluated image's quality.",
        "In terms of image quality, what is the most glaring ONE issue with the evaluated image?",
        "In the evaluated image, what ONE distortion is most detrimental to image quality?",
        "Pinpoint the foremost ONE image quality issue in the evaluated image.",
        "What ONE distortion is most apparent in the evaluated image?",
        "What ONE distortion is most evident in the evaluated image?",
        "What ONE distortion is most prominent in the evaluated image?",
        "What ONE distortion is most prominent when examining the evaluated image?",
        "What ONE distortion most detrimentally affects the overall quality of the evaluated image?",
        "What ONE distortion most notably affects the clarity of the evaluated image?",
        "What ONE distortion most significantly affects the evaluated image?",
        "What ONE distortion stands out in the evaluated image?",
        "What ONE quality degradation is most apparent in the evaluated image?",
        "What critical ONE quality degradation is present in the evaluated image?",
        "What is the foremost ONE distortion affecting the evaluated image's quality?",
        "What is the leading ONE distortion in the evaluated image?",
        "What is the most critical ONE image quality issue in the evaluated image?",
        "What is the most severe ONE degradation observed in the evaluated image?",
        "What is the primary ONE degradation observed in the evaluated image?"
    ],
}

def question_generate(task="Quality Comparison"):
    template = random.choice(question_dict[task])
    return template

def count_flops(model, inputs):
    model.eval()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True
    ) as prof:
        with torch.no_grad():
            model(*inputs) if isinstance(inputs, (tuple, list)) else model(inputs)

    flops = 0
    for evt in prof.key_averages():
        if evt.flops is not None:
            flops += evt.flops

    return flops

class GenerateWrapper(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn 
    
    def forward(self, inputs, latent_input=True, save_hidden=True):
        return self.fn(inputs, latent_input, save_hidden)

class DiTWrapper(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn 
    
    def forward(self, z, step=4):
        return self.fn(z, step)

class IQAIR(nn.Module):
    def __init__(self, vae, model, connector, assessment=None, device='cuda', latent_size=16, config=None):
        super().__init__()
        self.vae = vae.to(device).eval()
        self.model = model.to(device).eval()
        self.connector = connector.to(device).eval()
        if assessment:
            self.assessment = assessment.to(device).eval()
        else:
            self.assessment = None
        self.device = device
        self.latent_size = latent_size
        self.config = config
        self.gflop_dict = {}
        self.time_dict = {}

    def forward(self, lq):
        """
        VAE time: 10.821063995361328
        torch.Size([1, 16, 32, 32])
        IQA.ident time: 8.021036624908447
        torch.Size([1, 4, 4096]) torch.Size([1, 355, 4096])
        Connector time: 1.6227524280548096
        torch.Size([1, 400, 4096]) torch.Size([1, 400])
        DiT time: 0.26990318298339844
        torch.Size([1, 154, 4096]) torch.Size([1, 2048]) torch.Size([1, 16, 32, 32])
        IQA.comp time: 0.40886902809143066
        torch.Size([1, 154, 4096]) torch.Size([1, 2048]) torch.Size([1, 16, 32, 32]) torch.Size([1, 16, 32, 32])
        VAE-dec time: 2.8421449661254883
        torch.Size([1, 16, 32, 32]) torch.Size([1, 3, 256, 256])
        """

        device = self.device

        ## 1: VAE
        time_strat = time.time()
        input_images = self.vae.encode(lq.to(device)).to(device)
        input_images = self.vae.process_in(input_images).to(device)
        vae_time = time.time() - time_strat
        print("VAE.enc time:", vae_time)
        flops_vae_enc = count_flops(self.vae.encoder, (lq.to(device).to(next(self.vae.encoder.parameters()).dtype)))
        print("VAE.enc Flops:", flops_vae_enc/1e9, " GFLOPs.")
        if "vae.enc" in self.gflop_dict:
            self.gflop_dict["vae.enc"] += flops_vae_enc
        else:
            self.gflop_dict["vae.enc"] = flops_vae_enc
        if "vae.enc" in self.time_dict:
            self.time_dict["vae.enc"] += vae_time
        else:
            self.time_dict["vae.enc"] = vae_time

        ## 2: IQA
        time_iqa_start = time.time()
        info = {
                "query": ["What are the leading distortion(s) in the evaluated image? Answer the question using a single word or phrase."],
                "img": input_images.to(device),
                "img_A": input_images.to(device),
                "img_B": [None],
                "img_path": ["lq"],
                "img_A_path": ["lq"],
                "img_B_path": [None],
                "temperature": 0.0,
                "top_p": 0.9,
                "max_new_tokens": 400,
                "task_type": "quality_single_A_noref",
                "output_prob_id": False,
                "output_confidence": True, 
                "sentence_model": "/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/ModelZoo/SentenceTransformers/all-MiniLM-L6-v2",
        }
        output_texts, _, _, confidences, output_pred_latent, output_prefix = self.assessment.generate(   
            info,
            latent_input=True,
            save_hidden=True,
        )
        IQA_iden_time = time.time() - time_iqa_start
        print("IQA.ident time:", IQA_iden_time)
        flops_iqa_iden = count_flops(GenerateWrapper(self.assessment.generate), (info, True, True))
        print("IQA.ident Flops:", flops_iqa_iden/1e9, " GFLOPs.")
        if "iqa.iden" in self.gflop_dict:
            self.gflop_dict["iqa.iden"] += flops_iqa_iden
        else:
            self.gflop_dict["iqa.iden"] = flops_iqa_iden
        if "iqa.iden" in self.time_dict:
            self.time_dict["iqa.iden"] += IQA_iden_time
        else:
            self.time_dict["iqa.iden"] = IQA_iden_time

        ## 3: connector
        time_connect_start = time.time()
        last_token, mask_token = token_pad_or_truncate(torch.cat([output_prefix, output_pred_latent], dim=1)) # torch.cat([prefix, pred_latent], dim=1)
        pred_tokens = self.connector(last_token, key_padding_mask=mask_token)
        connector_time = time.time() - time_connect_start
        print("Connector time:", connector_time)
        flops_connector_iden = count_flops(self.connector, (last_token, mask_token))
        print("Connector Flops:", flops_connector_iden/1e9, " GFLOPs.")
        if "connector" in self.gflop_dict:
            self.gflop_dict["connector"] += flops_connector_iden
        else:
            self.gflop_dict["connector"] = flops_connector_iden
        if "connector" in self.time_dict:
            self.time_dict["connector"] += connector_time
        else:
            self.time_dict["connector"] = connector_time

        ## 4.  DiT
        pred_context, pred_y = mapping_to_cond(pred_tokens) # (64, 154, 4096), (64, 1, 2048)
        cond, pooled = (pred_context.to(device), pred_y.to(device))
        caption_cond = {"c_crossattn": cond, "y": pooled}
        null_caption_cond = {"c_crossattn": torch.zeros(pred_context.shape).to(device), "y": torch.zeros(pred_y.shape).to(device)}
        generator = torch.Generator(device=device).manual_seed(1228)
        z = input_images
        model_kwargs = dict(img_cond=input_images, ) # model_kwargs=dict(img_cond=input_images, y=text_cond["y"], context=text_cond["c_crossattn"]
        flow_solver = FlowEuler(
            self.model,
            condition=caption_cond,
            uncondition=null_caption_cond,
            cfg_scale=1.0,
            model_kwargs=model_kwargs,
        )
        time_dit_start = time.time()
        samples = flow_solver.sample(
            z,
            steps=4,
        )
        dit_time = time.time() - time_dit_start
        print("DiT time:", dit_time)
        flops_dit = count_flops(DiTWrapper(flow_solver.sample), (z, 4))
        print("DiT Flops:", flops_dit/1e9, " GFLOPs.")
        if "DiT" in self.gflop_dict:
            self.gflop_dict["DiT"] += flops_dit
        else:
            self.gflop_dict["DiT"] = flops_dit
        if "DiT" in self.time_dict:
            self.time_dict["DiT"] += dit_time
        else:
            self.time_dict["DiT"] = dit_time

        # 5. Verify
        time_comp_start = time.time()
        info = {
                "query": ["Which image do you believe has better overall quality: Image A or Image B? Answer the question using a single word or phrase.""Which image do you believe has better overall quality: Image A or Image B? Answer the question using a single word or phrase."],
                "img": [None],
                "img_A": samples.to(device),       # current result
                "img_B": input_images.to(device),  # previous status
                "img_path": [None],
                "img_A_path": ["lq"],
                "img_B_path": ["lq"],
                "temperature": 0.0,
                "top_p": 0.9,
                "max_new_tokens": 400,
                "task_type": "quality_compare_noref",
                "output_prob_id": False,
                "output_confidence": True, 
                "sentence_model": "/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/ModelZoo/SentenceTransformers/all-MiniLM-L6-v2",
            }
        output_quality, _, _, confid_quality = self.assessment.generate(   # texts, output_ids, probs, confidences, 
            info, 
            latent_input=True,
            save_hidden=False,
        )
        IQA_comp_time = time.time() - time_comp_start
        print("IQA.comp time:", IQA_comp_time)
        flops_iqa_comp = count_flops(GenerateWrapper(self.assessment.generate), (info, True, False))
        print("IQA.comp Flops:", flops_iqa_comp/1e9, " GFLOPs.")
        if "IQA.comp" in self.gflop_dict:
            self.gflop_dict["IQA.comp"] += flops_iqa_comp
        else:
            self.gflop_dict["IQA.comp"] = flops_iqa_comp
        if "IQA.comp" in self.time_dict:
            self.time_dict["IQA.comp"] += IQA_comp_time
        else:
            self.time_dict["IQA.comp"] = IQA_comp_time

        # 6. decode
        time_decoder_start = time.time()
        recon = self.vae.process_out(samples.to(vae_dtype).to(device)).to(device)
        recon_clone = recon.clone()
        recon = self.vae.decode(recon)
        torch.cuda.empty_cache()
        recon = torch.clamp((recon + 1.0) / 2.0, min=0.0, max=1.0)
        vae_dec_time = time.time() - time_decoder_start
        print("VAE-dec time:", vae_dec_time)
        flops_vae_dec = count_flops(self.vae.decoder, (recon_clone.to(device).to(next(self.vae.decoder.parameters()).dtype)))
        print("VAE.dec Flops:", flops_vae_dec/1e9, " GFLOPs.")
        if "VAE.dec" in self.gflop_dict:
            self.gflop_dict["VAE.dec"] += flops_vae_dec
        else:
            self.gflop_dict["VAE.dec"] = flops_vae_dec
        if "VAE.dec" in self.time_dict:
            self.time_dict["VAE.dec"] += vae_dec_time
        else:
            self.time_dict["VAE.dec"] = vae_dec_time
        return recon

def token_pad_or_truncate(tokens):
    max_length = 400
    pad_side = "left"
    pad_value = 0.0

    B, _, D = tokens.shape
    y = torch.full((B, max_length, D), pad_value, dtype=tokens.dtype, device=tokens.device)
    mask = torch.ones((B, max_length), dtype=torch.bool, device=tokens.device) # T
    for idx, x in enumerate(tokens):
        L, _ = x.shape
        # truncate
        if L >= max_length: 
            if pad_side == "right":
                y[idx] = x[:max_length]
                mask[idx] = torch.zeros(max_length, dtype=torch.bool, device=x.device) # T
            else: 
                y[idx] = x[-max_length:]
                mask[idx] = torch.zeros(max_length, dtype=torch.bool, device=x.device) # T
        # padding
        else:
            if pad_side == "right":
                y[idx, :L] = x
                mask[idx, :L] = False
            else: 
                y[idx, max_length-L:] = x
                mask[idx, max_length-L:] = False
    return y, mask

def mapping_to_cond(tokens):
    l_out, l_pooled = tokens["l"]
    g_out, g_pooled = tokens["g"]
    t5_out, _ = tokens["t5xxl"]
    lg_out = torch.cat([l_out, g_out], dim=-1)  # (b, 77, 2048)
    lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1])) # (b, 77, 4096)
    context = torch.cat([lg_out, t5_out], dim=-2) # (b, 77+77, 4096)
    y = torch.cat((l_pooled, g_pooled), dim=-1)   # (b, 2048)  
    return context, y 

class ValP2PDataset(torch.utils.data.Dataset):
    """
    data format: .json: list[dict]
    [{
        "image_ref": hq_image,
        "image_A": lq_image, 
        "conversations":[
            .,
            .,
            {"from": instruct_source, "value": instruct}
        ]
    },]
    """
    def __init__(
        self,
        data_dir=None,
        meta_file=None,
        resolution=256,
        max_length=300,
        img_extension=".png",
        max_samples = None,
        caption_type = "question", # "question", "GT", "IQA"
        need_resize = True, 
        **kwargs,
    ):
        if isinstance(data_dir, list):
            data_dir = data_dir[0]
        self.data_dir = data_dir
        self.meta_file = meta_file 
        # Dataset Load
        self.dataset = []
        with open(os.path.join(self.data_dir, meta_file)) as fr:
            meta_data = json.load(fr)
            if max_samples:
                meta_data = meta_data[:min(len(meta_data), max_samples)]
            self.dataset += meta_data

        if need_resize: 
            self.transform = T.Compose([
                PairTrans(T.Resize((resolution, resolution))),  # Image.BICUBIC
                PairTrans(T.CenterCrop(resolution)),
                PairTrans(T.ToTensor()),
                PairTrans(T.Normalize([0.5], [0.5])),
            ])
        else:
            self.transform = T.Compose([
                PairTrans(T.CenterCrop(resolution)),
                PairTrans(T.ToTensor()),
                PairTrans(T.Normalize([0.5], [0.5])),
            ])
        
        self.transform.image_size = resolution
        self.resolution = resolution
        self.max_length = max_length
        self.default_prompt = "prompt"
        self.img_extension = img_extension
        self.caption_type = caption_type
        self.need_resize = need_resize

    def __getitem__(self, idx):
        # meta info
        meta = self.dataset[idx]
        # adding question
        meta["quality_comparison"] = [
                {
                    "from": "human",
                    "value": question_generate(),
                },
                {
                    "from": "gpt",
                    "value": "Same"
                }
            ]

        lq_path = os.path.join(self.data_dir, meta["image_A"])
        hq_path = os.path.join(self.data_dir, meta["image_ref"])

        filename, ext = os.path.splitext(os.path.basename(lq_path))
        self.img_extension = ext
        self.key = filename
        info = {}
        if self.caption_type == "question":
            info[self.default_prompt] = meta["conversations"][0]["value"]
        elif self.caption_type == "IQA":
            info[self.default_prompt] = meta["conversations"][2]["value"].replace("\n ", "")
        else:
            info[self.default_prompt] = meta["conversations"][1]["value"]

        caption_type, caption_clipscore = self.default_prompt, 0.0
        txt_fea = info[self.default_prompt]

        lq = Image.open(lq_path).convert("RGB")
        hq = Image.open(hq_path).convert("RGB")
        lq, hq = self.transform([lq, hq])

        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)  # 1x1xT
        return {
            "lq": lq,
            "hq": hq,
            "txt_fea": txt_fea,
            "attention_mask": attention_mask.to(torch.int16),
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),  # "img_hw"
            "aspect_ratio": torch.tensor(1.0),  # "aspect_ratio"
            "idx": idx,
            "caption_type": caption_type,
            "clipscore": str(caption_clipscore),
            "fname": os.path.basename(lq_path),
            "lq_path": lq_path,
            "hq_path": hq_path,
            "conversation": meta["conversations"],
            "task_type": meta["task_type"],
            "meta": meta 
        }

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        output = {}
        for sample in batch:
            for k, v in sample.items():
                if k in output:
                    output[k].append(v)   
                else:
                    output[k] = [v]
        output["lq"] = torch.stack(output["lq"], dim=0) 
        output["hq"] = torch.stack(output["hq"], dim=0)
        return output 

class PairTrans:
    def __init__(self, t): self.t = t
    def __call__(self, pair):
        img1, img2 = pair
        return self.t(img1), self.t(img2)

class ImageRestorationMetric():
    # model cards: https://iqa-pytorch.readthedocs.io/en/latest/ModelCard.html
    def __init__(self, metrics=['psnr', 'ssim', 'lpips', 'clipiqa', 'musiq', 'maniqa', 'qalign', 'fid', 'niqe', 'nima'], nround=0, reduce = 'mean', device='cpu'):
        self.metric_objs = {}
        self.metric_vals = {}
        self.metric_results = {}
        self.nround = nround
        self.metrics = metrics
        self.device = device
        self.task = 'quality' # 'aesthetic'
        for m in metrics:
            if m == "fid":
                self.metric_objs[m] = {}
                for nr in range(nround):
                    self.metric_objs[m][nr] = FrechetInceptionDistance(feature=2048).to(device)
            else:
                self.metric_objs[m] = pyiqa.create_metric(m, device=device)
            
            self.metric_vals[m] = {}
            for nr in range(nround):
                self.metric_vals[m][nr] = []
            self.metric_results[m] = []

    def update(self, preds, gt, nround=0):
        preds = preds.to(torch.float32)
        gt = gt.to(torch.float32)
        metric_tmp = {}
        for m, obj in self.metric_objs.items():
            time1 = time.time()
            if m == 'fid':
                self.metric_objs[m][nround].update(gt.to(torch.uint8), real=True)
                self.metric_objs[m][nround].update(preds.to(torch.uint8), real=False)
                continue
            elif m in ['psnr', 'ssim', 'lpips']:  # FR
                vals = obj(preds, gt)
            elif m in ['qalign']: 
                vals = obj(preds, task_='quality')
            else:   # NR
                vals = obj(preds)
            time2 = time.time()
            self.metric_vals[m][nround].extend([v.item() for v in vals.flatten()])
            metric_tmp[m] = float(np.mean(self.metric_vals[m][nround][-1]))
        return metric_tmp

    def compute(self, nround=0):
        """Aggregate state over all processes and compute the metric."""
        # Return average loss over entire validation dataset
        if "fid" in self.metric_objs:
            self.metric_vals["fid"][nround] = [self.metric_objs["fid"][nround].compute().cpu()]
        return {m: float(np.mean(vs[nround])) if len(vs[nround]) > 0 else float("nan") for m, vs in self.metric_vals.items()}

    def results(self, ):
        return {m: float(np.mean(vs)) for m, vs in self.metric_results.items()}

    def reset(self, nround=0):
        if "fid" in self.metric_objs:
            self.metric_objs["fid"][nround].reset()
        for m in self.metric_vals.keys():
            self.metric_vals[m][nround] = []

def set_env(seed=1229):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True   # False -> True to speedup the inference time
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)

def setup_logger(name, save_dir, distributed_rank, train=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt" if train else 'log_eval.txt'), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    return parser.parse_known_args()[0]

def create_save_root(args, root, epoch_name, step_name, sample_steps, guidance_type):
    save_root = os.path.join(
        root,
        f"epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}"
        f"_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}"
        f"_seed{args.seed}",
    )
    if args.pag_scale != 1.0:
        save_root = save_root.replace(f"scale{args.cfg_scale}", f"scale{args.cfg_scale}_pagscale{args.pag_scale}")
    if flow_shift != 1.0:
        save_root += f"_flowshift{flow_shift}"
    if guidance_type != "classifier-free":
        save_root += f"_{guidance_type}"
    if args.interval_guidance[0] != 0 and args.interval_guidance[1] != 1:
        save_root += f"_intervalguidance{args.interval_guidance[0]}{args.interval_guidance[1]}"
    save_root += f"_imgnums{args.sample_nums}"
    return save_root

def guidance_type_select(default_guidance_type, pag_scale, attn_type):
    guidance_type = default_guidance_type
    if not (pag_scale > 1.0 and attn_type == "linear"):
        print("Setting back to classifier-free")
        guidance_type = "classifier-free"
    return guidance_type

def concat_horizontal(imgs):
    w, h = zip(*(i.size for i in imgs))
    total_width = sum(w)
    max_height = max(h)

    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in imgs:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width
    return new_im

@dataclass
class SanaInference(SanaConfig):
    config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"  # config
    model_path: Optional[str] = "hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth"
    work_dir: Optional[str] = None
    version: str = "sigma"
    data_dir: str = "/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/Datasets/IR" 
    meta_file: str = "Dehaze/SOTS/metas/test_iqa_A_brief_EQA_former2_ep2.json"
    tag: str = "SOTS"
    sample_nums: int = 300
    resolution: Optional[int] = None
    bs: int = 1
    cfg_scale: float = 4.5
    pag_scale: float = 1.0
    sampling_algo: str = "flow_dpm-solver"
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    seed: int = 1229
    num_workers: int = 10
    save_result: bool = False
    save_nums: int = 20
    step: int = -1
    num_rounds: int = 4
    flow_type: str = "d2c" # d2c or p2p
    mode: str = "online"   # online or offline
    assessment_model: str = "SDQA"
    assessment_config: str = "./iqa/config.yaml"
    weight_type: str = "bf16" 
    need_resize: bool = True

if __name__ == "__main__":
    # [0]: Config
    ## ======================================================================
    args = get_args()
    config = args = pyrallis.parse(config_class=SanaInference, config_path=args.config)
    args.image_size = config.model.image_size
    if args.resolution:
        args.image_size = args.resolution
    set_env(args.seed)
    
    if args.weight_type == "bf16":
        weight_type = torch.bfloat16
    elif args.weight_type == "fp16":
        weight_type = torch.float16
    elif args.weight_type == "fp32":
        weight_type = torch.float32
    else:
        raise KeyError(f"Unsupported Weight Type: {args.weight_type}")

    # only support fixed latent size currently
    latent_size = args.image_size // config.vae.vae_downsample_rate
    max_sequence_length = config.text_encoder.model_max_length
    flow_shift = config.scheduler.flow_shift
    pag_applied_layers = config.model.pag_applied_layers
    guidance_type = "classifier-free_PAG"
    assert (
        isinstance(args.interval_guidance, list)
        and len(args.interval_guidance) == 2
        and args.interval_guidance[0] <= args.interval_guidance[1]
    )
    args.interval_guidance = [max(0, args.interval_guidance[0]), min(1, args.interval_guidance[1])]
    
    # tags
    match = re.search(r".*epoch_(\d+).*step_(\d+).*", args.model_path)
    epoch_name, step_name = match.groups() if match else ("unknown", "unknown")
    dataset = args.tag
    guidance_type = guidance_type_select(guidance_type, args.pag_scale, config.model.attn_type)
    
    # Sampler Config
    args.sampling_algo = (
        args.sampling_algo
        if ("flow" not in args.model_path or args.sampling_algo == "flow_dpm-solver")
        else "flow_euler"
    )
    assert args.sampling_algo in ["flow_dpm-solver", "flow_euler"], f"Only support flow_dpm-solver and flow_euler now, but received {args.sampling_algo}."
    sample_steps_dict = {"flow_dpm-solver": 20, "flow_euler": 28} # {"dpm-solver": 20, "sa-solver": 25, "flow_dpm-solver": 20, "flow_euler": 28}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    
    # output setting
    work_dir = args.work_dir
    work_dir = os.path.join(work_dir, args.mode+"_compare")
    work_dir = os.path.join(work_dir, f"ep{epoch_name}_it{step_name}_r{args.image_size}_s{args.step}_n{args.num_rounds}_{args.flow_type}") # ep100_r256_s4_n4_p2p
    config.work_dir = work_dir
    os.umask(0o000)
    save_root = os.path.join(work_dir, args.tag) # $work_dirs/online/ep100_it32500_s4_n4_p2p/SOTS
    os.makedirs(save_root, exist_ok=True)
    save_path_dict = {"work_dir": work_dir, "root": save_root}  
    if args.save_result:
        result_img_rt = os.path.join(save_root, "images")
        os.makedirs(result_img_rt, exist_ok=True)
        save_path_dict['result'] = result_img_rt
        gt_img_rt = os.path.join(save_root, "gt")
        os.makedirs(gt_img_rt, exist_ok=True)
        save_path_dict['gt'] = gt_img_rt
        for n_round in range(args.num_rounds+1):
            sample_img_rt = os.path.join(save_root, str(n_round))
            os.makedirs(sample_img_rt, exist_ok=True)
            save_path_dict[n_round] = sample_img_rt

    # logger
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('SD35M', save_root, 0)
    logger.info("##############################################################")
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info('Running with config:\n{}'.format(config))
    logger.info('Running with args:\n{}'.format(args))
    logger.info(f"Sampler {args.sampling_algo}")
    logger.info(colored(f"Save Results: {save_root}", "blue"))
    logger.info("##############################################################")

    # [1]: model define
    ## ======================================================================
    weight_dtype = weight_type # get_weight_dtype(config.model.mixed_precision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ## [1-1]: Loading VAE ...
    vae = None
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    if not config.data.load_vae_feat:
        if config.vae.vae_type == "SDVAE":
            vae = load_vae(config.vae.vae_pretrained, device)
            vae = vae.to(vae_dtype).eval()
        else:
            raise KeyError(f"Only support VAE: 'SDVAE', but received {config.vae.vae_type}.")
    vae.to(vae_dtype)
    logger.info("##############################################################")
    logger.info(f"VAE type: {config.vae.vae_type}, path: {config.vae.vae_pretrained}, weight_dtype: {vae_dtype}")
    logger.info(f"VAE Params: {sum(p.numel() for p in vae.parameters())/1e6} M, dtype: {next(vae.parameters()).dtype}")
    logger.info("##############################################################")
        
    # [1-2]: Loading Tokenizer ...
    text_encoder = None
    logger.info("##############################################################")
    logger.info(f"text_encoder type: {config.text_encoder.text_encoder_name}, path: {config.text_encoder.text_encoder_pretrained}")
    if config.text_encoder.text_encoder_name == "sd35-text":
        text_encoder = load_text_encoder(config.text_encoder.text_encoder_pretrained, device)
    logger.info("##############################################################")
    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"

    ## [1-3]: Loading IQA model ...
    logger.info("##############################################################")
    if not args.assessment_model:
        args.assessment_model = "SDQA"
    logger.info(f"IQA type: {args.assessment_model}, config: {args.assessment_config}")
    if args.assessment_model == "SDQA":
        from iqa import DepictQA, load_pretrained_weights
        assert os.path.isfile(args.assessment_config)
        ## loading cfg
        with open(args.assessment_config, "r") as f:
            iqa_cfg = EasyDict(yaml.safe_load(f))
        ## Model
        assessment = DepictQA(iqa_cfg, training=False)
        assessment = load_pretrained_weights(iqa_cfg, assessment, logger=None)
    assessment.eval().to(weight_dtype).to(device)
    logger.propagate = False
    logger.info(f"IQA Params: {sum(p.numel() for p in assessment.parameters())/1e6} M, dtype: {next(assessment.parameters()).dtype}")
    logger.info("##############################################################")

    ## [1-4]: Loading Connector model ...
    connector_dtype = torch.float32
    logger.info("##############################################################")
    logger.info(f"Connector type: {config.connector.model}, path: {config.connector.model_pretrained}, weight_dtype: {connector_dtype}")
    if config.connector.model == "QFormer":
        from diffusion.model.qa_connector import QFormer
        connector = QFormer(
            hidden_dim = config.connector.hidden_dim,
            layers = config.connector.layers,
            heads = config.connector.heads
        )
    else:
        raise KeyError("Unknown Connector Type: only support [QFormer], but recieve {config.connector.model}.")
    if config.connector.load_from:
        logger.info(f"Loading Pre-trained Weight for Connector: {config.connector.load_from}")
        state_dict = torch.load(config.connector.load_from, map_location='cpu')
        missing, unexpected = connector.load_state_dict(state_dict["state_dict"], strict=False)
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")
    connector = connector.eval().to(weight_dtype).to(device)
    logger.info(f"Connector Params: {sum(p.numel() for p in connector.parameters())/1e6} M, dtype: {next(connector.parameters()).dtype}")
    logger.info("##############################################################")

    # [1-5]: Loading DiT model ...
    if config.model.model == "SD35M_P2P":
        assert args.flow_type == "p2p", f"Error: Model {config.model.model} only support 'p2p' mode."
        from diffusion.model.sd35 import load_mmdit_p2p
        DiT = load_mmdit_p2p(
                           config.model.model_pretrained, 
                           config.model.shift, 
                           False, 
                           device, 
                           config.model.image_size, 
                           config.model.input_channel,
                        ).eval().to(device)
    elif config.model.model == "SD35M_D2C":
        assert args.flow_type == "d2c", f"Error: Model {config.model.model} only support 'd2c' mode."
        from diffusion.model.sd35 import load_mmdit
        DiT = load_mmdit(
                           config.model.model_pretrained, 
                           config.model.shift, 
                           False, 
                           device, 
                           config.model.image_size, 
                           config.model.input_channel,
                        ).eval().to(device)
    else:
        raise KeyError(f"Only support Model: 'SD35M_P2P' or 'SD35M_D2C', but received {config.model.model}.")
     ## Load model
    state_dict = torch.load(config.model.load_from)
    if config.model.load_from.endswith(".bin"):
        logger.info("Loading fsdp bin checkpoint....")
        old_state_dict = state_dict
        state_dict = dict()
        state_dict["state_dict"] = old_state_dict
    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]
    missing, unexpected = DiT.load_state_dict(state_dict["state_dict"], strict=False)
    DiT.eval().to(weight_dtype)
    dit_dtype = weight_dtype
    logger.info("##############################################################")
    logger.info("# % Model Define .....  ")
    logger.info(f"Inference with {weight_dtype}, default guidance_type: {guidance_type}, flow_shift: {flow_shift}")
    logger.info(f"{DiT.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in DiT.parameters()):,}")
    logger.info("Generating sample from ckpt: %s" % config.model.load_from)
    logger.warning(f"Missing keys: {missing}")
    logger.warning(f"Unexpected keys: {unexpected}")
    logger.info(f"Parameter of DiT: {sum(p.numel() for p in DiT.parameters()) / 1000000} M")
    logger.info("##############################################################")

    # [1-6]: Combination Model
    model = IQAIR(vae, DiT, connector, assessment, device, latent_size, config)
    logger.info("##############################################################")
    logger.info("Summary: IQAIR")
    for param in model.parameters():
        param.requires_grad = False
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"All params: {round(num_total_params/1e6, 3)}M")
    logger.info("##############################################################")

    ## Load model
    if os.path.isfile(args.model_path):
        state_dict = torch.load(args.model_path)
        if args.model_path.endswith(".bin"):
            logger.info("Loading fsdp bin checkpoint....")
            old_state_dict = state_dict
            state_dict = dict()
            state_dict["state_dict"] = old_state_dict
        if "pos_embed" in state_dict["state_dict"]:
            del state_dict["state_dict"]["pos_embed"]
        missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
        model.eval().to(weight_dtype)
        dit_dtype = weight_dtype
        logger.info("##############################################################")
        logger.info("# % Model Define .....  ")
        logger.info(f"Inference with {weight_dtype}, default guidance_type: {guidance_type}, flow_shift: {flow_shift}")
        logger.info(f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info("Generating sample from ckpt: %s" % args.model_path)
        missing_ckpt = []
        for m in missing:
            if "llm" in m and "lora" not in m:
                continue
            missing_ckpt.append(m)
        logger.warning(f"Missing keys: {missing_ckpt}")
        logger.warning(f"Unexpected keys: {unexpected}")
        logger.info(f"Parameter of Model: {sum(p.numel() for p in model.parameters()) / 1000000} M")
        logger.info("##############################################################")
    else:
        logger.info("##############################################################")
        logger.info("Combination Model is inference from pre-trained weight!")
        logger.info("##############################################################")

    # [2]: dataloader
    ## ======================================================================
    logger.info("##############################################################")
    logger.info("# % Dataloader .....  ")
    dataset = ValP2PDataset(
                data_dir=args.data_dir,
                meta_file=args.meta_file,
                resolution=args.resolution,
                max_length=max_sequence_length,
                img_extension=".png",
                max_samples=args.sample_nums,
                caption_type = "question", # "question", "GT", "IQA"
                need_resize=args.need_resize,
            )
    loader = data.DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )
    logger.info("Datalist: {}".format(args.meta_file))
    logger.info("Samples: {}".format(len(dataset)))
    logger.info("##############################################################")

    # [3]: Inference
    # generator 
    # metric
    # IR_metric = ImageRestorationMetric(metrics=['psnr', 'ssim', 'lpips', 'clipiqa', 'musiq', 'maniqa', 'fid', 'niqe', 'nima'], nround=args.num_rounds+1, reduce = 'mean', device=device)
    # sampling (args.flow_type {d2c or p2p}, args.mode {online or offline})
    # sample_rate = math.ceil(len(loader)/args.save_nums)  
    # dit_step = args.step
    # num_rounds = args.num_rounds 
    # save_idx = 0

    for bidx, batch in enumerate(tqdm(loader)):
        """
        batch: "lq", "hq", "txt_fea", "attention_mask", "img_hw", "aspect_ratio", "idx", "caption_type", "clipscore"
               "fname", "lq_path", "hq_path", "conversation", "task_type", "meta"
        """
        print("="*20)
        # start sampling
        with torch.no_grad():
            recon = model(batch["lq"].to(device))

    print("="*100)
    print("Summary (GFLOPs):")
    for module_name, gflops in model.gflop_dict.items():
        print(f"{module_name}: {gflops/1e9}, average: {gflops/1e9/len(loader)}")

    print("="*100)
    print("Summary (Times(s)):")
    for module_name, times in model.time_dict.items():
        print(f"{module_name}: {times}, average: {times/len(loader)}")
    logger.info("====================================================================================================")
