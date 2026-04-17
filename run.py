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
CUDA_VISIBLE_DEVICES=0 streamlit run run.py
** change the configuration in RARInference if needed **
    work_dir: save_path 
    mode: online/offline inference 
    step: number of step for DiT
    num_rounds: maxinum round for RAR 
    detail: True (show every step), False (show only final results)
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
from PIL import Image, ImageOps
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
from diffusion.model.sd35 import load_scheduler, load_vae, load_mmdit, load_text_encoder
import torchvision.transforms.functional as F 
from torchvision.transforms import InterpolationMode
from diffusion.model.utils import set_fp32_attention, set_grad_checkpoint
# Interface
import streamlit as st

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
        "Determine the leading ONE degradation in the evaluated image. Answer the question using a single word or phrase.",
        "Determine the most impactful ONE distortion in the evaluated image. Answer the question using a single word or phrase.",
        "Highlight the most significant ONE distortion in the evaluated image. Answer the question using a single word or phrase.",
        "Identify the chief ONE degradation in the evaluated image. Answer the question using a single word or phrase.",
        "Identify the most critical ONE distortion in the evaluated image. Answer the question using a single word or phrase.",
        "Identify the most notable ONE distortion in the evaluated image's quality. Answer the question using a single word or phrase.",
        "In terms of image quality, what is the most glaring ONE issue with the evaluated image? Answer the question using a single word or phrase.",
        "In the evaluated image, what ONE distortion is most detrimental to image quality? Answer the question using a single word or phrase.",
        "Pinpoint the foremost ONE image quality issue in the evaluated image. Answer the question using a single word or phrase.",
        "What ONE distortion is most apparent in the evaluated image? Answer the question using a single word or phrase.",
        "What ONE distortion is most evident in the evaluated image? Answer the question using a single word or phrase.",
        "What ONE distortion is most prominent in the evaluated image? Answer the question using a single word or phrase.",
        "What ONE distortion is most prominent when examining the evaluated image? Answer the question using a single word or phrase.",
        "What ONE distortion most detrimentally affects the overall quality of the evaluated image? Answer the question using a single word or phrase.",
        "What ONE distortion most notably affects the clarity of the evaluated image? Answer the question using a single word or phrase.",
        "What ONE distortion most significantly affects the evaluated image? Answer the question using a single word or phrase.",
        "What ONE distortion stands out in the evaluated image? Answer the question using a single word or phrase.",
        "What ONE quality degradation is most apparent in the evaluated image? Answer the question using a single word or phrase.",
        "What critical ONE quality degradation is present in the evaluated image? Answer the question using a single word or phrase.",
        "What is the foremost ONE distortion affecting the evaluated image's quality? Answer the question using a single word or phrase.",
        "What is the leading ONE distortion in the evaluated image? Answer the question using a single word or phrase.",
        "What is the most critical ONE image quality issue in the evaluated image? Answer the question using a single word or phrase.",
        "What is the most severe ONE degradation observed in the evaluated image? Answer the question using a single word or phrase.",
        "What is the primary ONE degradation observed in the evaluated image? Answer the question using a single word or phrase.",
    ],
}

def question_generate(task="Quality Comparison"):
    # template = random.choice(question_dict[task])
    template = question_dict[task][-1]
    return template

#### Model
class IQAIR(nn.Module):
    def __init__(self, model, connector, assessment=None, device='cuda'):
        super().__init__()
        self.model = model.to(device).eval()
        self.connector = connector.to(device).eval()
        if assessment:
            self.assessment = assessment.to(device).eval()
        else:
            self.assessment = None

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


#### data process
def image_process(img):
    resolution = 256
    transform = T.Compose([
                T.Resize((resolution, resolution)),  # Image.BICUBIC
                T.CenterCrop(resolution),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ])
    # lq = Image.open(image_path).convert("RGB")
    lq = transform(img)
    return lq.unsqueeze(0)

class PairTrans:
    def __init__(self, t): self.t = t
    def __call__(self, pair):
        img1, img2 = pair
        return self.t(img1), self.t(img2)

#### Configure
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

def guidance_type_select(default_guidance_type, pag_scale, attn_type):
    guidance_type = default_guidance_type
    if not (pag_scale > 1.0 and attn_type == "linear"):
        print("Setting back to classifier-free")
        guidance_type = "classifier-free"
    return guidance_type

@dataclass
class RARInference(SanaConfig):
    config: Optional[str] = "./configs/infer_cfg.yaml"  # config
    model_path: Optional[str] = "./checkpoints/RAR_modelzoo/RAR/checkpoints/latest.pth"
    sentence_model: str ="./checkpoints/RAR_modelzoo/all-MiniLM-L6-v2",
    work_dir: Optional[str] = "tmp/"
    version: str = "output_composite/"
    resolution: Optional[int] = 256
    bs: int = 1
    cfg_scale: float = 1.0
    pag_scale: float = 1.0
    sampling_algo: str = "flow_euler"
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    seed: int = 1229
    num_workers: int = 10
    detail: bool = True
    step: int = 4
    num_rounds: int = 4
    flow_type: str = "d2c" # d2c or p2p
    mode: str = "online"   # online or offline
    assessment_model: str = "SDQA"
    assessment_config: str = "./iqa/config.yaml"
    weight_type: str = "bf16" 
    need_resize: bool = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    return parser.parse_known_args()[0]

@st.cache_resource
def load_rar_model():
    # [0]: Config
    ## ======================================================================
    # args = get_args()
    config = args = pyrallis.parse(config_class=RARInference, config_path="./configs/infer_cfg.yaml")
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
    # work_dir = os.path.join(work_dir, f"ep{epoch_name}_it{step_name}_r{args.image_size}_s{args.step}_n{args.num_rounds}_{args.flow_type}") # ep100_r256_s4_n4_p2p
    config.work_dir = work_dir
    os.umask(0o000)
    save_root = work_dir # $work_dirs/online/ep100_it32500_s4_n4_p2p/SOTS
    os.makedirs(work_dir, exist_ok=True)
    save_detail = args.detail

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
    # if config.connector.load_from:
        # logger.info(f"Loading Pre-trained Weight for Connector: {config.connector.load_from}")
        # state_dict = torch.load(config.connector.load_from, map_location='cpu')
        # missing, unexpected = connector.load_state_dict(state_dict["state_dict"], strict=False)
        # logger.warning(f"Missing keys: {missing}")
        # logger.warning(f"Unexpected keys: {unexpected}")
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
    state_dict = torch.load(config.model.load_from, map_location="cpu", weights_only=False)

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
    model = IQAIR(DiT, connector, assessment, device)
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

    print("\n##############################################################")
    print("% RAR Model loaded successfully!")
    print("% Please 'Upload' your image or 'Select' sample to start the RAR process!")
    print("##############################################################\n")
    return model, vae, args, config, device

# [2]: Inference
def RAR_process(img, fname, model, vae, args, config, device):
    lq = image_process(img)
    lq = lq.to(device)
    bs = lq.shape[0]
    latent_size = args.image_size // config.vae.vae_downsample_rate
    latent_size_h, latent_size_w = latent_size, latent_size
    sentence_model = config.sentence_model

    save_detail = args.detail
    save_root = config.work_dir
    save_folder = os.path.join(save_root, fname)
    os.makedirs(save_folder, exist_ok=True)
    restore_results = [] 
    restore_process = []
    flag = 0
    # generator 
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sample_steps = args.step
    num_rounds = args.num_rounds 
    print("="*20)

    if args.weight_type == "bf16":
        weight_dtype = torch.bfloat16
    elif args.weight_type == "fp16":
        weight_dtype = torch.float16
    elif args.weight_type == "fp32":
        weight_dtype = torch.float32
    else:
        raise KeyError(f"Unsupported Weight Type: {args.weight_type}")
    dit_dtype = weight_dtype
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    # start sampling
    with torch.no_grad():
        ## [a1]: VAE
        input_images = vae.encode(lq.to(device)).to(device)
        input_images = vae.process_in(input_images).to(device)
        lq = torch.clamp((lq + 1.0) / 2.0, min=0.0, max=1.0)
        save_input_path = os.path.join(save_folder, "%s_input.png"%(fname.split(".")[0]))
        save_input = 255.0 * rearrange(lq[0], "c h w -> h w c")
        save_input = Image.fromarray(save_input.type(torch.uint8).cpu().numpy())
        save_input.save(save_input_path)

        ## [a2]: predefined
        pred_latent, prefix = None, None
        prompts = [""]
        samples = input_images.to(weight_dtype)

        ## [a3]: RAR process in shared latent space
        for n_round in range(num_rounds+1):
            ## [a3-1]: Quality Analysis
            output_texts, _, _, confidences, output_pred_latent, output_prefix = model.assessment.generate(   
                {
                    "query": [question_generate("Distortion Identification")],
                    "img": samples.to(device),
                    "img_A": samples.to(device),
                    "img_B": [None],
                    "img_path": ["input"],
                    "img_A_path": ["input"],
                    "img_B_path": [None],
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "max_new_tokens": 400,
                    "task_type": "quality_single_A_noref",
                    "output_prob_id": False,
                    "output_confidence": True, 
                    "sentence_model": sentence_model,
                },
                latent_input=True,
                save_hidden=True,
            )
            prompts = [p.replace("\n ", "") for p in output_texts]
            prompts = [p.replace("Snow", "Rain") for p in prompts]
            pred_latent, prefix = output_pred_latent, output_prefix
            restore_process.append(prompts[0])

            ## [a3-2]: Connector (Token Condition)
            print("round:", n_round, " prompt:", prompts)
            last_token, mask_token = token_pad_or_truncate(torch.cat([prefix, pred_latent], dim=1)) # torch.cat([prefix, pred_latent], dim=1)
            pred_tokens = model.connector(last_token, key_padding_mask=mask_token)
            pred_context, pred_y = mapping_to_cond(pred_tokens) # (64, 154, 4096), (64, 1, 2048)
            cond, pooled = (pred_context.to(dit_dtype).to(device), pred_y.to(dit_dtype).to(device))
            caption_cond = {"c_crossattn": cond, "y": pooled}
            null_caption_cond = {"c_crossattn": torch.zeros(pred_context.shape).to(dit_dtype).to(device), "y": torch.zeros(pred_y.shape).to(dit_dtype).to(device)}
            ## DiT
            if args.flow_type == "d2c":
                z = input_images
            else:
                z = torch.randn(
                    bs,
                    config.vae.vae_latent_dim,
                    latent_size,
                    latent_size,
                    device=device,
                    generator=generator,
                )
            model_kwargs = dict(img_cond=input_images.to(dit_dtype), ) # model_kwargs=dict(img_cond=input_images, y=text_cond["y"], context=text_cond["c_crossattn"])
            if args.sampling_algo == "flow_dpm-solver":
                dpm_solver = DPMS(
                    model.model,
                    condition=caption_cond,
                    uncondition=null_caption_cond,
                    guidance_type=guidance_type,
                    cfg_scale=args.cfg_scale,
                    pag_scale=args.pag_scale,
                    pag_applied_layers=config.model.pag_applied_layers,
                    model_type="flow",
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                    interval_guidance=args.interval_guidance,
                )
                samples = dpm_solver.sample(
                    z.to(dit_dtype),
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=config.scheduler.flow_shift,
                )
            elif args.sampling_algo == "flow_euler": 
                flow_solver = FlowEuler(
                    model.model,
                    condition=caption_cond,
                    uncondition=null_caption_cond,
                    cfg_scale=args.cfg_scale,
                    model_kwargs=model_kwargs,
                )
                samples = flow_solver.sample(
                    z,
                    steps=sample_steps,
                )
            
            # [a3-3]: quality comparison
            output_quality, _, _, confid_quality = model.assessment.generate(   # texts, output_ids, probs, confidences, 
                {
                    "query": [question_generate("Quality Comparison")],
                    "img": [None],
                    "img_A": samples.to(device),       # current result
                    "img_B": input_images.to(device),  # previous status
                    "img_path": [None],
                    "img_A_path": ["input"],
                    "img_B_path": ["input"],
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "max_new_tokens": 400,
                    "task_type": "quality_compare_noref",
                    "output_prob_id": False,
                    "output_confidence": True, 
                    "sentence_model": sentence_model,
                },
                latent_input=True,
                save_hidden=False,
            )
            # output_quality = ["Image A"]

            if "Image B" in output_quality[0] or n_round == num_rounds:
                flag = 1

            # [3d]: decode & save
            if flag == 1 and not save_detail:
                recon = vae.process_out(input_images.to(vae_dtype).to(device)).to(device)
                recon = vae.decode(recon)
                recon = torch.clamp((recon + 1.0) / 2.0, min=0.0, max=1.0)
                # save recon image
                save_recon_path = os.path.join(save_folder, "%s_step%d_%s.png"%(fname.split(".")[0], n_round, prompts[0]))
                save_recon = 255.0 * rearrange(recon[0], "c h w -> h w c")
                save_recon = Image.fromarray(save_recon.type(torch.uint8).cpu().numpy())
                save_recon.save(save_recon_path)
                restore_results.append(save_recon)
                return restore_results, restore_process
            elif save_detail:
                recon = vae.process_out(samples.to(vae_dtype).to(device)).to(device)
                recon = vae.decode(recon)
                recon = torch.clamp((recon + 1.0) / 2.0, min=0.0, max=1.0)
                # save recon image
                save_recon_path = os.path.join(save_folder, "%s_step%d_%s.png"%(fname.split(".")[0], n_round, prompts[0]))
                save_recon = 255.0 * rearrange(recon[0], "c h w -> h w c")
                save_recon = Image.fromarray(save_recon.type(torch.uint8).cpu().numpy())
                save_recon.save(save_recon_path)
                restore_results.append(save_recon)
                if flag == 1:
                    return restore_results, restore_process
            # prepare for next round
            input_images = samples 
            torch.cuda.empty_cache()

model, vae, args, config, device = load_rar_model()

# ==========================================================================================================
PREDEFINED_IMAGES = {  
    "Composite1":  { "input": "demo_sample/.default/Composite1/input.png",
                     "output": ["demo_sample/.default/Composite1/RAR_s1_denoise.png", 
                                "demo_sample/.default/Composite1/RAR_s2_dehaze.png",
                                "demo_sample/.default/Composite1/RAR_s3_deblur.png"],
                     "process": ["Noise", "Haze", "Blur"]},
    "Composite2":  { "input": "demo_sample/.default/Composite2/input.png",
                     "output": ["demo_sample/.default/Composite2/RAR_s1_defog.png", 
                                "demo_sample/.default/Composite2/RAR_s2_deblur.png",
                                "demo_sample/.default/Composite2/RAR_s3_sr.png"],
                     "process": ["Haze", "Blur", "Low-Resolution"]},
    "Composite3":  { "input": "demo_sample/.default/Composite3/input.png",
                     "output": ["demo_sample/.default/Composite3/RAR_s1_dehaze.png", 
                                "demo_sample/.default/Composite3/RAR_s2_deblur.png",
                                "demo_sample/.default/Composite3/RAR_s3_sr.png"],
                     "process": ["Haze", "Blur", "Low-Resolution"]},
    "Composite4":  { "input": "demo_sample/.default/Composite4/input.png",
                     "output": ["demo_sample/.default/Composite4/RAR_s1_derain.png", 
                                "demo_sample/.default/Composite4/RAR_s2_sr.png",
                                "demo_sample/.default/Composite4/RAR_s3_defog.png"],
                     "process": ["Rain", "Low-Resolution", "Haze"]},
    "Composite5":  { "input": "demo_sample/.default/Composite5/input.png",
                     "output": ["demo_sample/.default/Composite5/RAR_s1_derain.png", 
                                "demo_sample/.default/Composite5/RAR_s2_deblur.png",
                                "demo_sample/.default/Composite5/RAR_s3_sr.png"],
                     "process": ["Rain", "Blur", "Low-Resolution"]},
    "Underwater1":  { "input": "demo_sample/.default/Underwater1/input.jpg",
                     "output": ["demo_sample/.default/Underwater1/RAR_s1_deblur.jpg", 
                                "demo_sample/.default/Underwater1/RAR_s2_sr.jpg"],
                     "process": ["Blur", "Low-Resolution"]},
    "Underwater2":  { "input": "demo_sample/.default/Underwater2/input.jpg",
                     "output": ["demo_sample/.default/Underwater2/RAR_s1_deblur.jpg", 
                                "demo_sample/.default/Underwater2/RAR_s2_saturation.jpg"],
                     "process": ["Blur", "Saturation"]},                    
    "Prague":       {"input": "demo_sample/.default/Prague/Prague.jpg",    "output": None}
}

st.set_page_config(page_title="RAR Demo", layout="wide", initial_sidebar_state='expanded')
st.title("RAR Demo")
# =============================
# Sidebar – Input Panel
# =============================
st.sidebar.header("Input Options")

upload_file = st.sidebar.file_uploader(
    "Upload an image", 
    type=["png", "jpg", "jpeg"]
)

selected_name = st.sidebar.selectbox(
    "Or choose a predefined image",
    ["Upload"] + list(PREDEFINED_IMAGES.keys())
)

# Load image based on user choice
input_image = None
if selected_name == "Upload" and upload_file is not None:
    filename = upload_file.name
    input_image = Image.open(upload_file)
    input_image = ImageOps.exif_transpose(input_image)
    # input_image = input_image.resize((256, 256))
    print(filename)
elif selected_name != "Upload":
    # input_image = Image.open(os.path.join(root_dir, PREDEFINED_IMAGES[selected_name][0]))
    filename = os.path.basename(PREDEFINED_IMAGES[selected_name]["input"])
    input_image = Image.open(PREDEFINED_IMAGES[selected_name]["input"])
    input_image = ImageOps.exif_transpose(input_image)
    # input_image = input_image.resize((256, 256))
    try:
        if selected_name == "Prague" and st.session_state.previous_select != "Prague":
            st.session_state.cached_filename = None
    except:
        pass
    st.session_state.previous_select = selected_name
    print(filename)

# =============================
# Image Processing Section
# =============================
def process_image(image: Image.Image, selected_name: str, filename: str):
    """
    ---------------------------------------------------------
    PLACE YOUR IMAGE PROCESSING CODE HERE.
    The input is a PIL Image, and you should return a PIL Image.
    ---------------------------------------------------------

    For now, we simply return the original image.
    """

    output_images = [image]
    output_process = ["Input"]
    # if selected_name in ["Rain1", "Lowlight1", "Lowlight2", "Lowlight3", "Noise1", "Noise2", "Underwater1", "Underwater2"]:
    if selected_name in ["Composite1", "Composite2", "Composite3", "Composite4", "Composite5", "Underwater1", "Underwater2"]:
        for idx, image_path in enumerate(PREDEFINED_IMAGES[selected_name]["output"]):
            output_images.append(Image.open(image_path))
            output_process.append(PREDEFINED_IMAGES[selected_name]["process"][idx])
    else:
        if "cached_filename" in st.session_state and st.session_state.cached_filename == filename:
            return st.session_state.cached_results, st.session_state.cached_process

        restored_images, restored_process = RAR_process(image, filename, model, vae, args, config, device)
        output_images += restored_images
        output_process += restored_process

        st.session_state.cached_filename = filename 
        st.session_state.cached_results = output_images 
        st.session_state.cached_process = output_process

    # Example placeholder: return the input image unchanged
    return output_images, output_process


# =============================
# Main Layout – Columns
# =============================
left_col, right_col = st.columns(2)

# =============================
# Input Image Panel
# =============================
with left_col:
    st.subheader("Input Image")
    if input_image:
        st.image(input_image.resize((256, 256))) # , use_column_width=False)
    else:
        st.info("Please upload or select an image from the left sidebar.")

# =============================
# Output Image Panel
# =============================
with right_col:
    st.subheader("Processed Output")
    if input_image:
        img_results, img_process = process_image(input_image, selected_name, filename)
        stage_idx = st.session_state.get("stage_idx", len(img_results)-1) if st.session_state.get("stage_idx", len(img_results)-1) < len(img_results)  else len(img_results)-1
        st.image(img_results[stage_idx].resize((256,256)))
        st.write(f"After **{img_process[stage_idx]}** restoration..." if img_process[stage_idx] != "Input" else  "Input Image")

        # slider to select stage
        stage_idx = st.slider(
            "*# Select Processing Stage*",
            min_value=0,
            max_value=len(img_results)-1,
            value=stage_idx,
            step=1,
            key="stage_idx"
        )
    else:
        st.info("No processed image to display yet.")
