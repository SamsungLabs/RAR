import datetime
import math
import os
import pickle
import re
import time 

import fire
import numpy as np
import .sd3_impls as sd3_impls
import torch
from .text_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from safetensors import safe_open
from tqdm import tqdm

def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e

CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}

class TextEncoder:
    @torch.no_grad()
    def __init__(self, model_folder, text_encoder_device):
        self.device = text_encoder_device
        self.verbose = False
        
        print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer()

        print("Loading Google T5-v1-XXL...")
        with safe_open(f"{model_folder}/t5xxl_fp16.safetensors", framework="pt", device="cpu") as f:
            self.t5xxl = T5XXLModel(T5_CONFIG, device=self.device, dtype=torch.float32)
            load_into(f, self.t5xxl.transformer, "", self.device, torch.float32)

        print("Loading OpenAI CLIP L...")
        with safe_open(f"{model_folder}/clip_l.safetensors", framework="pt", device="cpu") as f:
            self.clip_l = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device=self.device,
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.clip_l.transformer, "", self.device, torch.float32)

        print("Loading OpenCLIP bigG...")
        with safe_open(f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu") as f:
            self.clip_g = SDXLClipG(CLIPG_CONFIG, device=self.device, dtype=torch.float32)
            load_into(f, self.clip_g.transformer, "", self.device, torch.float32)

        print("="*50)
        print(f"Text Encoder:")
        print(f"T5XXL: params: {sum(p.numel() for p in self.t5xxl.parameters())/1e6} M, dtype: {next(self.t5xxl.parameters()).dtype}")
        print(f"CLIP-L: params: {sum(p.numel() for p in self.clip_l.parameters())/1e6} M, dtype: {next(self.clip_l.parameters()).dtype}")
        print(f"CLIP-G: params: {sum(p.numel() for p in self.clip_g.parameters())/1e6} M, dtype: {next(self.clip_g.parameters()).dtype}")
        print("="*50)

    @torch.no_grad()
    def forward(self, prompts):
        # print("Encode prompt... ", len(prompts))
        tokens = self.tokenizer.tokenize(prompts)
        l_out, l_pooled = self.clip_l.encode_tokens(tokens["l"])
        g_out, g_pooled = self.clip_g.encode_tokens(tokens["g"])
        t5_out, _ = self.t5xxl.encode_tokens(tokens["t5xxl"])
        return {
            "g": (g_out, g_pooled),
            "l": (l_out, l_pooled),
            "t5xxl": (t5_out, None)
        }

    @torch.no_grad()
    def get_cond(self, prompts):
        # print("Encode prompt... ", len(prompts))
        tokens = self.tokenizer.tokenize(prompts)
        l_out, l_pooled = self.clip_l.encode_tokens(tokens["l"])
        g_out, g_pooled = self.clip_g.encode_tokens(tokens["g"])
        t5_out, _ = self.t5xxl.encode_tokens(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        context = torch.cat([lg_out, t5_out], dim=-2)
        y = torch.cat((l_pooled, g_pooled), dim=-1)
        return context.to(self.device), y.to(self.device)

if __name__ == "__main__":
    MODEL_FOLDER = "/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/ModelZoo/stable-diffusion-3.5-medium/text_encoders"
    PROMPT = [
        "a photo of a cat",
    ]
    extra_prompt = "Sunrise beach, shallow tide washing over smooth sand. A piece of weathered driftwood lies near the shoreline with a subtle branded text [SOS] on its surface; wet sand reflections, micro-ripples, sun flare at horizon."
    # PROMPT = "a photo of a cat"
    txt_enc = TextEncoder(MODEL_FOLDER, "cuda")
    for i in range(1):
        PROMPT.append(extra_prompt)
    context, y = txt_enc.get_cond(PROMPT)
    print(context.shape, y.shape)