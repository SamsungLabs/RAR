import datetime
import math
import os, sys
import pickle
import re

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from .load_encoder import load_encoder

from tqdm import tqdm
from torchvision.utils import save_image
import yaml

import hashlib
import urllib
import warnings
from typing import List, Union

from tqdm import tqdm

from omegaconf import OmegaConf
import torch
from einops import rearrange
import random

def load_vision_encoder(
    configs: str,
    training: bool,
    vision_preprocess: dict,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32
):
    if not configs or not os.path.isfile(configs):
        raise KeyError(f"Error: the configs of {configs} for AutoencoderDC is not exisiting.")

    # load pre-trained model
    with open(configs, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vae_cfg = cfg["vae"]    
    model = load_encoder(vae_cfg["vae_pretrained"], device=device, dtype=dtype)
    return model

def build_abstractor(abstractor_type, pos, layers):
    if abstractor_type == 'simple':
        from .abstractor import SimpleAbstractorModel
        abstractor = SimpleAbstractorModel(pos = pos)
    elif abstractor_type == 'former':
        from .abstractor import FormerAbstractorModel
        abstractor = FormerAbstractorModel(layers=layers)
    else:
        raise KeyError(f"Error: abstractor must be [simple, former], but receive {abstractor_type}.")
    return abstractor

