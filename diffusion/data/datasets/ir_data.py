# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma

"""
meta data format:
    {
        "distortion_class": "Blur",
        "distortion_name": "Blur",
        "severity": 3,
        "id": "000001.png",
        "image_ref": "Deblur/GoPro/test/GOPR0384_11_00/sharp/000001.png",
        "image_A": "Deblur/GoPro/test/GOPR0384_11_00/blur/000001.png",
        "image_B": null,
        "task_type": "quality_single_A_noref",
        "conversations": [
            {
                "from": "human",
                "value": "What distortion(s) most detrimentally affect the overall quality of the evaluated image? Answer the question using a single word or phrase."
            },
            {
                "from": "gpt",
                "value": "Blur"
            },
            {
                "from": "qa",
                "value": "Blur\n ",
                "output_id": null,
                "prob": null,
                "confidence": 0.9658203125
            }
        ]
    },
"""

import getpass
import json
import os
import os.path as osp
import random
import warnings 
from typing import Optional, Iterable

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset, Sampler

from diffusion.data.builder import DATASETS, get_data_path
from diffusion.data.wids import lru_json_load
from diffusion.utils.logger import get_root_logger
import torchvision.transforms as T
import torchvision.transforms.functional as F 
from torchvision.transforms import InterpolationMode
from PIL import Image
from diffusion.data.corruption_unirestore import corrupt, init_corruption_function

data_tag = "SD35M_ep1_wstatus"  # option: EQA_former2_ep2, SD35M_ep1_wstatus, SD15_former2_ep2_wstatus

train_brief_data_weight = [
    [f"Denoise/SIDD/metas/train_iqa_A_brief_{data_tag}.json", 5],                                    # 320 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_Noise_L1_iqa_A_brief_{data_tag}.json", 1],       # 800 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_Noise_L3_iqa_A_brief_{data_tag}.json", 1],       # 800 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_Noise_L5_iqa_A_brief_{data_tag}.json", 1],       # 800 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_Noise_L1_iqa_A_brief_{data_tag}.json", 1], # 2650 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_Noise_L3_iqa_A_brief_{data_tag}.json", 1], # 2650 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_Noise_L5_iqa_A_brief_{data_tag}.json", 1], # 2650 pairs
    [f"Deblur/GoPro/metas/train_iqa_A_brief_{data_tag}.json", 5],                                    # 2103 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_SR1_iqa_A_brief_{data_tag}.json", 1],            # 800 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_SR2_iqa_A_brief_{data_tag}.json", 1],            # 800 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_SR3_iqa_A_brief_{data_tag}.json", 1],            # 800 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_SR1_iqa_A_brief_{data_tag}.json", 1],      # 2650 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_SR2_iqa_A_brief_{data_tag}.json", 1],      # 2650 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_SR3_iqa_A_brief_{data_tag}.json", 1],      # 2650 pairs
    [f"LowLight/LOL/metas/train_iqa_A_brief_{data_tag}.json", 15],                                   # 485 pairs
    [f"Dehaze/OTS/metas/train_iqa_A_brief_{data_tag}.json", 0.2],                                    # 72135 pairs
    [f"Derain/RainTrainL/metas/train_iqa_A_brief_{data_tag}.json", 40],                              # 200 pairs
    [f"Derain/RainDrop/metas/Raindrop_train_iqa_A_brief_{data_tag}.json", 10],                       # 861 pairs
]  # [[path, weight], ...], noise: 11950, blur: 10515, sr: 10350, ll: 7275, haze: 14427, rain: 8000, raindrop: 8610

train_brief_composite_weight = [
    [f"Composite/CDD11/metas/train_iqa_A_brief_SD35M_ep2_wstatus.json", 1],   # 13013 pairs
    [f"Composite/MiO100/metas/train_iqa_A_brief_SD35M_ep1_wstatus.json", 10], # 160 pairs
]  # [[path, weight], ...], 

train_brief_all_weight = train_brief_data_weight + train_brief_composite_weight

test_brief_data_weight = [
    [f"Denoise/SIDD/metas/test_iqa_A_brief_{data_tag}.json", 1],                 # 1280 pairs
    [f"Denoise/Kodak/metas/Kodak_Noise_L1_iqa_A_brief_{data_tag}.json", 1],      # 24 pairs
    [f"Denoise/Kodak/metas/Kodak_Noise_L3_iqa_A_brief_{data_tag}.json", 1],      # 24 pairs
    [f"Denoise/Kodak/metas/Kodak_Noise_L5_iqa_A_brief_{data_tag}.json", 1],      # 24 pairs
    [f"Deblur/GoPro/metas/test_iqa_A_brief_{data_tag}.json", 1],                 # 2103 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_valid_pair_SR_iqa_A_brief_{data_tag}.json", 1],  # 100 pairs
    [f"LowLight/LOL/metas/test_iqa_A_brief_{data_tag}.json", 1],                 # 15 pairs
    [f"Dehaze/SOTS/metas/test_iqa_A_brief_{data_tag}.json", 1],                  # 500 pairs
    [f"Derain/Rain100L/metas/test_iqa_A_brief_{data_tag}.json", 1],              # 100 pairs
    [f"Derain/RainDrop/metas/Raindrop_test_a_iqa_A_brief_{data_tag}.json", 1],   # 58 pairs
    [f"Derain/RainDrop/metas/Raindrop_test_b_iqa_A_brief_{data_tag}.json", 1],   # 249 pairs
    [f"Other/UDC/metas/test_iqa_A_brief_{data_tag}.json", 1],                    # 60 pairs
    [f"Other/UDC/metas/val_iqa_A_brief_{data_tag}.json", 1],                     # 60 pairs
    [f"Desnow/Snow100k/metas/test_M_iqa_A_brief_{data_tag}.json", 1],            # 16588 pairs
    # ["Desnow/Snow100k/metas/test_realistic_iqa_A_brief_{data_tag}.json", 1],    # 1329 single images 
]  # [[path, weight], ...], val: 700 samples

train_detail_data_weight = [
    [f"Denoise/SIDD/metas/train_iqa_A_detail_{data_tag}.json", 5],                                    # 320 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_Noise_L1_iqa_A_detail_{data_tag}.json", 1],       # 800 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_Noise_L3_iqa_A_detail_{data_tag}.json", 1],       # 800 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_Noise_L5_iqa_A_detail_{data_tag}.json", 1],       # 800 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_Noise_L1_iqa_A_detail_{data_tag}.json", 1], # 2650 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_Noise_L3_iqa_A_detail_{data_tag}.json", 1], # 2650 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_Noise_L5_iqa_A_detail_{data_tag}.json", 1], # 2650 pairs
    [f"Deblur/GoPro/metas/train_iqa_A_detail_{data_tag}.json", 5],                                    # 2103 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_SR1_iqa_A_detail_{data_tag}.json", 1],            # 800 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_SR2_iqa_A_detail_{data_tag}.json", 1],            # 800 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_train_pair_SR3_iqa_A_detail_{data_tag}.json", 1],            # 800 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_SR1_iqa_A_detail_{data_tag}.json", 1],      # 2650 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_SR2_iqa_A_detail_{data_tag}.json", 1],      # 2650 pairs
    [f"SuperResolution/Flickr2K/metas/Flickr2K_train_pair_SR3_iqa_A_detail_{data_tag}.json", 1],      # 2650 pairs
    [f"LowLight/LOL/metas/train_iqa_A_detail_{data_tag}.json", 15],                                   # 485 pairs
    [f"Dehaze/OTS/metas/train_iqa_A_detail_{data_tag}.json", 0.2],                                    # 72135 pairs
    [f"Derain/RainTrainL/metas/train_iqa_A_detail_{data_tag}.json", 40],                              # 200 pairs
    [f"Derain/RainDrop/metas/Raindrop_train_iqa_A_detail_{data_tag}.json", 10],                       # 861 pairs
]  # [[path, weight], ...], noise: 11950, blur: 10515, sr: 10350, ll: 7275, haze: 14427, rain: 8000, raindrop: 8610

test_detail_data_weight = [
    [f"Denoise/SIDD/metas/test_iqa_A_detail_{data_tag}.json", 1],                 # 1280 pairs
    [f"Denoise/Kodak/metas/Kodak_Noise_L1_iqa_A_detail_{data_tag}.json", 1],      # 24 pairs
    [f"Denoise/Kodak/metas/Kodak_Noise_L3_iqa_A_detail_{data_tag}.json", 1],      # 24 pairs
    [f"Denoise/Kodak/metas/Kodak_Noise_L5_iqa_A_detail_{data_tag}.json", 1],      # 24 pairs
    [f"Deblur/GoPro/metas/test_iqa_A_detail_{data_tag}.json", 1],                 # 2103 pairs
    [f"SuperResolution/DIV2K/metas/DIV2K_valid_pair_SR_iqa_A_detail_{data_tag}.json", 1],  # 100 pairs
    [f"LowLight/LOL/metas/test_iqa_A_detail_{data_tag}.json", 1],                 # 15 pairs
    [f"Dehaze/SOTS/metas/test_iqa_A_detail_{data_tag}.json", 1],                  # 500 pairs
    [f"Derain/Rain100L/metas/test_iqa_A_detail_{data_tag}.json", 1],              # 100 pairs
    [f"Derain/RainDrop/metas/Raindrop_test_a_iqa_A_detail_{data_tag}.json", 1],   # 58 pairs
    [f"Derain/RainDrop/metas/Raindrop_test_b_iqa_A_detail_{data_tag}.json", 1],   # 249 pairs
    [f"Other/UDC/metas/test_iqa_A_detail_{data_tag}.json", 1],                    # 60 pairs
    [f"Other/UDC/metas/val_iqa_A_detail_{data_tag}.json", 1],                     # 60 pairs
    [f"Desnow/Snow100k/metas/test_M_iqa_A_detail_{data_tag}.json", 1],            # 16588 pairs
    # ["Desnow/Snow100k/metas/test_realistic_iqa_A_detail_{data_tag}.json", 1],    # 1329 single images 
]  # [[path, weight], ...], val: 700 samples

@DATASETS.register_module()
class IRImgDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir=None,
        transform=None,
        resolution=256,
        load_vae_feat=False,
        load_text_feat=False,
        max_length=300,
        config=None,
        caption_proportion=None,
        external_caption_suffixes=None,
        external_clipscore_suffixes=None,
        clip_thr=0.0,
        clip_thr_temperature=1.0,
        img_extension=".png",
        dset = "train_brief",
        max_samples = None,
        caption_type = "IQA", # "question", "GT", "IQA"
        return_meta = True,
        synthetic_composite = False, 
        **kwargs,
    ):
        if isinstance(data_dir, list):
            data_dir = data_dir[0]
        if external_caption_suffixes is None:
            external_caption_suffixes = []
        if external_clipscore_suffixes is None:
            external_clipscore_suffixes = []

        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )
        self.transform, self.final_transform = self.get_transform(resolution, dset) if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.load_text_feat = load_text_feat
        self.resolution = resolution
        self.max_length = max_length
        self.caption_proportion = caption_proportion if caption_proportion is not None else {"prompt": 1.0}
        self.external_caption_suffixes = external_caption_suffixes
        self.external_clipscore_suffixes = external_clipscore_suffixes
        self.clip_thr = clip_thr
        self.clip_thr_temperature = clip_thr_temperature
        self.default_prompt = "prompt"
        self.img_extension = img_extension
        self.data_dir = data_dir
        self.dset = dset
        self.caption_type = caption_type
        self.return_meta = return_meta
        self.synthetic_composite = synthetic_composite
        if self.synthetic_composite:
            assert not load_vae_feat, "Error: can't synthetic distortion on the vae_feat, please disable 'load_vae_feat'."
            self.corruption_funcs = init_corruption_function("all")

        assert dset in ["train_brief", "train_brief_composite", "test_brief", "train_detail", "test_detail"], f'Dataset.dset type is Wrong, only support ["train_brief", "test_brief", "train_detail", "test_detail"], but receive {dset}.'
        if dset == "train_brief":
            meta_paths_weights = train_brief_data_weight
        elif dset == "train_brief_composite":
            meta_paths_weights = train_brief_composite_weight
        elif dset == "train_brief_all":
            meta_paths_weights = train_brief_all_weight
        elif dset == "test_brief":
            meta_paths_weights = test_brief_data_weight
        elif dset == "train_detail":
            meta_paths_weights = train_detail_data_weight
        elif dset == "test_detail":
            meta_paths_weights = test_detail_data_weight

        # READ META DATA
        meta_paths = [_[0] for _ in meta_paths_weights]
        meta_weights = [_[1] for _ in meta_paths_weights]
        self.dataset = []
        self.weights = []
        for meta_weight, meta_path in zip(meta_weights, meta_paths):
            with open(os.path.join(self.data_dir, meta_path)) as fr:
                meta_data = json.load(fr)
                if max_samples:
                    meta_data = meta_data[:min(len(meta_data), max_samples)]
                self.dataset += meta_data
                self.weights.extend([meta_weight] * len(meta_data))
        self.weights = torch.tensor(self.weights, dtype=torch.float32)
        
        self.ori_imgs_nums = len(self.dataset)
        self.logger.info(colored(f"Use Image Restoration Dataset, number of samples: {len(self.dataset)}", "red", attrs=["bold"]))
        self.logger.info(f"Text max token length: {self.max_length}")

    def getdata(self, idx):
        # meta info
        meta = self.dataset[idx]

        lq_path = os.path.join(self.data_dir, meta["image_A"])
        hq_path = os.path.join(self.data_dir, meta["image_ref"])

        filename, ext = os.path.splitext(os.path.basename(lq_path))
        self.img_extension = ext
        self.key = filename
        info = {}
        if self.caption_type == "IQA":
            info[self.default_prompt] = meta["conversations"][2]["value"].replace("\n ", "")
        elif self.caption_type == "question":
            info[self.default_prompt] = meta["conversations"][0]["value"]
        else:
            info[self.default_prompt] = meta["conversations"][1]["value"]

        caption_type, caption_clipscore = self.caption_type, 0.0
        txt_fea = info[self.default_prompt]

        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
        }

        if self.load_vae_feat:
            assert ValueError("Load VAE is not supported now")
        else:
            lq = Image.open(lq_path).convert("RGB")
            hq = Image.open(hq_path).convert("RGB")

        if self.transform:
            lq, hq = self.transform([lq, hq])
            if self.synthetic_composite:
                # generate corruption sample
                corruption_mode = np.random.choice(self.corruption_funcs)
                severity = np.random.choice(5, p=[0.3, 0.4, 0.25, 0.05, 0.00]) + 1  # [1, 5]
                lq = self._degrade_image(hq, corruption_mode, severity)
            lq, hq = self.final_transform([lq, hq])
        
        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)  # 1x1xT
        if self.load_text_feat:
            pass

        if self.return_meta:
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
            }
        else:
            return (
                lq,
                hq,
                txt_fea,
                attention_mask.to(torch.int16),
                data_info,
                idx,
                caption_type,
                "",
                str(caption_clipscore),
            )

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = idx + 1
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.dataset)

    def get_transform(self, resolution, dset):
        if "train" in dset:
            transform = T.Compose([
                PairTrans(T.Resize((resolution, resolution), interpolation=InterpolationMode.LANCZOS, antialias=True)),  # Image.BICUBIC
                RandomCropPair(resolution)])
            final_transform = T.Compose([
                PairTrans(T.ToTensor()),
                PairTrans(T.Normalize([0.5], [0.5])),
            ])
        else:
            transform = T.Compose([
                PairTrans(T.Resize((resolution, resolution), interpolation=InterpolationMode.LANCZOS, antialias=True)),  # Image.BICUBIC
                PairTrans(T.CenterCrop(resolution))])
            final_transform = T.Compose([
                PairTrans(T.ToTensor()),
                PairTrans(T.Normalize([0.5], [0.5])),
            ])
        transform.image_size = resolution
        return transform, final_transform

    def _degrade_image(
        self,
        hq: torch.Tensor,
        corruption_mode: str, 
        severity: int
    ):
        """
        Resize to [patch_size//4, patch_size], corrupt, then resize back to original resolution.
        Input:
            hq: [0, 255] uint8 tensor (C, H, W)
        Output:
            lq: [0, 255] uint8 tensor (C, H, W)
        """
        if corruption_mode == "clean":
            return hq

        w, h = hq.size
        # # random resize short edge to [resolution//2, resolution]
        # size = int(torch.randint(self.resolution // 2, self.resolution, ()))
        # lq = F.resize(hq, (size,))
        # process lq image
        lq = np.array(hq) # [0, 255] uint8 np (H, W, C)
        lq = corrupt(lq.astype(np.uint8), corruption_name=corruption_mode, severity=severity)
        # to tensor
        lq = Image.fromarray(lq)
        # resize back to original resolution
        # lq = F.resize(lq, (h, w))
        return lq

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

@DATASETS.register_module()
class IRImgQTokenDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir=None,
        transform=None,
        resolution=256,
        load_vae_feat=False,
        load_text_feat=False,
        max_length=300,
        config=None,
        caption_proportion=None,
        external_caption_suffixes=None,
        external_clipscore_suffixes=None,
        clip_thr=0.0,
        clip_thr_temperature=1.0,
        img_extension=".png",
        dset = "train_brief",
        max_samples = None,
        use_tokens = "prefill", # "hidden_state", "prefill", "ALL", "None"
        max_tokens = 400, 
        **kwargs,
    ):
        if isinstance(data_dir, list):
            data_dir = data_dir[0]
        if external_caption_suffixes is None:
            external_caption_suffixes = []
        if external_clipscore_suffixes is None:
            external_clipscore_suffixes = []

        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )
        self.transform, self.final_transform = self.get_transform(resolution, dset) if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.load_text_feat = load_text_feat
        self.resolution = resolution
        self.max_length = max_length
        self.caption_proportion = caption_proportion if caption_proportion is not None else {"prompt": 1.0}
        self.external_caption_suffixes = external_caption_suffixes
        self.external_clipscore_suffixes = external_clipscore_suffixes
        self.clip_thr = clip_thr
        self.clip_thr_temperature = clip_thr_temperature
        self.default_prompt = "prompt"
        self.use_tokens = use_tokens
        self.img_extension = img_extension
        self.max_tokens = max_tokens
        self.data_dir = data_dir
        self.dset = dset
        assert dset in ["train_brief", "test_brief", "train_detail", "test_detail"], f'Dataset.dset type is Wrong, only support ["train_brief", "test_brief", "train_detail", "test_detail"], but receive {dset}.'
        if dset == "train_brief":
            meta_paths_weights = train_brief_data_weight
        elif dset == "test_brief":
            meta_paths_weights = test_brief_data_weight
        elif dset == "train_detail":
            meta_paths_weights = train_detail_data_weight
        elif dset == "test_detail":
            meta_paths_weights = test_detail_data_weight

        # READ META DATA
        meta_paths = [_[0] for _ in meta_paths_weights]
        meta_weights = [_[1] for _ in meta_paths_weights]
        self.dataset = []
        self.weights = []
        for meta_weight, meta_path in zip(meta_weights, meta_paths):
            with open(os.path.join(self.data_dir, meta_path)) as fr:
                meta_data = json.load(fr)
                if max_samples:
                    meta_data = meta_data[:min(len(meta_data), max_samples)]
                self.dataset += meta_data
                self.weights.extend([meta_weight] * len(meta_data))
        self.weights = torch.tensor(self.weights, dtype=torch.float32)
        
        self.ori_imgs_nums = len(self.dataset)
        self.logger.info(colored(f"Use Image Restoration Dataset, number of samples: {len(self.dataset)}", "red", attrs=["bold"]))
        self.logger.info(f"Text max token length: {self.max_length}")

    def getdata(self, idx):
        # meta info
        meta = self.dataset[idx]

        # load image 
        lq_path = os.path.join(self.data_dir, meta["image_A"])
        hq_path = os.path.join(self.data_dir, meta["image_ref"])
        filename, ext = os.path.splitext(os.path.basename(lq_path))
        self.img_extension = ext
        self.key = filename
        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
        }
        if self.load_vae_feat:
            assert ValueError("Load VAE is not supported now")
        else:
            lq = Image.open(lq_path).convert("RGB")
            hq = Image.open(hq_path).convert("RGB")
        
        if self.transform:
            lq, hq = self.transform([lq, hq])

        # load text information 
        info = {}
        info[self.default_prompt] = meta["conversations"][2]["value"].replace("\n ", "")
        caption_type, caption_clipscore = self.default_prompt, 0.0
        txt_fea = info[self.default_prompt]
        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)  # 1x1xT

        # load IQA last layer tokens
        if self.use_tokens == "prefill":
            last_tokens = torch.load(os.path.join(self.data_dir, meta["conversations"][2]["prefill"]), map_location="cpu")["latents"]
            last_tokens, mask_tokens, length_tokens = self.token_pad_or_truncate(last_tokens, max_length=self.max_tokens, pad_side="left", pad_value=0.0)  # padding or truncate to fixed length, (max_length, 4096)
        elif self.use_tokens == "hidden_state": 
            last_tokens = torch.load(os.path.join(self.data_dir, meta["conversations"][2]["hidden_state"]), map_location="cpu")["latents"]
            last_tokens, mask_tokens, length_tokens = self.token_pad_or_truncate(last_tokens, max_length=self.max_tokens, pad_side="left", pad_value=0.0)  # padding or truncate to fixed length, (max_length, 4096)
        elif self.use_tokens == "ALL": 
            # torch.Size([Th, 4096]) torch.Size([Tp, 4096]) -> torch.Size([Tp + Th, 4096]) 
            prefill_tokens = torch.load(os.path.join(self.data_dir, meta["conversations"][2]["prefill"]), map_location="cpu")["latents"]
            hidden_tokens = torch.load(os.path.join(self.data_dir, meta["conversations"][2]["hidden_state"]), map_location="cpu")["latents"]
            last_tokens = torch.cat([prefill_tokens, hidden_tokens], dim=0)
            last_tokens, mask_tokens, length_tokens = self.token_pad_or_truncate(last_tokens, max_length=self.max_tokens, pad_side="left", pad_value=0.0)  # padding or truncate to fixed length, (max_length, 4096)
        else:
            last_tokens = None
            mask_tokens = None 
            length_tokens = 0

        if self.load_text_feat:
            pass

        return (
            lq,
            hq,
            txt_fea,
            attention_mask.to(torch.int16),
            data_info,
            idx,
            caption_type,
            "",
            str(caption_clipscore),
            last_tokens, 
            mask_tokens, 
            length_tokens 
        )

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = idx + 1
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.dataset)

    def get_transform(self, resolution, dset):
        if "train" in dset:
            transform = T.Compose([
                PairTrans(T.Resize((resolution, resolution), interpolation=InterpolationMode.LANCZOS, antialias=True)),  # Image.BICUBIC
                RandomCropPair(resolution),
                PairTrans(T.ToTensor()),
                PairTrans(T.Normalize([0.5], [0.5])),
            ])
        else:
            transform = T.Compose([
                PairTrans(T.Resize((resolution, resolution), interpolation=InterpolationMode.LANCZOS, antialias=True)),  # Image.BICUBIC
                PairTrans(T.CenterCrop(resolution)),
                PairTrans(T.ToTensor()),
                PairTrans(T.Normalize([0.5], [0.5])),
            ])
        transform.image_size = resolution
        return transform
        
    def token_pad_or_truncate(self, x, max_length=400, pad_side="right", pad_value=0.0):
        assert x.ndim == 2
        L, D = x.shape
        # truncate
        if L >= max_length: 
            if pad_side == "right":
                y = x[:max_length]
                mask = torch.zeros(max_length, dtype=torch.bool, device=x.device) # T
            else: 
                y = x[-max_length:]
                mask = torch.zeros(max_length, dtype=torch.bool, device=x.device) # T
            return y, mask, L 
        # padding
        else:
            y = torch.full((max_length, D), pad_value, dtype=x.dtype, device=x.device)
            mask = torch.ones(max_length, dtype=torch.bool, device=x.device) # T
            if pad_side == "right":
                y[:L] = x
                mask[:L] = False
            else: 
                y[max_length-L:] = x
                mask[max_length-L:] = False
            return y, mask, L 

class PairTrans:
    def __init__(self, t): self.t = t
    def __call__(self, pair):
        img1, img2 = pair
        return self.t(img1), self.t(img2)

class RandomCropPair:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
    def __call__(self, pair):
        img1, img2 = pair
        i, j, h, w = T.RandomCrop.get_params(img1, self.size)
        return F.crop(img1, i, j, h, w), F.crop(img2, i, j, h, w)

class DistributedWeightedRangedSampler(Sampler):
    """
    chunk first than weighted.
    """
    def __init__(
        self,
        dataset: Dataset,
        weights: Optional[torch.Tensor] = None, 
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        local_num_samples: Optional[int] = None,
        drop_last: bool = False,
        seed: int = 1229
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.step_start = 0

        if not dist.is_available() or not dist.is_initialized():
            warnings.warn("DistributedWeightedRangedSampler: distributed not initialized; assuming single process.")
            num_replicas = 1
            rank = 0
        else:
            num_replicas = int(num_replicas) or dist.get_world_size()
            rank = int(rank) or dist.get_rank()
        assert rank >= 0 and rank < num_replicas, "invalid rank/num_replicas"
        self.num_replicas = num_replicas
        self.rank = rank

        num_samples = len(dataset)
        if drop_last: 
            self.worker_chunk = num_samples // num_replicas
            total = self.worker_chunk * num_replicas
        else:
            self.worker_chunk = (num_samples + num_replicas - 1) // num_replicas
            total = num_samples
        
        self.worker_start = rank * self.worker_chunk
        if self.worker_start + self.worker_chunk > num_samples:
            self.worker_start = num_samples - self.worker_chunk
            self.worker_end = num_samples
        else:
            self.worker_end = self.worker_start + self.worker_chunk
        self.local_indices = torch.arange(self.worker_start, self.worker_end, dtype=torch.long)

        # weighted sampler
        if weights is not None:
            assert isinstance(weights, torch.Tensor) and weights.dim() == 1 and len(weights) == num_samples, \
                "weights must be 1D tensor with length == len(dataset)"
            w = weights[self.local_indices].clone().to(dtype=torch.float64)
            w = torch.clamp(w, min=0)
            if w.sum() == 0:
                warnings.warn("All local weights are zero; falling back to uniform")
                w = torch.ones_list(w)
            self.local_weights = (w / w.sum())
        else:
            self.local_weights = None

        # rank sample
        self.replacement = bool(replacement)
        L = len(self.local_indices)
        if local_num_samples is None:
            self.local_num_samples = L 
        else:
            self.local_num_samples = int(local_num_samples)
            if not replacement and self.local_num_samples > L:
                warnings.warn(
                    f"local_num_samples ({self.local_num_samples}) > local shard size ({L}) "
                    "with replacement=False; capping to shard size."
                )
                self.local_num_samples = L

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_start(self, start):
        self.step_start = max(0, int(start))

    def __len__(self):
        return max(0, self.local_num_samples - self.step_start)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1000 + self.rank)
        if self.local_weights is None:
            order = torch.randperm(len(self.local_indices), generator=g)
            seq = self.local_indices[order]
            if self.local_num_samples < len(seq):
                seq = seq[: self.local_num_samples]
        else:
            take = torch.multinomial(self.local_weights, self.local_num_samples, replacement=self.replacement, generator=g)
            seq = self.local_indices[take]
        
        # offset
        if self.step_start > 0:
            seq = seq[self.step_start: ]
        
        # epoch 
        for idx in seq.tolist():
            yield idx 
        self.epoch += 1

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from diffusion.data.transforms import get_transform

    image_size = 256  # 256
    # # IR sample 
    # train_dataset = IRImgDataset(
    #     data_dir="/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/Datasets/IR",
    #     resolution=image_size,
    #     transform=None,
    #     max_length=300,
    #     load_vae_feat=False,
    #     num_replicas=1,
    #     dset = "train_brief",
    #     max_samples = None,
    # )
    # IR with last layer tokens sampel 
    train_dataset = IRImgQTokenDataset(
        data_dir="/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/Datasets/IR",
        resolution=image_size,
        transform=None,
        max_length=300,
        load_vae_feat=False,
        num_replicas=1,
        dset = "train_brief",
        max_samples = None,
        use_tokens = "prefill"
    )

    dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=20)

    def tensor_to_pil(x):
        x = x.detach().cpu()
        x = x * 0.5 + 0.5
        x = x.clamp(0, 1)
        return F.to_pil_image(x)

    print(len(dataloader))
    for it, data in enumerate(dataloader):
        lq, hq, txt_fea, attention_mask, data_info, _, _, _, _, last_tokens, mask_tokens, length_tokens = data
        # print(lq.shape, hq.shape, txt_fea, last_tokens.shape, mask_tokens.shape, length_tokens.shape)
        idx = 1
        print(mask_tokens)
        # lq1 = tensor_to_pil(lq[idx])
        # hq1 = tensor_to_pil(hq[idx])
        # lq1.save("lq1.png")
        # hq1.save("hq1.png")
        if it > 1:
            break
        print(it, end="\r")