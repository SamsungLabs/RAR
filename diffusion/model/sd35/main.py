# NOTE: Must have folder `models` with the following files:
# - `clip_g.safetensors` (openclip bigG, same as SDXL)
# - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL)
# - `t5xxl.safetensors` (google T5-v1.1-XXL)
# - `sd3_medium.safetensors` (or whichever main MMDiT model file)
# Also can have
# - `sd3_vae.safetensors` (holds the VAE separately if needed)

# not support for controlnet

import datetime
import math
import os
import pickle
import re

import fire
import numpy as np
# import sd3_impls
import torch
from PIL import Image
from safetensors import safe_open
from diffusion.model.sd35.sd3_impls import (
    SDVAE,
    BaseModel,
    P2PModel,
    CFGDenoiser,
    SkipLayerCFGDenoiser,
    ModelSamplingDiscreteFlow
)
import diffusion.model.sd35.sd3_impls as sd3_impls
from diffusion.model.sd35.text_encoder import TextEncoder

from tqdm import tqdm
from torchvision.utils import save_image

#################################################################################################
### Configs
#################################################################################################
# Note: Sigma shift value, publicly released models use 3.0
SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 1024
HEIGHT = 1024
# Pick your prompt
PROMPT = [
    "a photo of a cat",
    # "Convenience store entrance at night. On the glass door, a vinyl decal reads 'OPEN FOR QUALITY'. Inside, shelves and fluorescent lights; outside, a cyclist passing by",
    # "Sunrise beach, shallow tide washing over smooth sand. A piece of weathered driftwood lies near the shoreline with a subtle branded text [SOS] on its surface; wet sand reflections, micro-ripples, sun flare at horizon.",
]
# PROMPT = "A 25-year-old East Asian woman with shoulder-length dark hair and soft bangs, minimal natural makeup, wearing a textured linen blazer over a white silk camisole, standing beside a north-facing window in a minimalist studio, soft rim light on the hair, gentle Rembrandt key light, half-body portrait, eye-level view, balanced negative space, shallow depth of field with circular bokeh, ultra-detailed skin pores and fine peach fuzz, subtle film grain, neutral warm tones with a hint of teal in the shadows, Vogue editorial style, 50mm lens, f/2, ISO 200, crisp yet organic, color-accurate, natural retouching, no over-smoothing, award-winning fashion photography."
# Most models prefer the range of 4-5, but still work well around 7
CFG_SCALE = 4.5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
# sd3_medium is quite decent at 28 steps
STEPS = 40
# Seed
SEED = 23
SEEDTYPE = "fixed"
# SEEDTYPE = "rand"
# SEEDTYPE = "roll"
# Actual model file path
MODEL = "/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/ModelZoo/stable-diffusion-3.5-medium/sd3.5_medium.safetensors"
# VAE model file path, or set None to use the same model file
VAEFile = None  # "models/sd3_vae.safetensors"
# Optional init image file path
# INIT_IMAGE = "/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/Datasets/IR/Dehaze/SOTS/outdoor/hazy/0004_0.9_0.12.jpg"
INIT_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.9
# Output file path
OUTDIR = "tmp"
# SAMPLER
SAMPLER = "dpmpp_2m"
# Text Encoder Ckpt
MODEL_FOLDER = "/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/ModelZoo/stable-diffusion-3.5-medium/text_encoders"

#################################################################################################
### Wrappers for model parts
#################################################################################################
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
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}, used from scratch"
                    )
                    continue
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e

def load_text_encoder(model_folder, device="cuda"):
    return TextEncoder(model_folder, device)

def load_mmdit(
        model = f"./ModelZoo/stable-diffusion-3.5-medium/sd3.5_medium.safetensors",
        shift = 3.0, 
        verbose=False, 
        device="cuda",
        input_size=None,
        in_channels=16,
        **kwargs
    ):
    with safe_open(model, framework="pt", device="cpu") as f:
        model = BaseModel(
            shift=shift,
            file=f,
            prefix="model.diffusion_model.",
            device="cuda",
            dtype=torch.float32,
            control_model_ckpt=None,
            verbose=verbose,
            input_size=input_size,
            in_channels=in_channels,
            **kwargs
        )
        load_into(f, model, "model.", "cuda", torch.float32)
    mmdit = model.diffusion_model.to(device)
    for p in mmdit.parameters():
        p.requires_grad = True
    return mmdit.train()

def load_mmdit_p2p(
        model = f"/home/work/shared-fi-datasets-01/users/hsiang.chen/Project/ModelZoo/stable-diffusion-3.5-medium/sd3.5_medium.safetensors",
        shift = 3.0, 
        verbose=False, 
        device="cuda",
        input_size=None,
        in_channels=16,
        **kwargs
    ):
    with safe_open(model, framework="pt", device="cpu") as f:
        model = P2PModel(
            shift=shift,
            file=f,
            prefix="model.diffusion_model.",
            device="cuda",
            dtype=torch.float32,
            control_model_ckpt=None,
            verbose=verbose,
            input_size=input_size,
            in_channels=in_channels,
            **kwargs
        )
        load_into(f, model, "model.", "cuda", torch.float32)
    mmdit_p2p = model.diffusion_model.to(device)
    for p in mmdit_p2p.parameters():
        p.requires_grad = True
    return mmdit_p2p.train()

def load_vae(model, device="cuda", dtype: torch.dtype = torch.float16):
    with safe_open(model, framework="pt", device="cpu") as f:
        VAE = SDVAE(device="cpu", dtype=dtype).eval().cpu()
        prefix = ""
        if any(k.startswith("first_stage_model.") for k in f.keys()):
            prefix = "first_stage_model."
        load_into(f, VAE, prefix, "cpu", dtype)
    return VAE.to(device)

def load_scheduler(shift):
    return ModelSamplingDiscreteFlow(shift=shift)

#################################################################################################
### Function
#################################################################################################
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

#################################################################################################
### Main inference logic
#################################################################################################
class SD3Inferencer:
    def __init__(self):
        self.verbose = False

    def print(self, txt):
        if self.verbose:
            print(txt)

    def load(
        self,
        model=MODEL,
        vae=VAEFile,
        shift=SHIFT,
        controlnet_ckpt=None,
        verbose=False,
        model_folder=MODEL_FOLDER,
    ):
        self.verbose = verbose
        self.txt_enc = load_text_encoder(model_folder, "cuda")
        print(f"Loading SD3-p2p model {os.path.basename(model)}...")
        self.sd3_p2p = load_mmdit_p2p(model, shift, verbose, "cuda", None, 16)
        print(f"Loading SD3 model {os.path.basename(model)}...")
        self.sd3 = load_mmdit(model, shift, verbose, "cuda", None, 16)
        print(f"Loading Sampler...")
        self.scheduler = load_scheduler(shift)
        print("Loading VAE model...")
        self.vae = load_vae(vae or model, "cuda")
        print("Models loaded.")

    def get_empty_latent(self, batch_size, width, height, seed, device="cuda"):
        self.print("Prep an empty latent...")
        shape = (batch_size, 16, height // 8, width // 8)
        latents = torch.zeros(shape, device=device)
        for i in range(shape[0]):
            prng = torch.Generator(device=device).manual_seed(int(seed + i))
            latents[i] = torch.randn(shape[1:], generator=prng, device=device)
        return latents

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(
            f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}"
        )
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)

    def max_denoise(self, sigmas):
        max_sigma = float(self.scheduler.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}

    def do_sampling(
        self,
        latent,
        seed,
        conditioning,
        neg_cond,
        steps,
        cfg_scale,
        sampler="dpmpp_2m",
        denoise=1.0,
        skip_layer_config={},
    ) -> torch.Tensor:
        self.print("Sampling...")
        latent = latent.half().cuda()
        self.sd3 = self.sd3.cuda()
        noise = self.get_noise(seed, latent).cuda()
        sigmas = self.get_sigmas(self.scheduler, steps).cuda()
        sigmas = sigmas[int(steps * (1 - denoise)) :]
        conditioning = self.fix_cond(conditioning)
        neg_cond = self.fix_cond(neg_cond)
        extra_args = {
            "cond": conditioning,
            "uncond": neg_cond,
            "cond_scale": cfg_scale,
            # "controlnet_cond": None,
        }
        noise_scaled = self.scheduler.noise_scaling(
            sigmas[0], noise, latent, self.max_denoise(sigmas)
        )
        sample_fn = getattr(sd3_impls, f"sample_{sampler}")
        denoiser = (
            SkipLayerCFGDenoiser
            if skip_layer_config.get("scale", 0) > 0
            else CFGDenoiser
        )
        latent = sample_fn(
            denoiser(self.sd3, steps, skip_layer_config),
            noise_scaled,
            sigmas,
            self.scheduler,
            extra_args=extra_args,
        )
        self.sd3 = self.sd3.cpu()
        self.print("Sampling done")
        return latent

    def load_image(self, image, width, height):
        image = Image.open(image)
        image = image.resize((width, height), Image.LANCZOS)
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images).cuda()
        image_torch = 2.0 * image_torch - 1.0
        return image_torch

    def gen_image(
        self,
        prompts=[],
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        sampler=SAMPLER,
        seed=SEED,
        seed_type=SEEDTYPE,
        out_dir=OUTDIR,
        init_image=INIT_IMAGE,
        denoise=DENOISE,
        skip_layer_config={},
    ):
        # sampling init noise
        if init_image:
            print("Encoding image to latent (sampling)...")
            image_torch = self.load_image(init_image, width, height).cuda()
            latent = self.vae.encode(image_torch).cuda()
            latent = self.vae.process_in(latent)
        else:
            latent = self.get_empty_latent(1, width, height, seed, "cpu")
            latent = latent.cuda()

        seed_num = None
        if seed_type == "roll":
            seed_num = seed if seed_num is None else seed_num + 1
        elif seed_type == "rand":
            seed_num = torch.randint(0, 100000, (1,)).item()
        else:  # fixed
            seed_num = seed

        # condition
        neg_cond = self.txt_enc.get_cond("")
        conditioning = self.txt_enc.get_cond(prompts)
        print(conditioning[0].shape, conditioning[1].shape)

        # repeat for different prompts
        batch_size = conditioning[0].size(0)
        latent = latent.repeat(batch_size, 1, 1, 1)
        neg_cond = [neg_cond[0].repeat(batch_size, 1, 1), neg_cond[1].repeat(batch_size, 1)]

        # sampling 
        sampled_latent = self.do_sampling(
            latent,
            seed_num,
            conditioning,
            neg_cond,
            steps,
            cfg_scale,
            sampler,
            denoise if init_image else 1.0,
            skip_layer_config,
        )

        # decoding
        self.print(f"Decoding latent to image...")
        sampled_latent = self.vae.process_out(sampled_latent)
        image = self.vae.decode(sampled_latent.cuda())
        self.print(f"sampling: {image.shape}")

        # image saving
        save_path = os.path.join(out_dir, f"sample.png")
        image = image.float()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0) 
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 1, 3)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = []
        for img in decoded_np:
            out_image.append(Image.fromarray(img))
        out_image = concat_horizontal(out_image)
        self.print(f"Saving to to {save_path}")
        out_image.save(save_path)
        self.print("Done")
            

CONFIGS = {
    "sd3_medium": {
        "shift": 1.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_medium": {
        "shift": 3.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
        "skip_layer_config": {
            "scale": 2.5,
            "start": 0.01,
            "end": 0.20,
            "layers": [7, 8, 9],
            "cfg": 4.0,
        },
    },
    "sd3.5_large": {
        "shift": 3.0,
        "steps": 40,
        "cfg": 4.5,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_large_turbo": {"shift": 3.0, "cfg": 1.0, "steps": 4, "sampler": "euler"},
    "sd3.5_large_controlnet_blur": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_canny": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_depth": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
}

@torch.no_grad()
def main(
    prompts=PROMPT,
    model=MODEL,
    out_dir=OUTDIR,
    postfix=None,
    seed=SEED,
    seed_type=SEEDTYPE,
    sampler=None,
    steps=None,
    cfg=None,
    shift=None,
    width=WIDTH,
    height=HEIGHT,
    controlnet_ckpt=None,
    vae=VAEFile,
    init_image=INIT_IMAGE,
    denoise=DENOISE,
    skip_layer_cfg=False,
    verbose=False,
    model_folder=MODEL_FOLDER,
    **kwargs,
):
    assert not kwargs, f"Unknown arguments: {kwargs}"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False 
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    config = CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {})
    _shift = shift or config.get("shift", 3)
    _steps = steps or config.get("steps", 50)
    _cfg = cfg or config.get("cfg", 5)
    _sampler = sampler or config.get("sampler", "dpmpp_2m")

    if skip_layer_cfg:
        skip_layer_config = CONFIGS.get(
            os.path.splitext(os.path.basename(model))[0], {}
        ).get("skip_layer_config", {})
        cfg = skip_layer_config.get("cfg", cfg)
    else:
        skip_layer_config = {}

    if controlnet_ckpt is not None:
        controlnet_config = CONFIGS.get(
            os.path.splitext(os.path.basename(controlnet_ckpt))[0], {}
        )
        _shift = shift or controlnet_config.get("shift", shift)
        _steps = steps or controlnet_config.get("steps", steps)
        _cfg = cfg or controlnet_config.get("cfg", cfg)
        _sampler = sampler or controlnet_config.get("sampler", sampler)

    inferencer = SD3Inferencer()
    inferencer.load(
        model,
        vae,
        _shift,
        controlnet_ckpt,
        verbose,
        model_folder,
    )

    if isinstance(prompts, str):
        if os.path.splitext(prompt)[-1] == ".txt":
            with open(prompt, "r") as f:
                prompts = [l.strip() for l in f.readlines()]
        else:
            prompts = [prompt]

    out_dir = os.path.join(
        out_dir,
        (postfix or datetime.datetime.now().strftime("_%Y-%m-%dT%H-%M-%S")),
    )
    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, "prompts.txt"), "w") as f:
        for tt in prompts:
            f.writelines(f"{tt}\n")

    print(prompts)
    inferencer.gen_image(
        prompts,
        width,
        height,
        _steps,
        _cfg,
        _sampler,
        seed,
        seed_type,
        out_dir,
        init_image,
        denoise,
        skip_layer_config,
    )

if __name__ == "__main__":
    fire.Fire(main)
