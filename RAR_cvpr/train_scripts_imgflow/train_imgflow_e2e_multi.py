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

import datetime
import gc
import getpass
import hashlib
import json
import os, sys
import os.path as osp
import time
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import yaml
from easydict import EasyDict

warnings.filterwarnings("ignore")  # ignore warning

import numpy as np
import pyrallis
import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs, skip_first_batches
from PIL import Image
from termcolor import colored

from diffusion import DPMS, FlowEuler, Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.data.datasets import DistributedWeightedRangedSampler
from diffusion.model.builder import build_model
from diffusion.model.model_growth_utils import ModelGrowthInitializer
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, set_random_seed
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer
from diffusion.model.sd35 import load_scheduler, load_vae, load_mmdit, load_text_encoder, load_mmdit_p2p
from diffusion.model.utils import set_fp32_attention, set_grad_checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def set_fsdp_env():
    # Basic FSDP settings
    os.environ["ACCELERATE_USE_FSDP"] = "true"

    # Auto wrapping policy
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaMSBlock"  # Your transformer block name

    # Performance optimization settings
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_FORWARD_PREFETCH"] = "false"

    # State dict settings
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"
    os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"

    # Sharding strategy
    os.environ["FSDP_SHARDING_STRATEGY"] = "FULL_SHARD"

    # Memory optimization settings (optional)
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
    os.environ["FSDP_OFFLOAD_PARAMS"] = "false"

    # Precision settings
    os.environ["FSDP_REDUCE_SCATTER_PRECISION"] = "fp32"
    os.environ["FSDP_ALL_GATHER_PRECISION"] = "fp32"
    os.environ["FSDP_OPTIMIZER_STATE_PRECISION"] = "fp32"

class IQAIR(nn.Module):
    def __init__(self, model, connector, assessment, device):
        super().__init__()
        self.model = model.to(device).train()
        self.assessment = assessment.to(device).train()
        self.connector = connector.to(device).train()

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

def train(
    config, args, accelerator, model, optimizer, lr_scheduler, train_dataloader, train_diffusion, logger
):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model, max_frames_to_save=100)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    loss_nan_timer = 0
    model_instance.to(accelerator.device)

    # Cache Dataset for BatchSampler
    if args.caching and config.model.multi_scale:
        caching_start = time.time()
        logger.info(
            f"Start caching your dataset for batch_sampler at {cache_file}. \n"
            f"This may take a lot of time...No training will launch"
        )
        train_dataloader.batch_sampler.sampler.set_start(max(train_dataloader.batch_sampler.exist_ids, 0))
        accelerator.wait_for_everyone()
        for index, _ in enumerate(train_dataloader):
            accelerator.wait_for_everyone()
            if index % 2000 == 0:
                logger.info(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                print(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
            if (time.time() - caching_start) / 3600 > 3.7:
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
                accelerator.wait_for_everyone()
                break
            if len(train_dataloader.batch_sampler.cached_idx) == len(train_dataloader) - 1000:
                logger.info(
                    f"Saving rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
            accelerator.wait_for_everyone()
            continue
        accelerator.wait_for_everyone()
        print(f"Saving rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
        json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
        return

    # Now you train the model
    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        sampler = (
            train_dataloader.batch_sampler.sampler
            if (num_replicas > 1 or config.model.multi_scale)
            else train_dataloader.sampler
        )
        sampler.set_epoch(epoch)
        sampler.set_start(max((skip_step - 1) * config.train.train_batch_size, 0))
        if skip_step > 1 and accelerator.is_main_process:
            logger.info(f"Skipped Steps: {skip_step}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        IQA_time_all = 0
        connector_time_all = 0
        vae_time_all = 0
        model_time_all = 0
        recur_round = config.scheduler.recur_round
        recur_truncated = config.scheduler.recur_truncated
        recur_loss_weights = config.scheduler.recur_loss_weights
        update_image = config.scheduler.update_image
        for step, batch in enumerate(train_dataloader):
            # image, json_info, key = batch
            """
            batch (IR_dataset): (
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
            """
            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start
            vae_time_start = time.time()

            ## Step1: VAE
            if load_vae_feat:
                src_z = batch["lq"].to(accelerator.device)
                tgt_z = batch["hq"].to(accelerator.device)
            else:
                with torch.no_grad():
                    src_z = vae.encode(batch["lq"].to(accelerator.device)).to(accelerator.device)
                    src_z = vae.process_in(src_z).to(accelerator.device)
                    tgt_z = vae.encode(batch["hq"].to(accelerator.device)).to(accelerator.device)
                    tgt_z = vae.process_in(tgt_z).to(accelerator.device)
            accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start
            input_images = src_z.detach()
            target_images = tgt_z.detach()

            # Sample a random timestep for each image
            bs = target_images.shape[0]
            timesteps = torch.randint(
                0, config.scheduler.train_sampling_steps, (bs,), device=target_images.device
            ).long()
            if config.scheduler.weighting_scheme in ["logit_normal"]:
                # adapting from diffusers.training_utils
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.scheduler.weighting_scheme,
                    batch_size=bs,
                    logit_mean=config.scheduler.logit_mean,
                    logit_std=config.scheduler.logit_std,
                    mode_scale=None,  # not used
                )
                timesteps = (u * config.scheduler.train_sampling_steps).long().to(target_images.device)
            
            # forward
            grad_norm = None
            accelerator.wait_for_everyone()
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss = 0
                # recursive
                for n_round in range(recur_round):
                    # Step2: IQA
                    IQA_time_start = time.time()
                    IQA_token = model.module.assessment(
                        {
                            "img": target_images,    
                            "img_A": input_images,
                            "img_B": [None], 
                            "img_path": batch["hq_path"],
                            "img_A_path": batch["lq_path"],
                            "img_B_path": [None],
                            "conversation": batch["conversation"],
                            "task_type": batch["task_type"],
                        }, 
                        latent_input=True)
                    IQA_time_all += time.time() - IQA_time_start
                    # print("IQA_token", IQA_token.shape)

                    # Step3: Connector
                    connector_time_start = time.time()
                    last_token, mask_token = token_pad_or_truncate(IQA_token)
                    # print("Token_Pad:", last_token.shape, mask_token.shape)
                    pred_tokens = model.module.connector(last_token, key_padding_mask=mask_token)
                    pred_context, pred_y = mapping_to_cond(pred_tokens) # (64, 154, 4096), (64, 1, 2048)
                    # print("Connector: ", pred_context.shape, pred_y.shape)
                    connector_time_all += time.time() - connector_time_start

                    # Step4: DiT
                    model_time_start = time.time()
                    dit_output = train_diffusion.training_losses_d2c(
                        model.module.model, target_images, timesteps, img_cond=input_images, return_output=True,
                        model_kwargs=dict(y=pred_y, context=pred_context)
                    )

                    # Step5: Update
                    loss += recur_loss_weights[n_round] * dit_output["loss"].mean()

                    # Step6. Prepare for next round
                    if update_image: 
                        if recur_truncated:
                            input_images = dit_output["pred_x0"].detach()
                        else:
                            input_images = dit_output["pred_x0"]

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)

                # accelerator.step(optimizer)
                optimizer.step()
                lr_scheduler.step()
                accelerator.wait_for_everyone()
                model_time_all += time.time() - model_time_start

            if torch.any(torch.isnan(loss)):
                loss_nan_timer += 1
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_iqa = IQA_time_all / config.train.log_interval
                t_vae = vae_time_all / config.train.log_interval
                t_connecotr = connector_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(
                    datetime.timedelta(
                        seconds=int(
                            avg_time
                            * (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step - 1)
                        )
                    )
                )
                log_buffer.average()

                current_step = (
                    global_step - sampler.step_start // config.train.train_batch_size
                ) % train_dataloader_len
                current_step = train_dataloader_len if current_step == 0 else current_step

                info = (
                    f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {train_dataloader_len}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, data:{t_d:.3f}, vae:{t_vae:.3f}, "
                    f"IQA:{t_iqa:.3f}, Connector:{t_connecotr:.3f}, model:{t_m:.3f}, lr:{lr:.3e}, loss:{loss.item():.3f}"
                )

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                IQA_time_all = 0
                vae_time_all = 0
                connector_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if loss_nan_timer > 20:
                raise ValueError("Loss is NaN too much times. Break here.")
            if (
                global_step % config.train.save_model_steps == 0
                or (time.time() - training_start_time) / 3600 > config.train.early_stop_hours
            ):
                torch.cuda.synchronize()
                accelerator.wait_for_everyone()

                # Choose different saving methods based on whether FSDP is used
                if config.train.use_fsdp:
                    # FSDP mode
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        work_dir=osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        model=model,
                        accelerator=accelerator,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        step=global_step,
                        add_symlink=True,
                    )
                else:
                    # DDP mode
                    if accelerator.is_main_process:
                        os.umask(0o000)
                        ckpt_saved_path = save_checkpoint(
                            work_dir=osp.join(config.work_dir, "checkpoints"),
                            epoch=epoch,
                            model=accelerator.unwrap_model(model),
                            model_ema=None, 
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            step=global_step,
                            generator=generator,
                            add_symlink=True,
                            keep_last=True,
                        )

                if accelerator.is_main_process:
                    if config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1:
                        online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                        os.makedirs(online_metric_monitor_dir, exist_ok=True)
                        with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                            f.write(osp.join(config.work_dir, "config.py") + "\n")
                            f.write(ckpt_saved_path)

                if (time.time() - training_start_time) / 3600 > config.train.early_stop_hours:
                    logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                    return

            if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or (step + 1) == 1):
                print("config.train.visualize")
                if config.train.use_fsdp:
                    merged_state_dict = accelerator.get_state_dict(model)
                print("merged_state_dict")
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if config.train.use_fsdp:
                        model_instance.load_state_dict(merged_state_dict)
                    torch.cuda.empty_cache()

            # avoid dead-lock of multiscale data batch sampler
            if (
                config.model.multi_scale
                and (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step) < 30
            ):
                print("multi_scale")
                global_step = (
                    (global_step + train_dataloader_len - 1) // train_dataloader_len
                ) * train_dataloader_len + 1
                logger.info("Early stop current iteration")
                skip_first_batches(train_dataloader, True)
                break

            data_time_start = time.time()
        
        print("save_model_epochs")
        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.print(f"[rank {accelerator.process_index}] about to wait_for_everyone at epoch {epoch}")
            sys.stdout.flush()
            accelerator.wait_for_everyone()
            torch.cuda.synchronize()

            print("save_checkpoint...")
            # Choose different saving methods based on whether FSDP is used
            if config.train.use_fsdp:
                # FSDP mode
                os.umask(0o000)
                ckpt_saved_path = save_checkpoint(
                    work_dir=osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    model=model,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step=global_step,
                    add_symlink=True,
                )
            else:
                # DDP mode
                if accelerator.is_main_process:
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        model_ema=None,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        generator=generator,
                        add_symlink=True,
                        keep_last=True,
                    )
            print("save_checkpoint, finishing...")

            if accelerator.is_main_process:
                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)
            print("save_checkpoint, finishing2...")

@pyrallis.wrap()
def main(cfg: SanaConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, vae, generator, num_replicas, rank, training_start_time
    global load_vae_feat, load_text_feat, text_encoder
    global max_length, validation_prompts, latent_size, valid_prompt_embed_suffix, null_embed_path
    global image_size, cache_file, total_steps, vae_dtype, model_instance
    global assessment
    global train_type

    # [1]: Config & Environments
    ## ============================================================================
    config = cfg
    args = cfg
    # 1.Initialize training mode
    if config.train.use_fsdp:
        set_fsdp_env()
        init_train = "FSDP"
    else:
        init_train = "DDP"
    training_start_time = time.time()
    if args.debug:
        # config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        args.report_to = "tensorboard"
    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # 2. Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        kwargs_handlers=[init_handler],
    )
    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)
    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)
    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb
            wandb.init(project=args.tracker_project_name, name=args.name, resume="allow", id=args.name)

    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for training")

    # [2]: Model
    ## ============================================================================
    image_size = config.model.image_size
    latent_size = int(image_size) // config.vae.vae_downsample_rate
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    max_length = config.text_encoder.model_max_length
    train_type = config.train.train_type
    if train_type == "bf16":
        train_type = torch.bfloat16
    elif train_type == "fp16":
        train_type = torch.float16
    elif train_type == "fp32":
        train_type = torch.float32
    else:
        raise KeyError("Error: Unsupported Train Type: {train_type}")

    ## [2-1]: Loading VAE ...
    vae = None
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    if not config.data.load_vae_feat:
        if config.vae.vae_type == "SDVAE":
            vae = load_vae(config.vae.vae_pretrained, accelerator.device)
            vae = vae.to(train_type).eval()
        else:
            raise KeyError(f"Only support VAE: 'SDVAE', but received {config.vae.vae_type}.")
    logger.info("##############################################################")
    trainable_params = []
    for name, param in vae.named_parameters():
        if param.requires_grad:
            trainable_params.append(".".join(name.split(".")[:2]))
    logger.info(f"VAE type: {config.vae.vae_type}, path: {config.vae.vae_pretrained}, weight_dtype: {vae_dtype}")
    logger.info(f"VAE Params: {sum(p.numel() for p in vae.parameters())/1e6} M, dtype: {next(vae.parameters()).dtype}")
    logger.info(f"Trainable Params: {sorted(set(trainable_params))}")
    logger.info("##############################################################")

    ## [2-2]: Loading Tokenizer ...
    text_encoder = None
    if config.text_encoder.text_encoder_name == "sd35-text":
        text_encoder = load_text_encoder(config.text_encoder.text_encoder_pretrained, accelerator.device)
    elif config.text_encoder.text_encoder_name == "empty":
        text_encoder = None 
    else:
        raise KeyError(f"Only support Text Encoder: 'sd35-text', but received {config.text_encoder.text_encoder_name}.")
    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"
    logger.info("##############################################################")
    logger.info(f"text_encoder type: {config.text_encoder.text_encoder_name}, path: {config.text_encoder.text_encoder_pretrained}")
    logger.info("##############################################################")

    ## [2-3]: Loading DiT model ...
    logger.info("##############################################################")
    logger.info(f"DiT type: {config.model.model}, path: {config.model.model_pretrained}, weight_dtype: {config.model.mixed_precision}")
    if config.model.model == "SD35M_D2C":
        DiT = load_mmdit(
                           config.model.model_pretrained, 
                           config.model.shift, 
                           False, 
                           accelerator.device, 
                           config.model.image_size, 
                           config.model.input_channel
                        ).train().to(accelerator.device)
        if config.train.grad_checkpointing:
            set_grad_checkpoint(DiT, gc_step=config.train.gc_step)
        if config.model.fp32_attention:
            set_fp32_attention(DiT)
    else:
        raise KeyError(f"Only support Model: 'SD35M_D2C', but received {config.model.model}.")
    if config.model.load_from:
        logger.info(f"Loading Pre-trained Weight for SD35M-D2C: {config.model.load_from}")
        state_dict = torch.load(config.model.load_from, map_location='cpu')
        missing, unexpected = DiT.load_state_dict(state_dict["state_dict"], strict=False)
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")
    DiT.to(train_type).train().to(accelerator.device)
    trainable_params = []
    for name, param in DiT.named_parameters():
        if param.requires_grad:
            trainable_params.append(".".join(name.split(".")[:2]))
    logger.info(f"DiT Params: {sum(p.numel() for p in DiT.parameters())/1e6} M, dtype: {next(DiT.parameters()).dtype}")
    logger.info(f"Trainable Params: {sorted(set(trainable_params))}")
    logger.info("##############################################################")

    ## [2-4]: Loading Connector model ...
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
    connector = connector.train().to(train_type).to(accelerator.device)
    trainable_params = []
    for name, param in connector.named_parameters():
        if param.requires_grad:
            trainable_params.append(".".join(name.split(".")[:2]))
    logger.info(f"Connector Params: {sum(p.numel() for p in connector.parameters())/1e6} M, dtype: {next(connector.parameters()).dtype}")
    logger.info(f"Trainable Params: {sorted(set(trainable_params))}")
    logger.info("##############################################################")

    ## [2-5]: Loading IQA model ...
    logger.info("##############################################################")
    logger.info(f"IQA type: {config.assessment.model}, config: {config.assessment.model_config}")
    if config.assessment.model == "SDQA":
        from iqa import DepictQA, load_pretrained_weights
        assert os.path.isfile(config.assessment.model_config)
        ## loading cfg
        iqa_cfg_path = config.assessment.model_config
        with open(iqa_cfg_path, "r") as f:
            iqa_cfg = EasyDict(yaml.safe_load(f))
        ## Model
        assessment = DepictQA(iqa_cfg, training=True)
        assessment = load_pretrained_weights(iqa_cfg, assessment, logger=logger)
    assessment.train().to(train_type).to(accelerator.device)
    trainable_params = []
    for name, param in assessment.named_parameters():
        if param.requires_grad:
            trainable_params.append(".".join(name.split(".")[:2]))
    logger.info(f"IQA Params: {sum(p.numel() for p in assessment.parameters())/1e6} M, dtype: {next(assessment.parameters()).dtype}")
    logger.info(f"Trainable Params: {sorted(set(trainable_params))}")
    logger.info("##############################################################")

    ## [2-6]: combination model
    model = IQAIR(DiT, connector, assessment, accelerator.device).to(train_type)
    logger.info("##############################################################")
    logger.info("Summary: IQAIR")
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(".".join(name.split(".")[:3]))
    logger.info(f"Trainable Params Layers: {sorted(set(trainable_params))}")
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable Params: {round(num_trainable_params / 1e6, 3)}M, all params: {round(num_total_params/1e6, 3)}M, trainable%: {round(num_trainable_params/num_total_params * 100, 4)}%")
    logger.info("##############################################################")

    if config.train.use_fsdp:
        model_instance = deepcopy(model)
    else:
        model_instance = model

    # [3]: build scheduler
    ## ============================================================================
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_flow_v=config.scheduler.predict_flow_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )
    predict_info = (
        f"flow-prediction: {config.scheduler.predict_flow_v}, noise schedule: {config.scheduler.noise_schedule}"
    )
    if "flow" in config.scheduler.noise_schedule:
        predict_info += f", flow shift: {config.scheduler.flow_shift}"
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        predict_info += (
            f", flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
        )
    logger.info(predict_info)

    # [4]: build dataloader
    ## ============================================================================
    config.data.data_dir = config.data.data_dir if isinstance(config.data.data_dir, list) else [config.data.data_dir]
    config.data.data_dir = [
        data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
        for data in config.data.data_dir
    ]
    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dataset = build_dataset(
        asdict(config.data),
        resolution=image_size,
        aspect_ratio_type=config.model.aspect_ratio_type,
        real_prompt_ratio=config.train.real_prompt_ratio,
        max_length=max_length,
        config=config,
        caption_proportion=config.data.caption_proportion,
        sort_dataset=config.data.sort_dataset,
        vae_downsample_rate=config.vae.vae_downsample_rate,
        dset=config.data.dset,
        max_samples=config.data.max_samples,
        caption_type=config.data.caption_type,
        return_meta=config.data.return_meta,
    )
    accelerator.wait_for_everyone()
    if config.model.multi_scale:
        drop_last = True
        uuid = hashlib.sha256("-".join(config.data.data_dir).encode()).hexdigest()[:8]
        cache_dir = osp.expanduser(f"~/.cache/_wids_batchsampler_cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_pattern = (
            f"{cache_dir}/{getpass.getuser()}-{uuid}-sort_dataset{config.data.sort_dataset}"
            f"-hq_only{config.data.hq_only}-valid_num{config.data.valid_num}"
            f"-aspect_ratio{len(dataset.aspect_ratio)}-droplast{drop_last}"
            f"dataset_len{len(dataset)}"
        )
        cache_file = f"{base_pattern}-num_replicas{num_replicas}-rank{rank}"
        for i in config.data.data_dir:
            cache_file += f"-{i}"
        cache_file += ".json"

        if hasattr(dataset, "weights"):
            sampler = DistributedWeightedRangedSampler(dataset=dataset,
                                                       weights=dataset.weights,
                                                       replacement=True, 
                                                       num_replicas=num_replicas,
                                                       rank=rank)
        else:
            sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
        batch_sampler = AspectRatioBatchSampler(
            sampler=sampler,
            dataset=dataset,
            batch_size=config.train.train_batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=drop_last,
            ratio_nums=dataset.ratio_nums,
            config=config,
            valid_num=config.data.valid_num,
            hq_only=config.data.hq_only,
            cache_file=cache_file,
            caching=args.caching,
            clipscore_filter_thres=args.data.del_img_clip_thr,
        )
        if config.data.return_meta:
            train_dataloader = build_dataloader(
                dataset,
                batch_sampler=batch_sampler, 
                num_workers=config.train.num_workers,
                collate_fn=dataset.collate_fn,
            )
        else:
            train_dataloader = build_dataloader(
                dataset, 
                batch_sampler=batch_sampler, 
                num_workers=config.train.num_workers)
        train_dataloader_len = len(train_dataloader)
        logger.info(f"rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
    else:
        if hasattr(dataset, "weights"):
            sampler = DistributedWeightedRangedSampler(dataset=dataset,
                                                       weights=dataset.weights,
                                                       replacement=True, 
                                                       num_replicas=num_replicas,
                                                       rank=rank)
        else:
            sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
        if config.data.return_meta:
            train_dataloader = build_dataloader(
                dataset,
                num_workers=config.train.num_workers,
                batch_size=config.train.train_batch_size,
                shuffle=False,
                sampler=sampler,
                collate_fn=dataset.collate_fn,
            )
        else:
            train_dataloader = build_dataloader(
                dataset,
                num_workers=config.train.num_workers,
                batch_size=config.train.train_batch_size,
                shuffle=False,
                sampler=sampler,
            )
        train_dataloader_len = len(train_dataloader)
    load_vae_feat = getattr(train_dataloader.dataset, "load_vae_feat", False)
    load_text_feat = getattr(train_dataloader.dataset, "load_text_feat", False)

    # [6]: build optimizer and lr scheduler
    ## ============================================================================
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)
    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)
    logger.warning(
        f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.5f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}, qk norm: {config.model.qk_norm}, "
        f"fp32 attn: {config.model.fp32_attention}, attn type: {config.model.attn_type}, ffn type: {config.model.ffn_type}, "
        f"text encoder: {config.text_encoder.text_encoder_name}, captions: {config.data.caption_proportion}, precision: {config.model.mixed_precision}"
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = train_dataloader_len * config.train.num_epochs

    # [7]: Resume training 
    ## ============================================================================
    if config.train.resume_from and os.path.isfile(config.train.resume_from):
        config.train.resume_from = dict(
            checkpoint=config.train.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True,
        )
        rng_state = None

        _, missing, unexpected, _ = load_checkpoint(
            **config.train.resume_from,
            model=model,
            model_ema=None,
            FSDP=config.train.use_fsdp,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            null_embed_path=None,
        )
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")
        path = osp.basename(config.train.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass

    # [8]: Prepare Everyhting for training
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # load everything except model when resume
    if (
        config.train.use_fsdp
        and config.train.resume_from is not None
        and config.train.resume_from["checkpoint"] is not None
        and config.train.resume_from["resume_optimizer"]
        and config.train.resume_from["resume_lr_scheduler"]
    ):
        logger.info(f"FSDP resume: Loading optimizer, scheduler, scaler, random_states...")
        accelerator.load_state(
            os.path.join(config.train.resume_from["checkpoint"], "model"),
            state_dict_key=["optimizer", "scheduler", "scaler", "random_states"],
        )

    set_random_seed((start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"]))
    logger.info(f'Set seed: {(start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"])}')
    logger.info("##############################################################")

    # Start Training
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_diffusion=train_diffusion,
        logger=logger,
    )

if __name__ == "__main__":
    main()
