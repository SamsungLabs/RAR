import datetime
# import logging
import os
import types
from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from easydict import EasyDict
from .depictqa import DepictQA

def load_iqa(cfg_path):
    ## loading cfg
    with open(cfg_path, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    ## Model
    model = DepictQA(cfg, training=True)
    model = load_pretrained_weights(cfg, model)
    return model

def load_pretrained_weights(args, model, device="cuda", logger=None):
    # load checkpoint from pretrained model.
    model_state_dict = model.state_dict()
    scratch_weights_dict = list(model_state_dict.keys())
    delta_path = args.model["delta_path"]
    if os.path.exists(delta_path) and args.model["resume"]: # train from pre-trained models
        delta_ckpt = torch.load(delta_path, map_location=torch.device("cpu"))
        matched_state_dict = {}
        for name, param in delta_ckpt.items():
            if name in model_state_dict:
                if model_state_dict[name].shape == param.shape:
                    matched_state_dict[name] = param
                    if name in scratch_weights_dict:
                        scratch_weights_dict.remove(name)
                else:
                    print(f"pretrained weight {name} not matched!")
        model.load_state_dict(matched_state_dict, strict=False)
        if logger:
            logger.info(f"[!] Load pretrained delta ckpt from {delta_path}")
        else:
            print(f"[!] Load pretrained delta ckpt from {delta_path}")
    else:
        if logger:
            logger.info(f"[!] Train Whole Model from scratch, delta_path ({delta_path}) not exist")
        else:
            print(f"[!] Train Whole Model from scratch, delta_path ({delta_path}) not exist")
        print("=========================================")
        print(" Train Whole Model from scratch ...\n")

    # load checkpoint from pretrained model.
    tgt_proj_path = args.model["tgt_vision_projector_weight"]
    if tgt_proj_path and args.model["resume"]:
        tgt_ckpt = torch.load(tgt_proj_path, map_location=torch.device("cpu"))
        matched_state_dict2 = {}
        for name, param in tgt_ckpt.items():
            if name in model_state_dict:
                if model_state_dict[name].shape == param.shape:
                    matched_state_dict2[name] = param
                    if name in scratch_weights_dict:
                        scratch_weights_dict.remove(name)
                else:
                    print("pretrained target projector weight {name} not matched!")
        model.load_state_dict(matched_state_dict2, strict=False)
        if logger:
            logger.info(f"[!] Load pretrained delta ckpt from {tgt_proj_path}")
        else:
            print(f"[!] Load pretrained delta ckpt from {tgt_proj_path}")
    else:
        if logger:
            logger.info(f"[!] Train from scratch, tgt_proj_path ({tgt_proj_path}) not exist")
        else:
            print(f"[!] Train from scratch, tgt_proj_path ({tgt_proj_path}) not exist")
        print("="*100)
        print(" Train Target Projector from scratch ...")

    # load abstractor and projector weights
    if args.model["checkpoints"]: # train from pre-trained models
        if os.path.exists(args.model["checkpoints"]) and args.model["resume"]:
            pretrained_ckpt = torch.load(args.model["checkpoints"], map_location=torch.device("cpu"))
            matched_state_dict3 = {}
            for name, param in pretrained_ckpt.items():
                if name in model_state_dict:
                    if model_state_dict[name].shape == param.shape:
                        matched_state_dict3[name] = param
                        if name in scratch_weights_dict:
                            scratch_weights_dict.remove(name)
                    else:
                        print("pretrained weight {name} not matched!")
            model.load_state_dict(matched_state_dict3, strict=False)
            if logger:
                logger.info(f"[!] Load pretrained abstractor and projector ckpt from {args.model['checkpoints']}")
            else:
                print(f"[!] Load pretrained abstractor and projector ckpt from {args.model['checkpoints']}")
    # check weights
    if len(scratch_weights_dict) > 0:
        for name in scratch_weights_dict:
            if "vision_encoder" in name:
                continue
            if "llm" in name and "lora" not in name: 
                continue
            if logger:
                logger.info(f"{name} is trained from scratch!")
            else:
                print(f"{name} is trained from scratch!")

    # print("="*100)
    # print("Trainable Parameters ...")
    # logger.info("="*100)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"\% {name}: {param.shape} is trainable.")
    #         logger.info(f"\% {name}: {param.shape} is trainable.")
    # total_params = sum(p.numel() for p in model.parameters()) # may add if p.requires_grad
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("IQA: total_params:", total_params, " trainable params:", trainable_params, " trainable ratio:", round(trainable_params/total_params*100,2), "%")
    # logger.info(f"IQA: total_params: {total_params},  trainable params: {trainable_params},  trainable ratio: {round(trainable_params/total_params*100,2)}%")
    # print("="*100)
    return model

def save_model(save_dir, ddp_model, epoch=None, step=None):
    os.makedirs(save_dir, exist_ok=True)
    # only save trainable parameters
    trainable_params = [
        k for (k, v) in ddp_model.module.named_parameters() if v.requires_grad
    ]
    # state_dict on rank 0
    # NOTE: state_dict is still none in other processes
    state_dict = ddp_model.module.state_dict()
    # only save ckpt in rank 0
    ckpt = OrderedDict((k, state_dict[k]) for k in trainable_params)
    if epoch is None:
        torch.save(ckpt, os.path.join(save_dir, "ckpt.pt"))
    elif step is None:
        torch.save(ckpt, os.path.join(save_dir, f"ckpt_epoch{epoch}.pt"))
    else:
        torch.save(
            ckpt, os.path.join(save_dir, f"ckpt_epoch{epoch}_step{step}.pt")
        )
        
    # save tokenizer
    ddp_model.module.tokenizer.save_pretrained(save_dir)
    # save configuration
    ddp_model.module.llm.config.save_pretrained(save_dir)
    # logging.info(f"[!] Save model in {save_dir}")