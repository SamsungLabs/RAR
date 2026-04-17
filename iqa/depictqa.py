import logging
import os

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image, ImageFile
from sentence_transformers import SentenceTransformer
from torch.nn.utils import rnn
# from torch.cuda.amp import autocast
from torch import autocast

from transformers import LlamaTokenizer, StoppingCriteriaList

from .conversations import conversation_dict, system_dict
from .model_llama import LlamaForCausalLM
from .utils import VISION_TAGS, DepictQAStop, cal_confidence_batch

ImageFile.LOAD_TRUNCATED_IMAGES = True
pos, eov, sov = VISION_TAGS["pos"]["img"], VISION_TAGS["eov"], VISION_TAGS["sov"]

def check_inf(name, value):
    if not torch.isfinite(value).all():
        print("\n===================================================")
        print(f"Error: {name} is not finite. There are NAN existing.")   

class DepictQA(nn.Module):
    def __init__(self, args, training):
        super(DepictQA, self).__init__()
        # [0]: Configs
        self.args = args
        self.vision_preprocess_config = args.vision_preprocess
        self.use_encoder = args.model["use_encoder"]  # [clip, SDAE]
        self.training = training
        

        ## Source Vision Encoder
        self.vision_encoder_type = args.model["vision_encoder_type"]
        self.vision_encoder_path = args.model["vision_encoder_path"]
        
        ## Target Vision Encoder
        self.tgt_vision_encoder_type = args.model["tgt_vision_encoder_type"]
        self.tgt_vision_encoder_config = args.model["tgt_vision_encoder_config"]
        if "tgt_vision_encoder_dtype" in args.model:
            if args.model["tgt_vision_encoder_dtype"] == "bfloat16":
                self.tgt_vision_encoder_dtype = torch.bfloat16
            elif args.model["tgt_vision_encoder_dtype"] == "float16":
                self.tgt_vision_encoder_dtype = torch.float16
            elif args.model["tgt_vision_encoder_dtype"] == "float32":
                self.tgt_vision_encoder_dtype = torch.float32
            else:
                raise KeyError(f"Load Vision Encoder dtype should be [bfloat16, float16, float32], but receive {args.model['tgt_vision_encoder_dtype']}")
        else: 
            self.tgt_vision_encoder_dtype = torch.bfloat16
        self.tgt_vision_projector_type = args.model["tgt_vision_projector_type"] # [simple, former]
        
        self.keep_encoder = args.model["keep_encoder"]
        self.end2end = args.model["end2end"]

        ## Feature Type
        self.vision_feature_type = args.model["vision_feature_type"] # [local, global]
        if self.vision_feature_type == "local":
            self.vision_size = 1024
        elif self.vision_feature_type == "global":
            self.vision_size = 768
        else:
            raise KeyError("vision_feature_type only support (global, local), but input is {self.vision_feature_type}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [1]: LLM Model
        ## roles and seps
        self.conversation = conversation_dict[args.model["llm_conv_type"]]
        self.roles = self.conversation.roles
        sep1 = self.conversation.sep
        self.sep2 = (
            self.conversation.sep2 if self.conversation.sep2 else self.conversation.sep
        )
        self.seps = (f" {sep1} ", f" {self.sep2} ")
        self.unique_tag = args.model.get("unique_tag", True)
        ## add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.model.lora["lora_r"],
            lora_alpha=args.model.lora["lora_alpha"],
            lora_dropout=args.model.lora["lora_dropout"],
            target_modules=args.model.lora["lora_target_modules"],
        )

        llm_path = args.model["llm_path"]
        logging.info(f"Initializing LLM from {llm_path} ...")
        self.llm = LlamaForCausalLM.from_pretrained(llm_path)
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        self.tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        logging.info("LLM initialized.")

        # [2]: Source Vision Encoder
        if self.use_encoder == "clip":
            assert self.vision_encoder_type == 'clip', f"Error: Only support source_vision_encoder: CLIP, but receive {self.vision_encoder_type}"
            from .clip import build_abstractor, load_clip
            logging.info(f"Initializing CLIP vision encoder from {self.vision_encoder_path} ...")
            if self.keep_encoder:
                clip_encoder = load_clip(self.vision_encoder_path, training, self.vision_preprocess_config, device=device)
                self.vision_encoder = clip_encoder.visual
                ## freeze vision encoder
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
                self.vision_encoder.eval()
                logging.info("Vision encoder initialized.")
            else:
                self.vision_encoder = None 
                logging.info("Disable Vision Enocder in IQA.")

            ## Vision Abstractor
            self.abstractor = nn.Identity()
            if args.model.get("abstractor", None):
                args.model.abstractor["hidden_dim"] = self.vision_size
                self.abstractor = build_abstractor(args.model["abstractor"])
                logging.info("Vision abstractor initialized.")

        # [3]: Target Vision Encoder
        elif self.use_encoder == "autodir":
            raise NotImplementedError
            # To-Do: self.keep_encoder, check dtype
            # assert self.tgt_vision_encoder_type == 'autodir', f"Error: Only support tgt_vision_encoder_type: AutoDIR, but receive {self.tgt_vision_encoder_type}"  # only stable diffusion's image encoder
            # from .autodir_modules import load_vision_encoder
            # ## Build Vision Encoder
            # logging.info(f"Initializing AutoDIR vision encoder from {self.tgt_vision_encoder_config} ...")
            # self.tgt_vision_encoder = load_vision_encoder(self.tgt_vision_encoder_config, training, self.vision_preprocess_config, device=device)
            # ## freeze vision encoder
            # for param in self.tgt_vision_encoder.parameters():
            #     param.requires_grad = False
            # self.tgt_vision_encoder.eval()
            # logging.info("Target Vision encoder initialized.")
            # ## Vision Abstractor
            # from .autodir_modules import build_abstractor as tgt_build_abstractor
            # self.tgt_abstractor = tgt_build_abstractor(self.tgt_vision_projector_type, args.model["tgt_pos"], args.model["tgt_vision_projector_layers"]).to(device)                  
        
        # [3-2]: Target Vision Encoder
        elif self.use_encoder == 'DC-AE':
            raise NotImplementedError
            # To-Do: self.keep_encoder, check dtype
            # assert self.tgt_vision_encoder_type == 'DC-AE', f"Error: Only support tgt_vision_encoder_type: DC-AE, but receive {self.tgt_vision_encoder_type}"  # only stable diffusion's image encoder
            # from .sana import load_vision_encoder
            # from .sana import build_abstractor as tgt_build_abstractor
            # ## Build Vision Encoder
            # logging.info(f"Initializing DC-AE vision encoder from {self.tgt_vision_encoder_config} ...")
            # self.tgt_vision_encoder = load_vision_encoder(self.tgt_vision_encoder_config, training, self.vision_preprocess_config, device=device)
            # ## freeze vision encoder
            # for param in self.tgt_vision_encoder.parameters():
            #     param.requires_grad = False
            # self.tgt_vision_encoder.eval()
            # logging.info("Target Vision encoder initialized.")
            # ## Vision Abstractor
            # self.tgt_abstractor = tgt_build_abstractor(self.tgt_vision_projector_type, args.model["tgt_pos"], args.model["tgt_vision_projector_layers"]).to(device)                  

        # [3-3]: Target Vision Encoder
        elif self.use_encoder == 'SDVAE':
            assert self.tgt_vision_encoder_type == 'SDVAE', f"Error: Only support tgt_vision_encoder_type: SDVAE, but receive {self.tgt_vision_encoder_type}"  # only stable diffusion's image encoder
            if self.keep_encoder:
                from .sd35 import load_vision_encoder
                ## Build Vision Encoder
                logging.info(f"Initializing SDVAE vision encoder from {self.tgt_vision_encoder_config} ...")
                self.tgt_vision_encoder = load_vision_encoder(self.tgt_vision_encoder_config, training, self.vision_preprocess_config, device=device, dtype=self.tgt_vision_encoder_dtype)
                ## freeze vision encoder
                for param in self.tgt_vision_encoder.parameters():
                    param.requires_grad = False
                self.tgt_vision_encoder.eval()
                logging.info("Target Vision encoder initialized.")
            else:
                self.tgt_vision_encoder = None 
                logging.info("Disable Vision Enocder in IQA.")
            ## Vision Abstractor
            from .sd35 import build_abstractor as tgt_build_abstractor
            self.tgt_abstractor = tgt_build_abstractor(self.tgt_vision_projector_type, args.model["tgt_pos"], args.model["tgt_vision_projector_layers"]).to(device) # .to(self.tgt_vision_encoder_dtype)              

        ## Vision Projector
        self.vision_proj = nn.Linear(self.vision_size, self.llm.config.hidden_size)
        logging.info("Vision projection layer initialized.")

        self.max_tokens = args.train.get("max_tokens", 512)
        self.device = torch.cuda.current_device()
        self.dictionary = {}
        # print("self.LLM.dtype:", self.llm.dtype)
        # if self.tgt_vision_encoder:
        #     print("self.tgt_vision_encoder.dtype:", next(self.tgt_vision_encoder.parameters()).dtype)
        # print("self.tgt_abstractor.dtype:", next(self.tgt_abstractor.parameters()).dtype)
        # print("self.vision_proj.dtype:", next(self.vision_proj.parameters()).dtype)


    def emb_img(self, imgs):
        imgs = imgs.to(self.llm.dtype).to(self.device)
        if self.use_encoder  == 'clip':
            raise NotImplementedError
            # if self.vision_feature_type == "global":
            #     with torch.no_grad():
            #         img_embs = self.vision_encoder(imgs)  # [B, C]
            #     img_embs = self.vision_proj(img_embs).unsqueeze(1)  # [B, 1, C]
            # elif self.vision_feature_type == "local":
            #     with torch.no_grad():
            #         img_embs = self.vision_encoder.forward_patch_features(imgs)  # [B, N, C]
            #     img_embs = self.vision_proj(self.abstractor(img_embs))  # [B, N, C]
                
        elif self.use_encoder  == 'autodir':
            raise NotImplementedError
            ## input: torch.Size([1, 3, 80, 80]) torch.float16   
            # with torch.no_grad(), autocast("cuda"):
            #     posterior = self.tgt_vision_encoder.encode(imgs)  # [B, 4, H/8, W/8]
            #     img_embs = posterior.mode()
            # if self.vision_feature_type == "global":
            #     img_embs = self.tgt_abstractor(img_embs).unsqueeze(1) # [B, 1, C]
            # elif self.vision_feature_type == "local":
            #     img_embs = self.tgt_abstractor.forward_patch_features(img_embs) # [B, N, C] # abstractor: torch.Size([1, 25, 1024]) torch.float16
            # img_embs = self.vision_proj(img_embs)  # [B, N, C] # projector: torch.Size([1, 25, 4096]) torch.float16

        elif self.use_encoder  == 'DC-AE':
            raise NotImplementedError
            # # input: torch.Size([1, 3, 512, 512]) torch.float16   
            # check_inf("DC-AE inputs", imgs)
            # ae = self.tgt_vision_encoder
            # scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
            # img_embs = ae.encode(imgs.to(self.tgt_vision_encoder.device))[0]
            # check_inf("DC-AE output", img_embs)
            # img_embs = img_embs * scaling_factor # [B, 32, H/32, W/32]
            # if self.vision_feature_type == "global":
            #     img_embs = self.tgt_abstractor(img_embs).unsqueeze(1) # [B, 1, C]
            # elif self.vision_feature_type == "local":
            #     img_embs = self.tgt_abstractor.forward_patch_features(img_embs) # [B, N, C] # abstractor: torch.Size([1, N, 1024]) torch.float16
            # check_inf("DC-AE abstractor", img_embs)
            # img_embs = self.vision_proj(img_embs)  # [B, N, C] # projector: torch.Size([1, N, 4096]) torch.float16
            # check_inf("DC-AE projector", img_embs)

        elif self.use_encoder  == 'SDVAE':
            device = next(self.tgt_vision_encoder.parameters()).device
            abs_device = next(self.tgt_vision_encoder.parameters()).device
            check_inf("SDVAE inputs", imgs)
            img_embs = self.tgt_vision_encoder.encode(imgs.to(device))
            img_embs = self.tgt_vision_encoder.process_in(img_embs)
            if self.vision_feature_type == "global":
                img_embs = self.tgt_abstractor(img_embs.to(abs_device)).unsqueeze(1) # [B, 1, C]
            elif self.vision_feature_type == "local":
                img_embs = self.tgt_abstractor.forward_patch_features(img_embs.to(abs_device)) # [B, N, C] # abstractor: torch.Size([1, N, 1024]) torch.float16
            check_inf("SDVAE abstractor", img_embs)
            img_embs = self.vision_proj(img_embs)  # [B, N, C] # projector: torch.Size([1, N, 4096]) torch.float16
            check_inf("SDVAE projector", img_embs)
        return img_embs

    def emb_latent(self, latents):
        latents = latents.to(self.llm.dtype).to(self.device)
        if self.use_encoder  == 'autodir':
            raise NotImplementedError
            # img_embs = latents / 0.18215
            # if self.vision_feature_type == "global":
            #     img_embs = self.tgt_abstractor(img_embs).unsqueeze(1) # [B, 1, C]
            # elif self.vision_feature_type == "local":
            #     img_embs = self.tgt_abstractor.forward_patch_features(img_embs) # [B, N, C] # abstractor: torch.Size([1, 25, 1024]) torch.float16
            # img_embs = self.vision_proj(img_embs)  # [B, N, C] # projector: torch.Size([1, 25, 4096]) torch.float16

        elif self.use_encoder  == 'DC-AE':
            raise NotImplementedError
            # input: torch.Size([1, 3, 512, 512]) torch.float16   
            # check_inf("DC-AE inputs", latents)
            # img_embs = latents
            # if self.vision_feature_type == "global":
            #     img_embs = self.tgt_abstractor(img_embs).unsqueeze(1) # [B, 1, C]
            # elif self.vision_feature_type == "local":
            #     img_embs = self.tgt_abstractor.forward_patch_features(img_embs) # [B, N, C] # abstractor: torch.Size([1, N, 1024]) torch.float16
            # check_inf("DC-AE abstractor", img_embs)
            # img_embs = self.vision_proj(img_embs)  # [B, N, C] # projector: torch.Size([1, N, 4096]) torch.float16
            # check_inf("DC-AE projector", img_embs)

        elif self.use_encoder  == 'SDVAE':
            # input: torch.Size([1, 3, 256, 256]) torch.float16   
            check_inf("SDVAE inputs", latents)
            img_embs = latents
            if self.vision_feature_type == "global":
                img_embs = self.tgt_abstractor(img_embs).unsqueeze(1) # [B, 1, C]
            elif self.vision_feature_type == "local":
                img_embs = self.tgt_abstractor.forward_patch_features(img_embs) # [B, N, C] # abstractor: torch.Size([1, N, 1024]) torch.float16
            check_inf("SDVAE abstractor", img_embs)
            img_embs = self.vision_proj(img_embs)  # [B, N, C] # projector: torch.Size([1, N, 4096]) torch.float16
            check_inf("SDVAE projector", img_embs)
        return img_embs

    def emb_text(self, text, bsz):
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.device
        )
        embs = self.llm.model.model.embed_tokens(tokens.input_ids).expand(bsz, -1, -1)
        return embs

    def fuse_vision_iqa(self, vision_embs, input_ids, tgt_ids, attn_mask, task_type):
        img_embs, img_A_embs, img_B_embs = vision_embs
        input_ids = input_ids.to(self.device)  # [B, N2]
        tgt_ids = tgt_ids.to(self.device)  # [B, N2]
        attn_mask = attn_mask.to(self.device)  # [B, N2]
        bsz = input_ids.shape[0]

        system = self.get_system(task_type=task_type)
        no_ref = "noref" in task_type
        if "quality_compare" in task_type:
            if self.unique_tag and no_ref:
                systems = [
                    [system + self.seps[1] + "Image A: " + sov["A"]],
                    [eov["A"] + "\n\n" + self.seps[1] + "Image B: " + sov["B"]],
                ]
            elif self.unique_tag and not no_ref:
                systems = [
                    [system + self.seps[1] + "Reference Image: " + sov["ref"]],
                    [eov["ref"] + "\n\n" + self.seps[1] + "Image A: " + sov["A"]],
                    [eov["A"] + "\n\n" + self.seps[1] + "Image B: " + sov["B"]],
                ]
            elif not self.unique_tag and no_ref:
                systems = [
                    [system + self.seps[1] + "Image A: " + sov["img"]],
                    [eov["img"] + "\n\n" + self.seps[1] + "Image B: " + sov["img"]],
                ]
            else:
                assert not self.unique_tag and not no_ref
                systems = [
                    [system + self.seps[1] + "Reference Image: " + sov["img"]],
                    [eov["img"] + "\n\n" + self.seps[1] + "Image A: " + sov["img"]],
                    [eov["img"] + "\n\n" + self.seps[1] + "Image B: " + sov["img"]],
                ]
            if no_ref:
                vision_embs = [self.emb_text(text, bsz) for text in systems]
                vision_embs.insert(2, img_B_embs)
                vision_embs.insert(1, img_A_embs)
                vision_embs = torch.cat(vision_embs, dim=1)
            else:
                vision_embs = [self.emb_text(text, bsz) for text in systems]
                vision_embs.insert(3, img_B_embs)
                vision_embs.insert(2, img_A_embs)
                vision_embs.insert(1, img_embs)
                vision_embs = torch.cat(vision_embs, dim=1)
        elif "quality_single" in task_type:
            if self.unique_tag and no_ref:
                systems = [
                    [system + self.seps[1] + "Image: " + sov["img"]],
                ]
            elif self.unique_tag and not no_ref:
                systems = [
                    [system + self.seps[1] + "Reference Image: " + sov["ref"]],
                    [eov["ref"] + "\n\n" + self.seps[1] + "Image: " + sov["img"]],
                ]
            elif not self.unique_tag and no_ref:
                systems = [
                    [system + self.seps[1] + "Image: " + sov["img"]],
                ]
            else:
                assert not self.unique_tag and not no_ref
                systems = [
                    [system + self.seps[1] + "Reference Image: " + sov["img"]],
                    [eov["img"] + "\n\n" + self.seps[1] + "Image: " + sov["img"]],
                ]
            vision_embs = [self.emb_text(text, bsz) for text in systems]
            if "quality_single_A" in task_type:
                img_AB_embs = img_A_embs
            elif "quality_single_B" in task_type:
                img_AB_embs = img_B_embs
            if no_ref:
                vision_embs.insert(1, img_AB_embs)
                vision_embs = torch.cat(vision_embs, dim=1)
            else:
                vision_embs.insert(2, img_AB_embs)
                vision_embs.insert(1, img_embs)
                vision_embs = torch.cat(vision_embs, dim=1)
        else:
            raise ValueError

        # construct embs
        bos = (
            torch.ones([bsz, 1], dtype=torch.long, device=self.device)
            * self.tokenizer.bos_token_id
        )
        bos_embs = self.llm.model.model.embed_tokens(bos)  # [B, 1, C]
        input_embs = self.llm.model.model.embed_tokens(input_ids)  # [B, N2, C]
        input_embs = torch.cat(
            [bos_embs, vision_embs, input_embs], dim=1
        )  # [B, 1+N1+Nimg+N2, C]

        # construct targets, -100 means no loss on such tokens
        n = 1 + vision_embs.shape[1]
        pre_tgt_ids = torch.ones((bsz, n), dtype=torch.long).to(self.device).fill_(-100)
        targets = torch.cat([pre_tgt_ids, tgt_ids], dim=1)
        assert input_embs.shape[:2] == targets.shape  # [B, 1+N1+Nimg+N2]

        # construct attn mask
        mask_bos = torch.ones([bsz, 1], dtype=torch.long).to(self.device)
        n = vision_embs.shape[1]
        mask_vision = torch.ones((bsz, n), dtype=torch.long).to(self.device)
        attn_mask = torch.cat([mask_bos, mask_vision, attn_mask], dim=1)
        assert attn_mask.shape == targets.shape  # [B, 1+N1+Nimg+N2]

        return input_embs, targets, attn_mask

    def fuse_vision(self, img_embs, input_ids, tgt_ids, attn_mask, task_type):
        input_ids = input_ids.to(self.device)  # [B, N2]
        tgt_ids = tgt_ids.to(self.device)  # [B, N2]
        attn_mask = attn_mask.to(self.device)  # [B, N2]
        bsz = img_embs.shape[0]

        system = self.get_system(task_type=task_type)
        system_embs = self.emb_text(system, bsz)  # [B, N1, C]

        # construct embs
        bos = (
            torch.ones([bsz, 1], dtype=torch.long, device=self.device)
            * self.tokenizer.bos_token_id
        )
        bos_embs = self.llm.model.model.embed_tokens(bos)  # [B, 1, C]
        input_embs = self.llm.model.model.embed_tokens(input_ids)  # [B, N2, C]
        input_embs = torch.cat(
            [bos_embs, system_embs, img_embs, input_embs], dim=1
        )  # [B, 1+N1+Nimg+N2, C]

        # construct targets, -100 means no loss on such tokens
        n = 1 + system_embs.shape[1] + img_embs.shape[1]
        pre_tgt_ids = torch.ones((bsz, n), dtype=torch.long).to(self.device).fill_(-100)
        targets = torch.cat([pre_tgt_ids, tgt_ids], dim=1)
        assert input_embs.shape[:2] == targets.shape  # [B, 1+N1+Nimg+N2]

        # construct attn mask
        mask_bos = torch.ones((bsz, 1), dtype=torch.long).to(self.device)
        n = system_embs.shape[1]
        mask_system = torch.ones((bsz, n), dtype=torch.long).to(self.device)
        n = img_embs.shape[1]
        mask_img = torch.ones((bsz, n), dtype=torch.long).to(self.device)
        attn_mask = torch.cat([mask_bos, mask_system, mask_img, attn_mask], dim=1)
        assert attn_mask.shape == targets.shape  # [B, 1+N1+Nimg+N2]

        return input_embs, targets, attn_mask

    def get_system(self, task_type):
        PROMPT_START = ""
        if "quality" not in task_type:
            PROMPT_START = f'{self.seps[1]}{self.roles[0]}: {VISION_TAGS["sov"]["img"]}'
        system = system_dict[task_type] + "\n\n" + PROMPT_START
        return system

    def tokenize_conv(self, conversations, task_type):
        input_ids, tgt_ids = [], []
        for conversation in conversations:
            input_id, tgt_id = self._tokenize_conv(conversation, task_type)
            input_ids.append(torch.LongTensor(input_id))
            tgt_ids.append(torch.LongTensor(tgt_id))
        pad_id = self.tokenizer.pad_token_id
        input_ids = rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        tgt_ids = rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=-100)
        assert input_ids.shape == tgt_ids.shape
        input_ids = input_ids[:, : self.max_tokens]
        tgt_ids = tgt_ids[:, : self.max_tokens]
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        assert attn_mask.shape == input_ids.shape
        return input_ids, tgt_ids, attn_mask.long()

    def _tokenize_conv(self, conversation, task_type):
        input_ids, tgt_ids = [], []
        for idx in range(len(conversation)):
            if idx > 1:
                break
            turn = conversation[idx]
            role = turn["from"]
            if idx == 0:  # the first human turn
                assert role == "human"
                if self.unique_tag and "quality_compare" in task_type:
                    text = f'{eov["B"]}\n\n{self.seps[1]}{self.roles[0]}: {turn["value"]}\n{self.seps[0]}{self.roles[1]}: '
                elif "quality" in task_type:
                    text = f'{eov["img"]}\n\n{self.seps[1]}{self.roles[0]}: {turn["value"]}\n{self.seps[0]}{self.roles[1]}: '
                else:
                    turn["value"] = (
                        turn["value"].replace(f"{pos}\n", "").replace(f"\n{pos}", "")
                    )
                    text = (
                        f'{eov["img"]} {turn["value"]}{self.seps[0]}{self.roles[1]}: '
                    )
                input_id = self.tokenizer(text, add_special_tokens=False).input_ids
                input_ids += input_id
                tgt_ids += [-100] * len(input_id)  # no loss on prompt
            else:
                if role == "human":
                    text = f'{self.roles[0]}: {turn["value"]}\n{self.seps[0]}{self.roles[1]}: '
                    input_id = self.tokenizer(text, add_special_tokens=False).input_ids
                    input_ids += input_id
                    tgt_ids += [-100] * len(input_id)  # no loss on prompt
                else:
                    assert role == "gpt"
                    text = turn["value"] + "\n" + self.seps[1].rstrip()
                    input_id = self.tokenizer(text, add_special_tokens=False).input_ids
                    input_ids += input_id
                    tgt_ids += input_id
            assert len(input_ids) == len(tgt_ids)
        return input_ids, tgt_ids

    def forward(self, inputs, latent_input=False):
        """
            {
                "img": imgs,    
                "img_A": imgs_A,
                "img_B": imgs_B, 
                "img_path": img_path,
                "img_A_path": img_A_path,
                "img_B_path": img_B_path,
                "conversation": meta["conversations"],
                "task_type": meta["task_type"],
            }
        """
        if not self.keep_encoder and not latent_input:
            raise KeyError("Error: IQA is loading in Latent Mode, please encode the image in advance!")

        bsz = len(inputs["task_type"])
        assert bsz == 1, "quality tasks only support bsz == 1"
        task_type = inputs["task_type"][0]
        is_iqa = False  # check is_iqa
        if "quality" in task_type:
            is_iqa = True

        img_paths = inputs["img_path"]
        img_A_paths = inputs["img_A_path"]
        img_B_paths = inputs["img_B_path"]
        if latent_input:
            img_embs = self.emb_latent(inputs["img"]) if img_paths[0] else None
            img_A_embs = self.emb_latent(inputs["img_A"]) if img_A_paths[0] else None
            img_B_embs = self.emb_latent(inputs["img_B"]) if img_B_paths[0] else None
        else: 
            img_embs = self.emb_img(inputs["img"]) if img_paths[0] else None
            img_A_embs = self.emb_img(inputs["img_A"]) if img_A_paths[0] else None
            img_B_embs = self.emb_img(inputs["img_B"]) if img_B_paths[0] else None

        for key, value in {'ref': img_embs, 'img_A_embs': img_A_embs, 'img_B_embs': img_B_embs}.items():
            if value is not None:
                if not torch.isfinite(value).all():
                    print("\n===================================================")
                    print(f"Error: {key} is not finite. There are NAN existing.")

        conversations = inputs["conversation"]
        input_ids, tgt_ids, attn_mask = self.tokenize_conv(conversations, task_type)
        if is_iqa:
            inputs_embs, targets, attn_mask = self.fuse_vision_iqa(
                vision_embs=[img_embs, img_A_embs, img_B_embs],
                input_ids=input_ids,
                tgt_ids=tgt_ids,
                attn_mask=attn_mask,
                task_type=task_type,
            )
        else:
            inputs_embs, targets, attn_mask = self.fuse_vision(
                img_embs=img_embs,
                input_ids=input_ids,
                tgt_ids=tgt_ids,
                attn_mask=attn_mask,
                task_type=task_type,
            )
            
        if not torch.isfinite(inputs_embs).all():
            print("\n===================================================")
            print(f"Error: fuse_embed is not finite. There are NAN existing.")
        
        outputs = self.llm(
            inputs_embeds=inputs_embs,
            attention_mask=attn_mask,
            return_dict=True,
            output_hidden_states=self.end2end,
            labels=targets,
        )

        if self.end2end:
            # print("hidden_state:", len(outputs.hidden_states), outputs.hidden_states[-1].shape)
            last_hidden = outputs.hidden_states[-1]
            return last_hidden
        else:
            if not torch.isfinite(outputs.logits).all():
                print("\n===================================================")
                print(f"Error: LLM.outputs is not finite. There are NAN existing.")
            
            loss = outputs.loss
            if not torch.isfinite(loss).all():
                print("\n===================================================")
                print(f"Error: loss is not finite. There are NAN existing.")

            # calculate token accuarcy
            preds = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, N-2]
            gts = targets[:, 2:]  # next token pred, 0 is bos, 1->2 ... N-1->N, begin from 2
            acc = (preds.reshape(-1) == gts.reshape(-1)).to(torch.long)  # [B(N-2), ]
            mask = (gts != -100).reshape(-1)
            acc_mask = acc & mask
            acc = acc_mask.sum().item() / mask.sum().item()
            return loss, acc

    def tokenize_query(self, queries, task_type):
        if self.unique_tag and "quality_compare" in task_type:
            eov_final = eov["B"]
        else:
            eov_final = eov["img"]
        texts = [
            f"{eov_final}\n\n{self.seps[1]}{self.roles[0]}: {query}\n{self.seps[0]}{self.roles[1]}: "
            for query in queries
        ]
        input_tokens = self.tokenizer(
            texts,
            padding="longest",  # padding right
            return_length=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        len_mask = input_tokens.length.max() - input_tokens.length
        input_ids = input_tokens.input_ids
        return input_ids, len_mask

    def get_generate_embs(self, inputs, latent_input=False):
        queries = inputs["query"]  # questions from user
        task_type = inputs["task_type"]
        input_ids, len_mask = self.tokenize_query(queries, task_type)
        bsz, n = input_ids.shape[:2]
        tgt_ids = torch.ones((bsz, n), dtype=torch.long).to(self.device).fill_(-100)
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        assert attn_mask.shape == input_ids.shape  # [B, N2]
        
        if latent_input:
            img_embs = self.emb_latent(inputs["img"]) if inputs["img_path"][0] else None
            img_A_embs = self.emb_latent(inputs["img_A"]) if inputs["img_A_path"][0] else None
            img_B_embs = self.emb_latent(inputs["img_B"]) if inputs["img_B_path"][0] else None
        else:
            img_embs = self.emb_img(inputs["img"]) if inputs["img_path"][0] else None
            img_A_embs = self.emb_img(inputs["img_A"]) if inputs["img_A_path"][0] else None
            img_B_embs = self.emb_img(inputs["img_B"]) if inputs["img_B_path"][0] else None

        input_embs, _, _ = self.fuse_vision_iqa(
            vision_embs=[img_embs, img_A_embs, img_B_embs],
            input_ids=input_ids,
            tgt_ids=tgt_ids,
            attn_mask=attn_mask,
            task_type=task_type,
        )

        # move right pad to left
        len_id = input_embs.shape[1] - len_mask
        input_embs_ = torch.zeros_like(input_embs)
        bsz, n = input_embs.shape[:2]
        attn_mask = torch.ones((bsz, n), dtype=torch.long).to(self.device)
        for idx in range(bsz):
            """
            A small proportion of samples (around 0.1%) suffer from stuck for batch inference.
            This is because the padding operation in batch inference. Two ways to solve:
            1. Set batch size = 1.
            2. Do not mask padding tokens if batch size is larger than 1 (as follows).
            """
            # attn_mask[idx, : -len_id[idx]] = 0
            input_embs_[idx, -len_id[idx] :, :] = input_embs[idx, : len_id[idx], :]
            input_embs_[idx, : -len_id[idx], :] = input_embs[idx, len_id[idx] :, :]
        return input_embs_, attn_mask

    def generate(self, inputs, latent_input=False, save_hidden=False):
        input_embs, attn_mask = self.get_generate_embs(inputs, latent_input=latent_input)
        stopping_criteria = StoppingCriteriaList(
            [DepictQAStop([self.sep2], self.tokenizer, input_embs)]
        )
        output_prob_id = inputs.get("output_prob_id", None)
        output_confidence = inputs.get("output_confidence", None)
        if output_confidence and not hasattr(self, "sentence_model"):
            self.sentence_model = None
            sentence_model = inputs.get("sentence_model", "")
            if os.path.exists(sentence_model):
                self.sentence_model = SentenceTransformer(
                    sentence_model, trust_remote_code=True
                )

        return_dict = output_prob_id or output_confidence
        do_sample = True if inputs["temperature"] > 1e-4 else False
        inputs_generate = dict(
            inputs_embeds=input_embs,
            attention_mask=attn_mask,
            max_new_tokens=inputs["max_new_tokens"],
            do_sample=do_sample,
            streamer=inputs.get("streamer", None),
            use_cache=True,
            output_scores=return_dict,
            output_hidden_states=save_hidden, 
            return_dict_in_generate=return_dict,
            stopping_criteria=stopping_criteria,
        )
        if do_sample:
            inputs_generate["temperature"] = inputs["temperature"]
            inputs_generate["top_p"] = inputs["top_p"]

        outputs = self.llm.generate(**inputs_generate) # sequences, scores, attentions, hidden_states

        bsz = input_embs.shape[0]
        if return_dict:
            output_ids = outputs.sequences  # B x (N + 1)
            num_tokens = output_ids.shape[1] 
            output_texts = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )

            logits = outputs.scores
            logits = [
                torch.softmax(logit, dim=-1) for logit in logits
            ]  # [B x V] with length N
            assert len(logits) == num_tokens and logits[0].shape[0] == bsz
            output_probs = torch.zeros((bsz, num_tokens))  # B x N
            for idx_batch in range(bsz):
                for idx_token in range(num_tokens):
                    idx_pred = output_ids[idx_batch][idx_token]
                    output_probs[idx_batch][idx_token] = logits[idx_token][idx_batch][
                        idx_pred
                    ]
        else:
            output_texts = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        confidences = [None] * bsz
        if output_confidence:
            self.dictionary[output_texts[0]] = [output_ids, output_probs]
            confidences = cal_confidence_batch(
                self, inputs, output_texts, output_ids, output_probs
            )
        if not output_prob_id:
            output_ids = output_probs = [None] * bsz

        
        
        if save_hidden:
            last_layer_state = [hs_step[-1][:, -1, :] for hs_step in outputs.hidden_states[1:]]
            last_layer_state = torch.stack(last_layer_state, dim=1)
            # print("="*100)
            # print("input_embs:", input_embs.shape)
            # print("max_new_tokens:", inputs["max_new_tokens"])
            # print("sequences:", outputs.sequences.shape)
            # print("scores:", outputs.scores[0].shape)
            # print("last_layer_state:", last_layer_state.shape)
            # print("hidden_states:", len(outputs.hidden_states), len(outputs.hidden_states[0]))
            # print("texts:", output_texts)
            # print("="*100)
            return output_texts, output_ids, output_probs, confidences, last_layer_state, outputs.hidden_states[0][-1]

        return output_texts, output_ids, output_probs, confidences
