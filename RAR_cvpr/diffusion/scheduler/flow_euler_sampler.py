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

import os

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from tqdm import tqdm


class FlowEuler:
    def __init__(self, model_fn, condition, uncondition, cfg_scale, model_kwargs):
        self.model = model_fn
        self.condition = condition
        self.uncondition = uncondition
        self.cfg_scale = cfg_scale
        self.model_kwargs = model_kwargs
        # repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    def sample(self, latents, steps=28, do_classifier_free_guidance=True):
        device = next(self.model.parameters()).device 
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, steps, device, None)
        if self.cfg_scale == 1.0:
            do_classifier_free_guidance = False
        else:
            do_classifier_free_guidance = do_classifier_free_guidance

        for i, t in tqdm(list(enumerate(timesteps)), disable=os.getenv("DPM_TQDM", "False") == "True"):
            # condition forward:
            ## expand the latents if we are doing classifier free guidance
            latent_model_input = latents.clone()
            ## broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            ## forward
            self.model_kwargs["y"]=self.condition["y"]
            self.model_kwargs["context"]=self.condition["c_crossattn"]
            cond_noise_pred = self.model(
                latent_model_input,
                timestep,
                **self.model_kwargs,
            )
            if isinstance(cond_noise_pred, Transformer2DModelOutput):
                cond_noise_pred = cond_noise_pred[0]
    
            # perform guidance
            if do_classifier_free_guidance:
                # uncondition forward:
                ## expand the latents if we are doing classifier free guidance
                latent_model_input2 = latents
                ## broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input2.shape[0])
                ## forward
                self.model_kwargs["y"]=self.uncondition["y"]
                self.model_kwargs["context"]=self.uncondition["c_crossattn"]
                uncond_noise_pred = self.model(
                    latent_model_input2,
                    timestep,
                    **self.model_kwargs,
                )
                if isinstance(uncond_noise_pred, Transformer2DModelOutput):
                    uncond_noise_pred = uncond_noise_pred[0]
                noise_pred = uncond_noise_pred + self.cfg_scale * (cond_noise_pred - uncond_noise_pred)
            else:
                noise_pred = cond_noise_pred

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]  # result = xt - output * sigma

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        return latents
