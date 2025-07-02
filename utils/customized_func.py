import os
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np

from einops import rearrange
from typing import Callable, List, Optional, Union, Tuple

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

from animatediff.utils.scheduling_ddim_inverse import DDIMInverseScheduler

import imageio
from PIL import Image

import torch.fft as fft

class MySelfAttnProcessor:
    def __init__(self, save_path):
        self.q = None
        self.k = None
        self.time = None
        self.save_path = save_path
        os.makedirs(self.save_path,exist_ok=True)
    
    def save_qk(self,query,key,blk_name):
        path = os.path.join(self.save_path,f'{blk_name}_{self.time_step.item()}.pth')
        torch.save({'q':query,'k':key},path)

    def save_qkv(self,query,key,value,blk_name):
        path = os.path.join(self.save_path,f'{blk_name}_{self.time_step.item()}.pth')
        torch.save({'q':query,'k':key,'v':value},path)
        
    def record_qk(self,query,key):
        self.q = query
        self.k = key
        
class MyResBlkProcessor:
    def __init__(self, save_path, patch_size):
        self.conv_fea = None
        self.save_path = save_path
        self.patch_size = patch_size
        os.makedirs(self.save_path,exist_ok=True)
            
    def save_conv_fea(self,conv_fea,blk_name):
        path = os.path.join(self.save_path,f'{blk_name}_{self.time_step.item()}.pth')
        torch.save({'conv_fea':conv_fea},path)
        
    def record_conv_fea(self,conv_fea):
        self.conv_fea = conv_fea

def use_motion_module(self, use=True):
    for name, module in self.unet.named_modules():
        if hasattr(module,'use_motion_module'):
            setattr(module,'use_motion_module',use)
            
def prepare_self_attn_module_id(self):
    idx = 0
    for name, module in self.unet.named_modules():
        module_name = type(module).__name__
        if 'CrossAttention' in module_name and 'up' in name and 'attn1' in name:
            setattr(module,'module_name',f'self_attn_{idx}')
            idx += 1
    return
            
def prepare_resblock_id(self):
    idx = 0
    for name, module in self.unet.named_modules():
        module_name = type(module).__name__
        if 'ResnetBlock3D' in module_name and 'up' in name:
            setattr(module,'module_name',f'ResnetBlock_{idx}')
            idx += 1
    return
            
def prepare_proccessor(self,save_path,debias_patch_size,attn_idx=[],conv_idx=[]):
    for name, module in self.unet.named_modules():
        module_name = type(module).__name__
        if 'CrossAttention' in module_name and 'up' in name and 'attn1' in name\
            and (True if len(attn_idx)==0 else module.module_name.split('_')[-1] in attn_idx):
            module.set_processor(MySelfAttnProcessor(save_path))
        if 'ResnetBlock3D' in module_name and 'up' in name\
            and (True if len(conv_idx)==0 else module.module_name.split('_')[-1] in conv_idx):
            module.set_processor(MyResBlkProcessor(save_path,debias_patch_size))
            
def prepare_save_feature(self, time_step, attn_idx=[], conv_idx=[]):
    for name, module in self.unet.named_modules():
        module_name = type(module).__name__

        if 'CrossAttention' in module_name and 'up' in name and 'attn1' in name \
            and (True if len(attn_idx) == 0 else int(module.module_name.split('_')[-1]) in attn_idx):
            setattr(module.processor, 'time_step', time_step)
            setattr(module, 'save_feature', True)

        if 'ResnetBlock3D' in module_name and 'up' in name \
            and (True if len(conv_idx) == 0 else int(module.module_name.split('_')[-1]) in conv_idx):
            setattr(module.processor, 'time_step', time_step)
            setattr(module, 'save_feature', True)

def prepare_load_feature(self, time_step, attn_idx=[], conv_idx=[]):
    for name, module in self.unet.named_modules():
        module_name = type(module).__name__

        if 'CrossAttention' in module_name and 'up' in name and 'attn1' in name \
            and (True if len(attn_idx) == 0 else int(module.module_name.split('_')[-1]) in attn_idx):
            setattr(module, 'load_feature', True)
            setattr(module, 'time_step', time_step)

        if 'ResnetBlock3D' in module_name and 'up' in name \
            and (True if len(conv_idx) == 0 else int(module.module_name.split('_')[-1]) in conv_idx):
            setattr(module, 'load_feature', True)
            setattr(module, 'time_step', time_step)
            
def prepare_record_feature(self, attn_idx=[]):
    for name, module in self.unet.named_modules():
        module_name = type(module).__name__

        if 'CrossAttention' in module_name and 'up' in name and 'attn1' in name \
            and (True if len(attn_idx) == 0 else int(module.module_name.split('_')[-1]) in attn_idx):
            setattr(module, 'record_feature', True)

def load_feature_from_processor(self, attn_idx=[]):
    features = []

    for name, module in self.unet.named_modules():
        module_name = type(module).__name__

        if 'CrossAttention' in module_name and 'up' in name and 'attn1' in name \
            and (True if len(attn_idx) == 0 else int(module.module_name.split('_')[-1]) in attn_idx):
            h = w = int(math.sqrt(module.processor.q.shape[1]))
            query = rearrange(module.processor.q,'f (h w) c -> f h w c',h=h,w=w)
            features.append(query)
            
    return features

def prepare_resolution(self, H, W, L):
    for name, module in self.unet.named_modules():
        setattr(module, 'resolution_H', H)
        setattr(module, 'resolution_W', W)
        setattr(module, 'resolution_L', L)
            
            
@torch.no_grad()
def load_and_encode_image(self, image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((512, 512))
    image = transforms.functional.to_tensor(image).unsqueeze(0).to(self.vae.device)
    image = (image - 0.5) * 2
    latent = self.vae.encode(image).latent_dist
    latent = latent.sample() * 0.18215
    latent = rearrange(latent,'(b f) c h w -> b c f h w',b=1,f=1)
    return latent
            

@torch.no_grad()
def DDIM_inversion(
    self,
    prompt: Union[str, List[str]],
    video_length: Optional[int],
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_videos_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "tensor",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: Optional[int] = 1,
    inversion_end_step: Optional[float] = 1.0,
    inversion_timestep: Optional[int] = 201,
    load_attn_list: Optional[list] = [0, 1, 3, 4],
    load_res_list: Optional[list] = [3, 4, 6, 7],
    **kwargs
):
    # Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # Check inputs. Raise error if not correct
    self.check_inputs(prompt, height, width, callback_steps)

    # Define call parameters
    # batch_size = 1 if isinstance(prompt, str) else len(prompt)
    batch_size = 1
    if latents is not None:
        batch_size = latents.shape[0]
    if isinstance(prompt, list):
        batch_size = len(prompt)

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # Encode input prompt
    prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
    if negative_prompt is not None:
        negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
    text_embeddings = self._encode_prompt(
        prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Prepare timesteps
    self.inverse_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
    self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.inverse_scheduler.timesteps

    # Prepare latent variables
    num_channels_latents = self.unet.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        video_length,
        height,
        width,
        text_embeddings.dtype,
        device,
        generator,
        latents,
    )
    latents_dtype = latents.dtype

    # Prepare extra step kwargs.
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # Denoising loop
    latents = latents.clone().detach()
    end_step = int(len(timesteps) * inversion_end_step)
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps[:end_step]):
            if t==inversion_timestep:
                self.prepare_save_feature(t,load_attn_list,load_res_list)
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, 
                encoder_hidden_states=text_embeddings,
            ).sample.to(dtype=latents_dtype)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the next noisy sample x_t-1 -> x_t
            latents = self.inverse_scheduler.step(noise_pred, t, latents)
            latents, pred_x0 = latents.prev_sample, latents.pred_original_sample
            
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
            
            if t==inversion_timestep: break
                    
        return latents.detach().clone()
