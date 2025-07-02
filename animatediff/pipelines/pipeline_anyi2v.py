# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py
import os
import sys
import cv2
import inspect
from sklearn.cluster import KMeans
from scipy.ndimage import label
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import imageio

from torch_pca import PCA
from kmeans_pytorch import kmeans

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


from .pipeline_animation import AnimationPipeline

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..models.sparse_controlnet import SparseControlNetModel
import pdb

from ..utils.util import save_videos_grid

logger = logging.get_logger(__name__)

import contextlib

@contextlib.contextmanager
def suppress_all_output():
    with open(os.devnull, 'w') as devnull:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        sys.stdout = devnull
        sys.stderr = devnull
        
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    latents: Union[torch.Tensor, np.ndarray]


class AnyI2V(AnimationPipeline):
    
    @torch.enable_grad()
    def optimize_latent(self, latents, t, text_embeddings, trajectory, query_points, inversion_timestep, opt_attn_list, load_attn_list, load_res_list, lr, pca_dim, compress_factor):
        latents = latents.clone().detach().requires_grad_(True)
        scaler = torch.cuda.amp.GradScaler()
        injected_features = [None] * len(trajectory)

        self.prepare_load_feature(inversion_timestep, load_attn_list, load_res_list)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            noise_pred = self.unet(
                latents, t,
                encoder_hidden_states=text_embeddings
            )

        recorded_features = self.load_feature_from_processor(attn_idx=opt_attn_list)
        features = []

        h, w = latents.shape[-2:]
        compress_factor = compress_factor
        scale_factor = self.vae_scale_factor//compress_factor
        h, w = scale_factor*h, scale_factor*w
        for feature in recorded_features:
            f, h_, w_, c = feature.shape
            feature = rearrange(feature, 'f h w c -> (f h w) c', h=h_, w=w_)

            with torch.autocast("cuda", dtype=torch.float32):
                pca = PCA(n_components=pca_dim)
                with torch.no_grad():
                    pca.fit(feature)
                feature = pca.transform(feature)

            feature = rearrange(feature, '(f h w) c -> f h w c', f=f, h=h_, w=w_)
            feature = feature.permute((0, 3, 1, 2))
            feature = F.interpolate(feature, size=(h, w), mode="bicubic")
            feature = feature.permute((0, 2, 3, 1))
            features.append(feature)

        stacked_feature = torch.cat(features, dim=-1)
        recorded_features = []

        frames = features[0].shape[0]

        loss, count = 0, 0
        first_frame_mask = None

        for j in range(frames):
            for traj_idx in range(len(trajectory)):
                current_bbox = (trajectory[traj_idx][j] // compress_factor).astype(np.int32)
                
                box_x1, box_y1 = current_bbox[0], current_bbox[1]
                box_x2 = max(box_x1 + 1, current_bbox[2])
                box_y2 = max(box_y1 + 1, current_bbox[3])

                clipped_x1 = max(box_x1, 0)
                clipped_y1 = max(box_y1, 0)
                clipped_x2 = min(box_x2, stacked_feature.shape[1])
                clipped_y2 = min(box_y2, stacked_feature.shape[2])

                src_slice_x1 = clipped_x1 - box_x1
                src_slice_y1 = clipped_y1 - box_y1
                src_slice_x2 = clipped_x2 - box_x1
                src_slice_y2 = clipped_y2 - box_y1

                if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
                    if j == 0:
                        exit(1)
                    continue

                if j == 0:
                    injected_features[traj_idx] = stacked_feature[0, box_x1:box_x2, box_y1:box_y2].detach().requires_grad_(False)

                if j >= 0:
                    injected_feature = injected_features[traj_idx].unsqueeze(0)
                    reshape_size = (box_x2 - box_x1, box_y2 - box_y1)
                    injected_feature = F.interpolate(injected_feature.permute((0, 3, 1, 2)), size=reshape_size, mode="bicubic")
                    injected_feature = injected_feature.permute((0, 2, 3, 1))[0]
                    injected_feature = injected_feature[src_slice_x1:src_slice_x2, src_slice_y1:src_slice_y2]
                    frame_feature = stacked_feature[j, clipped_x1:clipped_x2, clipped_y1:clipped_y2]

                    H, W, C = injected_feature.shape
                    chunk = len(opt_attn_list)
                    similarity, queried_similarities = [], []

                    for i in range(chunk):
                        chunk_size = C // chunk
                        start_idx = i * chunk_size
                        end_idx = (i + 1) * chunk_size
                        similarity.append(
                            (F.normalize(injected_feature[..., start_idx:end_idx], p=2, dim=-1) * F.normalize(frame_feature[..., start_idx:end_idx], p=2, dim=-1)).sum(dim=-1)
                        )

                        query = torch.tensor(np.array(query_points[traj_idx]), device=injected_feature.device) // compress_factor
                        query[:, 0], query[:, 1] = query[:, 0].clip(0, H - 1), query[:, 1].clip(0, W - 1)
                        query = query.to(torch.int)
                        query_token = F.normalize(injected_feature[query[:, 0], query[:, 1], start_idx:end_idx], p=2, dim=-1)[:, None, None]
                        queried_similarity = (query_token * F.normalize(frame_feature[..., start_idx:end_idx], p=2, dim=-1)[None]).sum(dim=-1)
                        queried_similarity = queried_similarity.max(dim=0).values
                        queried_similarities.append(queried_similarity)

                    queried_similarity = torch.stack(queried_similarities, dim=-1)

                    pixels = queried_similarity.detach().reshape(-1, 1)

                    with suppress_all_output():
                        cluster_ids_x, cluster_centers = kmeans(
                            X=pixels,
                            num_clusters=2,
                            distance='euclidean',
                            device=torch.device('cuda')
                        )

                    foreground_label = torch.argmax(cluster_centers)
                    mask = (cluster_ids_x == foreground_label).float().reshape(queried_similarity.shape)
                    mask = mask.cuda()

                    if first_frame_mask is None:
                        first_frame_mask = mask.clone()
                        continue

                    if mask.shape != first_frame_mask.shape:
                        first_frame_mask = first_frame_mask.permute(2, 0, 1).unsqueeze(0)
                        first_frame_mask = F.interpolate(first_frame_mask, mask.shape[:2], mode='bilinear')
                        first_frame_mask = first_frame_mask[0].permute(1, 2, 0)

                    for i in range(chunk):
                        chunk_size = C // chunk
                        start_idx = i * chunk_size
                        end_idx = (i + 1) * chunk_size
                        mse_loss = F.mse_loss(
                            injected_feature[..., start_idx:end_idx] * mask[..., i:i + 1] * first_frame_mask[..., i:i + 1],
                            frame_feature[..., start_idx:end_idx] * mask[..., i:i + 1] * first_frame_mask[..., i:i + 1],
                            reduction="none"
                        ).mean(dim=-1)

                        loss += mse_loss.sum()

                    count += (mask*first_frame_mask).sum()

        loss = loss / max(1e-8, count)

        optimizer = torch.optim.AdamW([latents], lr=lr)

        optimizer.zero_grad()
        if count > 0:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        del loss, stacked_feature, recorded_features
        torch.cuda.empty_cache()

        latents = latents.detach().requires_grad_(False)

        return latents


    @torch.no_grad()
    def __call__(
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
        trajectory: Optional[list] = None,
        query_points: Optional[list] = None,
        lr: Optional[float] = 0.01,
        PCA_dim: Optional[int] = 64,
        inversion_timestep: Optional[int] = 201,
        opt_step: Optional[int] = 5,
        opt_iters: Optional[int] = 5,
        compress_factor: Optional[int] = 4,
        load_feature_step: Optional[int] = 3,
        opt_attn_list: Optional[list] = [1, 3],
        load_attn_list: Optional[list] = [0, 1, 3, 4],
        load_res_list: Optional[list] = [3, 4, 6, 7],
        **kwargs,
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
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

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
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if trajectory is not None and i<=opt_step:
                    attn_idx = load_attn_list
                    self.prepare_record_feature(attn_idx)
                    last_layer = min((attn_idx[-1]+3)//3,4)
                    if last_layer==4: last_layer==None
                    setattr(self.unet,'last_recorded_layer',last_layer)
                    for _ in range(opt_iters):
                        latents = self.optimize_latent(latents,t,text_embeddings[1:],trajectory,query_points,inversion_timestep,opt_attn_list,
                                                    load_attn_list,load_res_list,lr,PCA_dim,compress_factor)
                    setattr(self.unet,'last_recorded_layer',None)
                
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if i<=load_feature_step:
                    self.prepare_load_feature(inversion_timestep, load_attn_list, load_res_list)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, 
                        encoder_hidden_states = text_embeddings
                    ).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents, pred_original = latents.prev_sample, latents.pred_original_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video, latents=latents)
