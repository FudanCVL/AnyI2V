import argparse
import datetime
import inspect
import json
import os
import math
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.fft
import torchvision.transforms as transforms
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.my_unet import MyUNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_anyi2v import AnyI2V
from animatediff.utils.util import (load_weights, save_videos_grid)
from utils.customized_func import (DDIM_inversion, load_and_encode_image, 
                            load_feature_from_processor, prepare_load_feature,
                            prepare_proccessor, prepare_record_feature,
                            prepare_resblock_id, prepare_save_feature,
                            prepare_self_attn_module_id, use_motion_module,
                            prepare_resolution)


VIDEO_LENGTH = 16

def parse_args():
    parser = argparse.ArgumentParser(description="AnimateDiff Video Generation with Trajectories")
    parser.add_argument("--pretrained-model-path", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to the pretrained Stable Diffusion v1.5 model.")
    parser.add_argument("--inference-config", type=str, default="configs/inference/inference-v1.yaml", help="Path to the inference configuration file.")
    parser.add_argument("--hyper-params-config", type=str, default="configs/prompts/hyper_params.yaml", help="Path to the inference configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main generation configuration file.")
    parser.add_argument("--L", type=int, default=16, help="Number of frames in the generated video.")
    parser.add_argument("--W", type=int, default=512, help="Width of the generated video.")
    parser.add_argument("--H", type=int, default=512, help="Height of the generated video.")
    parser.add_argument("--without-xformers", action="store_true", help="Disable xformers memory efficient attention.")
    return parser.parse_args()

def setup_env(config_path: str) -> Path:
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = Path(f"samples/{Path(config_path).stem}-{time_str}")
    savedir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to: {savedir}")
    return savedir

def load_trajectory_data(path: str) -> tuple[list, list]:
    with open(path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    
    bounding_box = np.array(data['bounding_box'], dtype=np.float32)
    target_box = np.array(data['target_box'], dtype=np.float32)
    center_traj = np.array(data['center_traj'], dtype=np.float32)
    query_point_orig = data['query_point']
    
    trajectory_points = []
    query_points = []

    for j, trajectory in enumerate(center_traj):
        w_start, h_start = bounding_box[j][2] - bounding_box[j][0], bounding_box[j][3] - bounding_box[j][1]
        w_end, h_end = target_box[j][2] - target_box[j][0], target_box[j][3] - target_box[j][1]
        
        w_offset = np.linspace(0, w_end - w_start, num=VIDEO_LENGTH) / 2
        h_offset = np.linspace(0, h_end - h_start, num=VIDEO_LENGTH) / 2
        
        box_traj = []
        for i in range(VIDEO_LENGTH):
            dx, dy = trajectory[i] - trajectory[0]
            
            top = bounding_box[j][1] + dy - h_offset[i]
            left = bounding_box[j][0] + dx - w_offset[i]
            bottom = bounding_box[j][3] + dy + h_offset[i]
            right = bounding_box[j][2] + dx + w_offset[i]
            box_traj.append(np.array([top, left, bottom, right], dtype=np.float32))
            
        trajectory_points.append(box_traj)

        query = []
        for q_point in query_point_orig[j]:
            x, y = q_point[0], q_point[1]
            b_left, b_top, b_right, b_bottom = bounding_box[j]

            if not (b_left < x < b_right and b_top < y < b_bottom):
                center_y = (b_bottom - b_top) / 2
                center_x = (b_right - b_left) / 2
                query.append([center_y, center_x])
            else:
                query.append([y - b_top, x - b_left])
        query_points.append(query)
        
    return trajectory_points, query_points

def load_base_models(pretrained_model_path: str) -> dict:
    print("Loading base models...")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").cuda().half()
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").cuda().half()
    return {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae}

def enhance_pipeline(pipeline: AnyI2V, debias_patch_size: int, H: int, W: int, L: int):
    methods_to_attach = {
        "use_motion_module": use_motion_module,
        "prepare_proccessor": prepare_proccessor,
        "prepare_self_attn_module_id": prepare_self_attn_module_id,
        "prepare_resblock_id": prepare_resblock_id,
        "prepare_save_feature": prepare_save_feature,
        "prepare_record_feature": prepare_record_feature,
        "prepare_load_feature": prepare_load_feature,
        "load_feature_from_processor": load_feature_from_processor,
        "DDIM_inversion": DDIM_inversion,
        "load_and_encode_image": load_and_encode_image,
        "prepare_resolution": prepare_resolution,
    }
    for name, method in methods_to_attach.items():
        setattr(pipeline, name, method.__get__(pipeline))
    
    pipeline.prepare_self_attn_module_id()
    pipeline.prepare_resblock_id()
    pipeline.prepare_proccessor("feature",debias_patch_size)
    pipeline.prepare_resolution(H, W, L)


@torch.no_grad()
def main():
    args = parse_args()
    savedir = setup_env(args.config)
    config = OmegaConf.load(args.config)
    hyper_param_config = OmegaConf.load(args.hyper_params_config)
    merged_list = [OmegaConf.merge(c, h) for c, h in zip(config, hyper_param_config)]
    config = merged_list
    
    base_models = load_base_models(args.pretrained_model_path)
    
    sample_idx_counter = 0

    for model_idx, model_config in enumerate(config):
        print(f"\nProcessing model configuration #{model_idx + 1}...")

        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        global VIDEO_LENGTH
        VIDEO_LENGTH = model_config.L
        
        lr = model_config.lr
        opt_iters = model_config.opt_iters
        PCA_dim = model_config.PCA_dim
        opt_step = model_config.opt_step
        compress_factor = model_config.compress_factor
        debias_patch_size = model_config.debias_patch_size
        load_feature_step = model_config.load_feature_step
        inversion_timestep = model_config.inversion_timestep
        opt_attn_list = model_config.opt_attn_list
        load_attn_list = model_config.load_attn_list
        load_res_list = model_config.load_res_list

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = MyUNet3DConditionModel.from_pretrained_2d(
            args.pretrained_model_path, 
            subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
        ).cuda()
        unet.requires_grad_(False)

        if is_xformers_available() and not args.without_xformers:
            print("Enabling xformers memory efficient attention.")
            unet.enable_xformers_memory_efficient_attention()
        
        pipeline = AnyI2V(
            vae=base_models["vae"],
            text_encoder=base_models["text_encoder"],
            tokenizer=base_models["tokenizer"],
            unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            motion_module_path=model_config.get("motion_module", ""),
            motion_module_lora_configs=model_config.get("motion_module_lora_configs", []),
            dreambooth_model_path=model_config.get("dreambooth_path", ""),
            lora_model_path=model_config.get("lora_model_path", ""),
            lora_alpha=model_config.get("lora_alpha", 0.8),
        ).to("cuda")
        
        enhance_pipeline(pipeline,debias_patch_size,model_config.H,model_config.W,model_config.L)

        prompts = model_config.prompt
        n_prompts = list(model_config.n_prompt) * len(prompts)
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        if len(random_seeds) == 1:
            random_seeds *= len(prompts)

        trajectory_paths = list(model_config.get("trajectory", []))
        trajectories = [load_trajectory_data(p) if p else (None, None) for p in trajectory_paths]
        
        config[model_idx].random_seed = []
        
        for prompt_idx, (prompt, n_prompt, (traj, q_points), seed, img_path) in enumerate(
            zip(prompts, n_prompts, trajectories, random_seeds, list(model_config.image_path))
        ):
            if seed != -1:
                torch.manual_seed(seed)
            else:
                torch.seed()
            current_seed = torch.initial_seed()
            config[model_idx].random_seed.append(current_seed)
            
            print(f"\n--- Generating for prompt: '{prompt}' (Seed: {current_seed}) ---")

            with torch.autocast("cuda", dtype=torch.float16):
                print("Step 1: Performing DDIM Inversion...")
                source_latents = pipeline.load_and_encode_image(img_path)
                pipeline.use_motion_module(False)

                inverted_latents = pipeline.DDIM_inversion(
                    prompt=prompt,
                    negative_prompt='',
                    num_inference_steps=999,
                    guidance_scale=1.0,
                    width=model_config.W,
                    height=model_config.H,
                    video_length=1,
                    latents=source_latents,
                    inversion_end_step=inversion_timestep/100+0.01,
                    inference_step=model_config.inference_steps,
                    inversion_timestep=inversion_timestep,
                    load_attn_list=load_attn_list,
                    load_res_list=load_res_list,
                )

                print("Step 2: Generating video...")
                initial_noise = torch.randn((1, 4, VIDEO_LENGTH, model_config.H//pipeline.vae_scale_factor, model_config.W//pipeline.vae_scale_factor), device="cuda")
                pipeline.use_motion_module(True)

                output = pipeline(
                    prompt,
                    negative_prompt=n_prompt,
                    num_inference_steps=model_config.inference_steps,
                    guidance_scale=model_config.guidance_scale,
                    width=model_config.W,
                    height=model_config.H,
                    video_length=model_config.L,
                    latents=initial_noise,
                    trajectory=traj,
                    query_points=q_points,
                    lr=lr,
                    PCA_dim=PCA_dim,
                    inversion_timestep=inversion_timestep,
                    opt_step=opt_step,
                    opt_iters=opt_iters,
                    compress_factor=compress_factor,
                    load_feature_step=load_feature_step,
                    opt_attn_list=opt_attn_list,
                    load_attn_list=load_attn_list,
                    load_res_list=load_res_list
                )

                sample = output.videos

                prompt_str_safe = "".join(filter(lambda x: x.isalnum() or x == "_", prompt.replace(" ", "_")))[:100]
                save_path = savedir / "samples" / f"{sample_idx_counter}-{prompt_str_safe}.gif"
                save_path.parent.mkdir(exist_ok=True)
                save_videos_grid(sample, str(save_path))
                print(f"Saved individual sample to: {save_path}")
            
            sample_idx_counter += 1

    OmegaConf.save(config, savedir / "config_with_seeds.yaml")
    print("Final configuration saved.")


if __name__ == "__main__":
    main()
