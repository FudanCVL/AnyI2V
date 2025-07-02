import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
import torch.distributed as dist
import torchvision.transforms as transforms

from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora, load_diffusers_lora


MOTION_MODULES = [
    "mm_sd_v14.ckpt", 
    "mm_sd_v15.ckpt", 
    "mm_sd_v15_v2.ckpt", 
    "v3_sd15_mm.ckpt",
]

ADAPTERS = [
    # "mm_sd_v14.ckpt",
    # "mm_sd_v15.ckpt",
    # "mm_sd_v15_v2.ckpt",
    # "mm_sdxl_v10_beta.ckpt",
    "v2_lora_PanLeft.ckpt",
    "v2_lora_PanRight.ckpt",
    "v2_lora_RollingAnticlockwise.ckpt",
    "v2_lora_RollingClockwise.ckpt",
    "v2_lora_TiltDown.ckpt",
    "v2_lora_TiltUp.ckpt",
    "v2_lora_ZoomIn.ckpt",
    "v2_lora_ZoomOut.ckpt",
    "v3_sd15_adapter.ckpt",
    # "v3_sd15_mm.ckpt",
    "v3_sd15_sparsectrl_rgb.ckpt",
    "v3_sd15_sparsectrl_scribble.ckpt",
]

BACKUP_DREAMBOOTH_MODELS = [
    "realisticVisionV60B1_v51VAE.safetensors",
    "majicmixRealistic_v4.safetensors",
    "leosamsFilmgirlUltra_velvia20Lora.safetensors",
    "toonyou_beta3.safetensors",
    "majicmixRealistic_v5Preview.safetensors",
    "rcnzCartoon3d_v10.safetensors",
    "lyriel_v16.safetensors",
    "leosamsHelloworldXL_filmGrain20.safetensors",
    "TUSUN.safetensors",
]


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


import os
import torch
import torchvision
import numpy as np
import imageio
from PIL import Image
from einops import rearrange

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    
    if videos.shape[0] == 1:
        video_frame = videos[0]
        
        grid = torchvision.utils.make_grid(video_frame, nrow=n_rows)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            grid = (grid + 1.0) / 2.0
        grid = (grid * 255).numpy().astype(np.uint8)
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        jpg_path = path.replace(".gif", ".jpg")
        image = Image.fromarray(grid)
        image.save(jpg_path, 'JPEG')
        print(f"Saved a single frame as: {jpg_path}")
        return

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    imageio.mimsave(path, outputs, fps=fps)
    print(f"Saved multiple frames as: {path}")



def auto_download(local_path, is_dreambooth_lora=False):
    hf_repo = "guoyww/animatediff_t2i_backups" if is_dreambooth_lora else "guoyww/animatediff"
    folder, filename = os.path.split(local_path)

    if not os.path.exists(local_path):
        print(f"local file {local_path} does not exist. trying to download from {hf_repo}")

        if is_dreambooth_lora: assert filename in BACKUP_DREAMBOOTH_MODELS, f"{filename} dose not exist in {hf_repo}"
        else: assert filename in MOTION_MODULES + ADAPTERS, f"{filename} dose not exist in {hf_repo}"

        folder = "." if folder == "" else folder
        os.makedirs(folder, exist_ok=True)
        snapshot_download(repo_id=hf_repo, local_dir=folder, allow_patterns=[filename])


def load_weights(
    animation_pipeline,
    # motion module
    motion_module_path         = "",
    motion_module_lora_configs = [],
    # domain adapter
    adapter_lora_path          = "",
    adapter_lora_scale         = 1.0,
    # image layers
    dreambooth_model_path      = "",
    lora_model_path            = "",
    lora_alpha                 = 0.8,
):
    # motion module
    unet_state_dict = {}
    if motion_module_path != "":
        auto_download(motion_module_path, is_dreambooth_lora=False)

        print(f"load motion module from {motion_module_path}")
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
        # filter parameters
        for name, param in motion_module_state_dict.items():
            if not "motion_modules." in name: continue
            if "pos_encoder.pe" in name: continue
            unet_state_dict.update({name: param})
        unet_state_dict.pop("animatediff_config", "")
    
    missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
    assert len(unexpected) == 0
    del unet_state_dict

    # base model
    if dreambooth_model_path != "":
        auto_download(dreambooth_model_path, is_dreambooth_lora=True)

        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
            
        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, animation_pipeline.vae.config)
        animation_pipeline.vae.load_state_dict(converted_vae_checkpoint)
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, animation_pipeline.unet.config)
        animation_pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # 3. text_model
        animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        del dreambooth_state_dict
        
    # lora layers
    if lora_model_path != "":
        auto_download(lora_model_path, is_dreambooth_lora=True)

        print(f"load lora model from {lora_model_path}")
        assert lora_model_path.endswith(".safetensors")
        lora_state_dict = {}
        with safe_open(lora_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                lora_state_dict[key] = f.get_tensor(key)
                
        animation_pipeline = convert_lora(animation_pipeline, lora_state_dict, alpha=lora_alpha)
        del lora_state_dict

    # domain adapter lora
    if adapter_lora_path != "":
        auto_download(adapter_lora_path, is_dreambooth_lora=False)

        print(f"load domain lora from {adapter_lora_path}")
        domain_lora_state_dict = torch.load(adapter_lora_path, map_location="cpu")
        domain_lora_state_dict = domain_lora_state_dict["state_dict"] if "state_dict" in domain_lora_state_dict else domain_lora_state_dict
        domain_lora_state_dict.pop("animatediff_config", "")

        animation_pipeline = load_diffusers_lora(animation_pipeline, domain_lora_state_dict, alpha=adapter_lora_scale)

    # motion module lora
    for motion_module_lora_config in motion_module_lora_configs:
        path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]

        auto_download(path, is_dreambooth_lora=False)

        print(f"load motion LoRA from {path}")
        motion_lora_state_dict = torch.load(path, map_location="cpu")
        motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
        motion_lora_state_dict.pop("animatediff_config", "")

        animation_pipeline = load_diffusers_lora(animation_pipeline, motion_lora_state_dict, alpha)

    return animation_pipeline

def read_video(file_name):
    vid = imageio.get_reader(file_name, 'ffmpeg') # 'RGB'
    num_frames = vid.count_frames()
    video = []
    for i in range(num_frames):
        img = vid.get_data(i)
        video.append(transforms.ToTensor()(img))  # 使用 transforms.ToTensor() 将图像转换为张量
    video = torch.stack(video, dim=0)
    return video

def read_gif(file_name):
    images = imageio.mimread(file_name)
    images = torch.from_numpy(np.array(images)) / 255.0
    images = images.permute(0, 3, 1, 2)
    return images[:,:3,:,:]


def read_file(file_name):
    if file_name.endswith(".gif"):
        video_tensor = read_gif(file_name)
    elif file_name.endswith(".mp4"):
        video_tensor = read_video(file_name)

    return video_tensor

def resize_video(video_tensor):
    num_original_frames = video_tensor.size(0)
    if num_original_frames > 16:
        num_frames = 16
        interval = num_original_frames // num_frames
        selected_frames_indices = torch.arange(0, num_original_frames, interval)[:num_frames]
        video_tensor = video_tensor[selected_frames_indices, :, :, :]
    if video_tensor.size(2) != 512 or video_tensor.size(3) != 512:
        video_tensor = torch.nn.functional.interpolate(video_tensor, size=(512, 512), mode='bilinear',
                                                       align_corners=False)
    return video_tensor


def encode_video(vae, sub_video):
    video_length = sub_video.shape[1]
    sub_video = (sub_video - 0.5) * 2
    sub_video = rearrange(sub_video, "b f c h w -> (b f) c h w")
    sub_latent = vae.encode(sub_video).latent_dist
    sub_latent = sub_latent.sample()
    sub_latent = rearrange(sub_latent, "(b f) c h w -> b c f h w", f = video_length)
    sub_latent = sub_latent * 0.18215
    return sub_latent