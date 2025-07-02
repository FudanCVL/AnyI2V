import gradio as gr
import torch
import numpy as np
import os
import datetime
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from omegaconf import OmegaConf
from pathlib import Path
import json

from animatediff.models.my_unet import MyUNet3DConditionModel
from animatediff.pipelines.pipeline_anyi2v import AnyI2V
from animatediff.utils.util import (load_weights, save_videos_grid)
from utils.customized_func import (DDIM_inversion, load_and_encode_image, 
                            load_feature_from_processor, prepare_load_feature,
                            prepare_proccessor, prepare_record_feature,
                            prepare_resblock_id, prepare_save_feature,
                            prepare_self_attn_module_id, use_motion_module,
                            prepare_resolution)

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================
DREAMBOOTH_PATH = "models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors"
PRETRAINED_MODEL_PATH = "/home/zyli/pretrained/models--runwayml--stable-diffusion-v1-5/"
MOTION_MODULE_PATH = "models/Motion_Module/v3_sd15_mm.ckpt"
INFERENCE_CONFIG_PATH = "configs/inference/inference-v3.yaml"
HYPER_PARAMS_CONFIG_PATH = "configs/prompts/hyper_params.yaml"
OUTPUT_DIR = "outputs"

hyper_params_config = OmegaConf.load(HYPER_PARAMS_CONFIG_PATH)[0]

VIDEO_LENGTH = 16
WIDTH = 512
HEIGHT = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. MODEL LOADING
# =============================================================================
print("Loading base models...")
tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="text_encoder").cuda().half()
vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae").cuda().half()

print("Loading UNet and Motion Module...")
inference_config = OmegaConf.load(INFERENCE_CONFIG_PATH)
unet = MyUNet3DConditionModel.from_pretrained_2d(
    PRETRAINED_MODEL_PATH, 
    subfolder="unet", 
    unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
).cuda()

if is_xformers_available():
    print("Enabling xformers for memory efficiency.")
    unet.enable_xformers_memory_efficient_attention()

pipeline = AnyI2V(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
    scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
).to("cuda")

pipeline = load_weights(
    pipeline,
    motion_module_path=MOTION_MODULE_PATH,
    motion_module_lora_configs=[],
    dreambooth_model_path=DREAMBOOTH_PATH,
    lora_model_path="",
    lora_alpha=0.8,
).to("cuda")

methods_to_attach = {
    "use_motion_module": use_motion_module, "prepare_proccessor": prepare_proccessor,
    "prepare_self_attn_module_id": prepare_self_attn_module_id, "prepare_resblock_id": prepare_resblock_id,
    "prepare_save_feature": prepare_save_feature, "prepare_record_feature": prepare_record_feature,
    "prepare_load_feature": prepare_load_feature, "load_feature_from_processor": load_feature_from_processor,
    "DDIM_inversion": DDIM_inversion, "load_and_encode_image": load_and_encode_image,
    "prepare_resolution": prepare_resolution,
}
for name, method in methods_to_attach.items():
    setattr(pipeline, name, method.__get__(pipeline))

pipeline.prepare_self_attn_module_id()
pipeline.prepare_resblock_id()
pipeline.prepare_resolution(HEIGHT, WIDTH, VIDEO_LENGTH)

print("Models loaded successfully. Gradio is starting...")


# =============================================================================
# 3. ANNOTATION & UI LOGIC
# =============================================================================
initial_state = { "image_path": None, "clicked_points": [], "trajectory_points": [], "is_trajectory_mode": False, "bounding_boxes": [], "target_boxes": [], "query_points_list": [], "center_traj_list": [] }

def interpolate_trajectory(points, n_points=VIDEO_LENGTH):
    if not points: return []
    if len(points) == 1: return [points[0]] * n_points
    x, y = [p[0] for p in points], [p[1] for p in points]
    t = np.linspace(0, 1, len(points))
    try:
        fx, fy = PchipInterpolator(t, x), PchipInterpolator(t, y)
        new_t = np.linspace(0, 1, n_points)
        return list(zip(fx(new_t), fy(new_t)))
    except Exception: return [points[-1]] * n_points

def draw_all_annotations(base_image, state):
    if base_image is None: return None
    img = base_image.copy(); draw = ImageDraw.Draw(img)

    for i in range(len(state["bounding_boxes"])):
        if len(state["bounding_boxes"][i]) == 2 and len(state["bounding_boxes"][i][0]) == 2:
            x1, y1 = state["bounding_boxes"][i][0]; x2, y2 = state["bounding_boxes"][i][1]
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=2)
        
        for point in state["query_points_list"][i]: x, y = point; draw.ellipse([x-5, y-5, x+5, y+5], fill="#FFFF00")
        group_traj = state["center_traj_list"][i]
        if group_traj:
            for j in range(len(group_traj)-1): draw.line([tuple(group_traj[j]), tuple(group_traj[j+1])], fill="#0000FF", width=2)
            for point in group_traj: x, y = point; draw.ellipse([x-5, y-5, x+5, y+5], outline="#FF0000", width=2)

    if len(state["clicked_points"]) >= 2:
        x1, y1 = state["clicked_points"][0]; x2, y2 = state["clicked_points"][1]
        draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=2)
    for point in state["clicked_points"][2:]: x, y = point; draw.ellipse([x-5, y-5, x+5, y+5], fill="#FFFF00")
    if state["trajectory_points"]:
        current_traj = interpolate_trajectory(state["trajectory_points"])
        if len(current_traj) >= 2:
            for j in range(len(current_traj)-1): draw.line([tuple(current_traj[j]), tuple(current_traj[j+1])], fill="#0000FF", width=2)
        for point in current_traj: x, y = point; draw.ellipse([x-5, y-5, x+5, y+5], outline="#FF0000", width=2)
    return img

def load_image(img_path, state):
    new_state = initial_state.copy()
    new_state["image_path"] = img_path
    img = Image.open(img_path)
    return new_state, img, "Bounding Box: Not Drawn", "Clicked Points: None", "Trajectory Mode: OFF"

def process_example(image_path, json_path):
    """
    This function is triggered when an example is clicked.
    It reads the image and json paths, loads the data, shifts the trajectory
    to the bounding box center, updates the state, and returns all the necessary
    values to update the UI.
    """
    if not image_path or not json_path: # Handles initial empty state
        return initial_state, None, "Ready", "N/A", "N/A"

    if not os.path.exists(image_path) or not os.path.exists(json_path):
        raise gr.Error(f"Example files not found. Check paths: {image_path}, {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)
    image = Image.open(image_path)
    
    new_state = initial_state.copy()
    new_state["image_path"] = image_path
    
    raw_bounding_boxes = data.get("bounding_box", [])
    raw_center_trajs = data.get("center_traj", [])
    
    modified_center_trajs = []

    for i, traj in enumerate(raw_center_trajs):
        if i < len(raw_bounding_boxes) and traj and len(traj) > 0:
            bbox = raw_bounding_boxes[i]
            
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            
            first_point = traj[0]
            
            delta_x = bbox_center_x - first_point[0]
            delta_y = bbox_center_y - first_point[1]
            
            shifted_traj = [[point[0] + delta_x, point[1] + delta_y] for point in traj]
            
            modified_center_trajs.append(shifted_traj)
        else:
            modified_center_trajs.append(traj)
            
    def convert_box_format(box_list):
        return [[[box[0], box[1]], [box[2], box[3]]] for box in box_list]

    new_state["bounding_boxes"] = convert_box_format(raw_bounding_boxes)
    new_state["target_boxes"] = convert_box_format(data.get("target_box", []))
    new_state["query_points_list"] = data.get("query_point", [])
    new_state["center_traj_list"] = modified_center_trajs
    
    annotated_image = draw_all_annotations(image, new_state)
    
    status_msg = f"Successfully loaded and centered example: {os.path.basename(image_path)}"
    bbox_msg = "Bounding Box: Loaded from file."
    points_msg = "Query Points: Loaded from file."
    
    return new_state, annotated_image, status_msg, bbox_msg, points_msg


def handle_click(state, evt: gr.SelectData):
    if state["image_path"] is None: return state, None, "Please upload an image first!", ""
    if state["is_trajectory_mode"]: state["trajectory_points"].append([evt.index[0], evt.index[1]])
    else: state["clicked_points"].append([evt.index[0], evt.index[1]])
    base_img = Image.open(state["image_path"])
    annotated_img = draw_all_annotations(base_img, state)
    bbox_info = f"Bounding Box: {state['clicked_points'][:2]}" if len(state['clicked_points']) >= 2 else "Bounding Box: Not Drawn"
    points_info = f"Clicked Points: {state['clicked_points'][2:]}" if len(state['clicked_points']) > 2 else "Clicked Points: None"
    return state, annotated_img, bbox_info, points_info

def toggle_trajectory_mode(state):
    if state["image_path"] is None: return state, None, "Please upload an image first!"
    state["is_trajectory_mode"] = not state["is_trajectory_mode"]
    base_img = Image.open(state["image_path"])
    annotated_img = draw_all_annotations(base_img, state)
    mode_text = "Trajectory Mode: ON" if state["is_trajectory_mode"] else "Trajectory Mode: OFF"
    return state, annotated_img, mode_text

def add_data_group(state):
    if state["image_path"] is None or len(state["clicked_points"]) < 2: return state, None, "Error: Please draw a Bounding Box first (at least 2 points)."
    current_traj = [[round(x, 2), round(y, 2)] for x, y in interpolate_trajectory(state["trajectory_points"])]
    state["bounding_boxes"].append(state["clicked_points"][:2]); state["target_boxes"].append(state["clicked_points"][:2])
    state["query_points_list"].append(state["clicked_points"][2:]); state["center_traj_list"].append(current_traj)
    state["clicked_points"], state["trajectory_points"], state["is_trajectory_mode"] = [], [], False
    base_img = Image.open(state["image_path"])
    annotated_img = draw_all_annotations(base_img, state)
    return state, annotated_img, "Status: Current group data added. Continue annotating or generate the video."

def clear_all(state):
    image_path = state.get("image_path")
    new_state = initial_state.copy()
    new_state["image_path"] = image_path
    new_state["clicked_points"] = []
    new_state["is_trajectory_mode"] = False
    new_state["trajectory_points"] = []
    new_state["bounding_boxes"] = []
    new_state["query_points_list"] = []
    new_state["center_traj_list"] = []
    
    display_image = None
    if image_path:
        try:
            display_image = Image.open(image_path)
        except (FileNotFoundError, TypeError):
            print(f"Warning: Could not find or open the image at {image_path}. Clearing image display.")
            new_state["image_path"] = None

    return (
        new_state,
        display_image,
        None,
        "Bounding Box: Not Drawn",
        "Clicked Points: None",
        "Annotation data has been cleared."
    )

# =============================================================================
# 4. VIDEO GENERATION LOGIC (未修改)
# =============================================================================
def process_annotation_data(state):
    if not state["bounding_boxes"]: return None, None
    bounding_box_list, target_box_list, center_traj_list, query_point_list = state["bounding_boxes"], state["target_boxes"], state["center_traj_list"], state["query_points_list"]
    
    bounding_boxes_np = np.array([[p[0][0], p[0][1], p[1][0], p[1][1]] for p in bounding_box_list], dtype=np.float32)
    target_boxes_np = np.array([[p[0][0], p[0][1], p[1][0], p[1][1]] for p in target_box_list], dtype=np.float32)
    
    all_box_traj, all_query_points_processed = [], []
    for j in range(len(center_traj_list)):
        center_traj = np.array(center_traj_list[j], dtype=np.float32)
        if center_traj.shape[0] == 0:
            center_of_box = np.array([(bounding_boxes_np[j][0] + bounding_boxes_np[j][2]) / 2, (bounding_boxes_np[j][1] + bounding_boxes_np[j][3]) / 2])
            center_traj = np.tile(center_of_box, (VIDEO_LENGTH, 1))
        w_start, h_start = bounding_boxes_np[j][2] - bounding_boxes_np[j][0], bounding_boxes_np[j][3] - bounding_boxes_np[j][1]
        w_end, h_end = target_boxes_np[j][2] - target_boxes_np[j][0], target_boxes_np[j][3] - target_boxes_np[j][1]
        w_offset, h_offset = np.linspace(0, w_end - w_start, num=VIDEO_LENGTH) / 2, np.linspace(0, h_end - h_start, num=VIDEO_LENGTH) / 2
        box_traj = []
        for i in range(VIDEO_LENGTH):
            dx, dy = center_traj[i] - center_traj[0]
            top, left = bounding_boxes_np[j][1] + dy - h_offset[i], bounding_boxes_np[j][0] + dx - w_offset[i]
            bottom, right = bounding_boxes_np[j][3] + dy + h_offset[i], bounding_boxes_np[j][2] + dx + w_offset[i]
            box_traj.append(np.array([top, left, bottom, right], dtype=np.float32))
        all_box_traj.append(box_traj)
        query = []
        for q_point in query_point_list[j]:
            x, y = q_point[0], q_point[1]
            b_left, b_top, b_right, b_bottom = bounding_boxes_np[j]
            if not (b_left < x < b_right and b_top < y < b_bottom):
                center_y, center_x = (b_bottom - b_top) / 2, (b_right - b_left) / 2
                query.append([center_y, center_x])
            else:
                query.append([y - b_top, x - b_left])
        all_query_points_processed.append(query)
    return all_box_traj, all_query_points_processed

@torch.no_grad()
def run_generation(prompt, n_prompt, state,
                inference_steps, guidance_scale, lr, opt_iters, pca_dim,
                opt_step, compress_factor, load_feature_step, inversion_timestep,
                debias_patch_size, opt_attn_list_str, load_attn_list_str, load_res_list_str,
                progress=gr.Progress()):
    if not state.get("image_path"): raise gr.Error("Please upload an image first!")
    if not state.get("bounding_boxes"): raise gr.Error("Please add at least one data group (including a Bounding Box)!")
    
    image_path = state["image_path"]

    try:
        opt_attn_list = [int(x.strip()) for x in opt_attn_list_str.split(',') if x.strip()]
        load_attn_list = [int(x.strip()) for x in load_attn_list_str.split(',') if x.strip()]
        load_res_list = [int(x.strip()) for x in load_res_list_str.split(',') if x.strip()]
    except ValueError:
        raise gr.Error("List parameter format error. Please enter comma-separated numbers, e.g., '0, 1, 3, 4'.")

    progress(0, desc="Processing annotation data...")
    trajectories, query_points = process_annotation_data(state)
    if trajectories is None: raise gr.Error("Annotation data processing failed. Please check the annotations.")
    
    torch.manual_seed(4162228652802886667)
    pipeline.prepare_proccessor("feature", debias_patch_size)

    with torch.autocast("cuda", dtype=torch.float16):
        progress(0.1, desc="Step 1/2: Performing DDIM Inversion...")
        source_latents = pipeline.load_and_encode_image(image_path)
        pipeline.use_motion_module(False)

        num_inversion_steps = 999
        def ddim_inversion_callback(step, timestep, latents):
            progress(0.1 + 0.4 * (step / inversion_timestep), desc=f"DDIM Inversion: Step {step}/{inversion_timestep}")
        
        _ = pipeline.DDIM_inversion(
            prompt=prompt, negative_prompt='', num_inference_steps=num_inversion_steps,
            guidance_scale=1.0, width=WIDTH, height=HEIGHT, video_length=1,
            latents=source_latents,
            inversion_end_step=inversion_timestep/100+0.01,
            inference_step=inference_steps,
            inversion_timestep=inversion_timestep,
            load_attn_list=load_attn_list, load_res_list=load_res_list,
            callback=ddim_inversion_callback
        )
        
        progress(0.5, desc="Step 2/2: Generating video...")
        initial_noise = torch.randn((1, 4, VIDEO_LENGTH, HEIGHT//pipeline.vae_scale_factor, WIDTH//pipeline.vae_scale_factor), device="cuda")
        pipeline.use_motion_module(True)

        def generation_callback(step, timestep, latents):
            progress(0.5 + 0.5 * (step / inference_steps), desc=f"Generating video: Step {step}/{inference_steps}")

        output = pipeline(
            prompt, negative_prompt=n_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            width=WIDTH, height=HEIGHT, video_length=VIDEO_LENGTH,
            latents=initial_noise, trajectory=trajectories, query_points=query_points,
            lr=lr, PCA_dim=pca_dim,
            inversion_timestep=inversion_timestep,
            opt_step=opt_step, opt_iters=opt_iters,
            compress_factor=compress_factor, load_feature_step=load_feature_step,
            opt_attn_list=opt_attn_list,
            load_attn_list=load_attn_list,
            load_res_list=load_res_list,
            callback=generation_callback
        )

        sample = output.videos
        prompt_str_safe = "".join(filter(lambda x: x.isalnum() or x == "_", prompt.replace(" ", "_")))[:50]
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = Path(OUTPUT_DIR) / f"{time_str}-{prompt_str_safe}.gif"
        
        progress(1, desc="Saving video...")
        save_videos_grid(sample, str(save_path))
        print(f"Video saved to: {save_path}")

    return str(save_path)

# =============================================================================
# 5. GRADIO INTERFACE
# =============================================================================
with gr.Blocks(title="AnyI2V Video Generation") as demo:
    annotation_state = gr.State(initial_state)

    gr.Markdown("## AnyI2V: Animating Any Conditional Image with Motion Control")
    gr.Markdown("""### User Guide
        - **Load Data**: Use the `Examples` below or upload your own image.
        - **Annotate**: The first two clicks draw a Bounding Box. Subsequent clicks are Query Points. Use `Toggle Trajectory Mode` to draw motion paths.
        - **Add Group**: Click `➕ Add Data Group` to save the current set of annotations. You can add multiple groups for multiple objects.
        - **Generate**: After annotating, enter a prompt and click `🎬 Generate Video`.""")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Prompt", lines=2)
            n_prompt_input = gr.Textbox(label="Negative Prompt", value="cartoon, anime, sketches,worst quality, low quality, deformed, distorted, disfigured, bad eyes, wrong lips, weird mouth, " \
            "bad teeth, mutated hands and fingers, bad anatomy, wrong anatomy, amputation, extra limb, missing limb, floating limbs, disconnected limbs, mutation, ugly, disgusting, bad_pictures, negative_hand-neg", lines=2)
            
            with gr.Accordion("Annotation Tools", open=True):
                img_input = gr.Image(type="filepath", label="Upload Source Image")
                with gr.Row():
                    traj_btn = gr.Button("🚀 Toggle Trajectory Mode")
                    add_group_btn = gr.Button("➕ Add Data Group")
                    clear_btn = gr.Button("🔄 Clear All Inputs and Annotations")

            with gr.Accordion("Advanced Parameters", open=False):
                inference_steps_slider = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=hyper_params_config.inference_steps)
                guidance_scale_slider = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, label="Guidance Scale (CFG)", value=hyper_params_config.guidance_scale)
                inversion_timestep_slider = gr.Slider(minimum=1, maximum=999, step=10, label="Inversion Timestep", value=hyper_params_config.inversion_timestep)
                lr_number = gr.Number(label="Learning Rate (lr)", value=hyper_params_config.lr, info="e.g., 0.01")
                opt_iters_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Optimization Iters", value=hyper_params_config.opt_iters)
                pca_dim_slider = gr.Slider(minimum=16, maximum=256, step=16, label="PCA Dimension", value=hyper_params_config.PCA_dim)
                opt_step_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Optimization Step", value=hyper_params_config.opt_step)
                compress_factor_slider = gr.Slider(minimum=1, maximum=16, step=1, label="Compress Factor", value=hyper_params_config.compress_factor)
                load_feature_step_number = gr.Number(label="Load Feature Step", value=hyper_params_config.load_feature_step)
                debias_patch_size_slider = gr.Slider(minimum=1, maximum=16, step=1, label="Debias Patch Size", value=hyper_params_config.debias_patch_size)
                
                opt_attn_list_text = gr.Textbox(label="Optimization Attention List", value=', '.join(map(str, hyper_params_config.opt_attn_list)))
                load_attn_list_text = gr.Textbox(label="Load Attention List", value=', '.join(map(str, hyper_params_config.load_attn_list)))
                load_res_list_text = gr.Textbox(label="Load Residual List", value=', '.join(map(str, hyper_params_config.load_res_list)))

            generate_btn = gr.Button("🎬 Generate Video", variant="primary")
            status_output = gr.Textbox(label="System Status", interactive=False)

        with gr.Column(scale=2):
            video_output = gr.Video(label="Generated Video", interactive=False, height=512)
            img_display = gr.Image(label="Annotation Canvas", type="pil", interactive=False)
            
            with gr.Row():
                bbox_output = gr.Textbox(label="Bounding Box Coordinates", interactive=False, scale=1)
                points_output = gr.Textbox(label="Query Point Coordinates", interactive=False, scale=1)

    # <<< MODIFICATION START: Define hidden components and Examples >>>
    # These hidden textboxes will be populated by gr.Examples to trigger the change event
    with gr.Row(visible=False):
        example_image_path = gr.Textbox()
        example_json_path = gr.Textbox()

    gr.Examples(
        examples=[
            [
                "__assets__/image/horse-canny.jpg",
                "__assets__/trajectory/horse-canny.json",
                "A majestic horse running across a grassy field, cinematic lighting, high resolution."
            ],
        ],
        inputs=[example_image_path, example_json_path, prompt_input],
        label="Click an Example to Load"
    )
    # <<< MODIFICATION END >>>

    hyper_param_inputs = [
        inference_steps_slider, guidance_scale_slider, lr_number, opt_iters_slider, pca_dim_slider,
        opt_step_slider, compress_factor_slider, load_feature_step_number, inversion_timestep_slider,
        debias_patch_size_slider, opt_attn_list_text, load_attn_list_text, load_res_list_text
    ]

    # <<< MODIFICATION START: Wire the change event from the hidden component >>>
    example_image_path.change(
        fn=process_example,
        inputs=[example_image_path, example_json_path],
        outputs=[
            annotation_state,
            img_display,
            status_output,
            bbox_output,
            points_output,
        ]
    )
    # <<< MODIFICATION END >>>

    img_input.upload(
        fn=load_image, 
        inputs=[img_input, annotation_state], 
        outputs=[annotation_state, img_display, bbox_output, points_output, status_output]
    )
    img_display.select(
        fn=handle_click, 
        inputs=[annotation_state], 
        outputs=[annotation_state, img_display, bbox_output, points_output]
    )
    traj_btn.click(
        fn=toggle_trajectory_mode, 
        inputs=[annotation_state], 
        outputs=[annotation_state, img_display, status_output]
    )
    add_group_btn.click(
        fn=add_data_group, 
        inputs=[annotation_state], 
        outputs=[annotation_state, img_display, status_output]
    )

    clear_btn.click(
        fn=clear_all, 
        inputs=[annotation_state],
        outputs=[
            annotation_state, img_display, video_output, 
            bbox_output, points_output, status_output
        ], 
        js="() => confirm('Are you sure you want to clear all inputs and annotations? This action cannot be undone.')"
    )

    generate_btn.click(
        fn=run_generation,
        inputs=[prompt_input, n_prompt_input, annotation_state] + hyper_param_inputs,
        outputs=[video_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
