import argparse
import os
from pathlib import Path

from tqdm import tqdm

from imaginaire.auxiliary.text_encoder import CosmosTextEncoder
from imaginaire.constants import (
    get_cosmos_predict2_video2world_tokenizer,
    print_environment_info,
)
# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time

import torch
from megatron.core import parallel_state
import torchvision.io as io
import torchvision.transforms.functional as F
from cosmos_predict2.configs.pose_conditioned.pipeline import get_cosmos_predict2_pose_conditioned_pipeline
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import LazyCall as L
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from cosmos_predict2.pipelines.video2world_pose import Video2WorldPoseConditionedPipeline
from re import match


_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

FRAMES_PER_CHUNK = 45
NUM_CONDITIONING_FRAMES = 13

def read_video(video_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """
    Args:
        video_path: Path to the video file
        target_h: Target height for frames
        target_w: Target width for frames

    Returns:
        torch.Tensor: [C, T, H, W] in [0, 255] range
    """
    # Load video into memory (video: [T, H, W, C], dtype=uint8, range=[0, 255])
    video, _, _ = io.read_video(str(video_path), pts_unit="sec")

    # Resize to target_h and target_w
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
    video = F.resize(video, (target_h, target_w))
    video = video.permute(0, 2, 3, 1)  # [T, H, W, C]

    video = video.permute(3, 0, 1, 2)

    return video

def save_video_chunk(chunk: torch.Tensor, save_path: Path, fps = 16) -> None:
    """
    Save a video chunk to file.
    
    Args:
        chunk: Video tensor of shape [C, T, H, W] in [-1, 1] range
        save_path: Path to save the video
        fps: Frames per second for the saved video
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(str(save_path)):
        save_image_or_video(chunk, str(save_path), fps=fps)



def setup_pipeline(args: argparse.Namespace, text_encoder: CosmosTextEncoder | None = None):
    print_environment_info(args)

    config = get_cosmos_predict2_pose_conditioned_pipeline(
        model_size=args.model_size, resolution=args.resolution, fps=args.fps
    )

    if "fullbody" in args.pose_process_method:
        config.net.pose_dim = (32, 23, 6)
    else: # head only
        config.net.pose_dim = (32, 1, 6)

    # Pose pipeline by default uses the Fake Tokenizer (passes through latents directly).
    # We need to replace it with the actual tokenizer that encodes videos.
    config.tokenizer = L(TokenizerInterface)(
        chunk_duration=81,
        load_mean_std=False,
        name="tokenizer",
        vae_pth=get_cosmos_predict2_video2world_tokenizer(model_size="2B"),
    )

    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path
    else:
        raise ValueError("Please provide --dit_path to the finetuned model checkpoint.")

    log.info(f"Using dit_path: {dit_path}")

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")

        # Check if distributed environment is already initialized
        if not parallel_state.is_initialized():
            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
            log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")
        else:
            log.info("Distributed environment already initialized, skipping initialization")
            # Check if we need to reinitialize with different context parallel size
            current_cp_size = parallel_state.get_context_parallel_world_size()
            if current_cp_size != args.num_gpus:
                log.warning(f"Context parallel size mismatch: current={current_cp_size}, requested={args.num_gpus}")
                log.warning("Using existing context parallel configuration")
            else:
                log.info(f"Using existing context parallel group with {current_cp_size} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False
    config.guardrail_config.offload_model_to_cpu = args.offload_guardrail

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False
    config.prompt_refiner_config.offload_model_to_cpu = args.offload_prompt_refiner

    # Load models
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    pipe = Video2WorldPoseConditionedPipeline.from_config(
        config=config,
        dit_path=dit_path,
        use_text_encoder=text_encoder is None,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=args.load_ema,
        load_prompt_refiner=True,
    )

    # Set the provided text encoder if one was passed
    if text_encoder is not None:
        pipe.text_encoder = text_encoder

    log.info(f"Type of Tokenizer: {type(pipe.tokenizer)}")

    return pipe


def get_target_resolution(args: argparse.Namespace) -> tuple[int, int]:
    """
    Get target resolution based on args.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple of (height, width)
    """
    # Map resolution string to actual dimensions
    resolution_map = {
        "720": (960, 960),   # 720p 1:1
        "480": (480, 480),    # 480p 1:1
    }
    
    return resolution_map.get(args.resolution, (480, 480))


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

def parse_chunk_filename(filename: str) -> tuple[str | None, str | None]:
    """
    Return (base, id) from filenames like ..._chunk_0133.mp4.
    """
    m = match(r"^(.*)_chunk_(\d+)\.[^.]+$", filename)
    if not m:
        return None, None
    return m.group(1), m.group(2)