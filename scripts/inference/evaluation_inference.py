import argparse
import os
import sys
import traceback
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tqdm import tqdm

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time

import torch
from megatron.core import parallel_state
from cosmos_predict2.pipelines.video2world_pose import Video2WorldPoseConditionedPipeline
from imaginaire.utils import distributed, log
from imaginaire.constants import (
    CosmosPredict2Video2WorldModelSize, CosmosPredict2Video2WorldResolution,
    CosmosPredict2Video2WorldFPS, CosmosPredict2Video2WorldAspectRatio
)
from cosmos_predict2.data.pose_conditioned.dataset_utils import load_pose_chunk
from scripts.inference.utils import (
    read_video, save_video_chunk, get_target_resolution,
    cleanup_distributed, parse_chunk_filename, setup_pipeline,
    FRAMES_PER_CHUNK, NUM_CONDITIONING_FRAMES, _DEFAULT_NEGATIVE_PROMPT
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-to-World Evaluation Inference with Streaming Chunks")
    parser.add_argument(
        "--gt_folder",
        type=str,
        required=True,
        help="Ground truth folder containing all the .mp4 files",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="prediction",
        help="Experiment name for saving outputs",
    )
    parser.add_argument(
        "--model_size",
        choices=CosmosPredict2Video2WorldModelSize.__args__,
        default="2B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--resolution",
        choices=CosmosPredict2Video2WorldResolution.__args__,
        default="480",
        type=str,
        help="Resolution of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--fps",
        choices=CosmosPredict2Video2WorldFPS.__args__,
        default=16,
        type=int,
        help="FPS of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        help="Use EMA weights for generation.",
    )
    parser.add_argument(
        "--cond_frames",
        type=int,
        default=NUM_CONDITIONING_FRAMES,
        help="Number of conditioning frames",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=FRAMES_PER_CHUNK,
        help="Number of frames per chunk (default: 93)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=35,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative text prompt for video-to-world generation",
    )
    parser.add_argument(
        "--aspect_ratio",
        choices=CosmosPredict2Video2WorldAspectRatio.__args__,
        default="16:9",
        type=str,
        help="Aspect ratio of the generated output (width:height)",
    )
    parser.add_argument("--guidance", type=float, default=2, help="Guidance value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument("--offload_guardrail", action="store_true", help="Offload guardrail to CPU to save GPU memory")
    parser.add_argument("--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts")
    parser.add_argument("--offload_prompt_refiner", action="store_true", help="Offload prompt refiner to CPU to save GPU memory")
    parser.add_argument("--offload_text_encoder", action="store_true", help="Offload text encoder to CPU to save GPU memory")
    parser.add_argument(
        "--downcast_text_encoder",
        action="store_true",
        help="Cast text encoder from checkpoint precision to pipeline precision",
    )
    parser.add_argument(
        "--natten",
        action="store_true",
        help="Run Video2World + NATTEN (sparse attention variant).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of chunks to skip after each processed chunk. "
             "Example: 5 -> process chunk 0, skip 1-5, process 6, skip 7-11, etc.",
    )
    parser.add_argument(
        "--pose_process_method",
        type=str,
        default="fullbody",
        choices=['fullbody', 'head'],
        help="select which pose to feed the model with.",
    )
    parser.add_argument(
        "--pose_root",
        type=str,
        required=True,
        help="Root path where to find pose files corresponding to the mp4 video chunk.",
    )
    return parser.parse_args()

def process_video_evaluation(args: argparse.Namespace, pipe: Video2WorldPoseConditionedPipeline) -> None:
    """
    Main processing function for video evaluation inference.
    
    Args:
        args: Command line arguments
        pipe: Initialized Video2World pipeline
    """
    gt_folder = Path(args.gt_folder)
    
    if not gt_folder.exists():
        log.error(f"Ground truth folder does not exist: {gt_folder}")
        return

    video_files = sorted(glob(os.path.join(gt_folder, "*.mp4")))
    
    if not video_files:
        log.warning(f"No files found in {gt_folder}")
        return
    
    log.info(f"Found {len(video_files)} mp4 files to process")
    
    # Get target resolution
    target_h, target_w = get_target_resolution(args)
    log.info(f"Target resolution: {target_h}x{target_w}")
    log.info(f"Chunk size: {args.frames} frames")
    log.info(f"Conditioning frames: {args.cond_frames} frames")

    if args.skip > 0:
        log.info(f"Skip policy active: process 1 chunk then skip {args.skip} chunk(s) "
                 f"(i.e., process every {args.skip + 1}th chunk).")
    else:
        log.info("Skip policy inactive: processing all chunks.")

    # Create output directories
    prediction_dir = gt_folder.parent / args.exp_name

    prediction_dir.mkdir(exist_ok=True)

    log.info(f"Prediction videos will be saved to: {prediction_dir}")

    num_videos = len(video_files)
    
    # Process each video
    for idx in tqdm(range(0, num_videos, max(1, args.skip)), desc="Processing videos"):
        torch.cuda.synchronize()
        video_path = video_files[idx]
        log.info(f"Using video: {video_path} as GT")

        video_name = os.path.basename(video_path)
        chunk = read_video(video_path, target_h, target_w)

        chunk_frames = chunk.shape[1]
            
        # Skip chunks with fewer than cond frames
        if chunk_frames < args.cond_frames:
            log.warning(f"Skipping chunk {idx} - only {chunk_frames} frames, need at least {args.cond_frames}")
            continue

        # Ensure only args.frames
        chunk = chunk[:, :args.frames]

        source_video, chunk_id = parse_chunk_filename(video_name)
        pose_path = os.path.join(
            args.pose_root,
            source_video,
            f"pose_{args.fps}.0fps_{args.frames}f",
            f"00{chunk_id}_*.pt",
        )
        # get the first matching pose file
        pose_files = glob(pose_path)
        if not pose_files:
            log.warning(f"No pose file found for chunk {video_name}, skipping.")
            continue
        pose_file = pose_files[0]
        pose_tensor = load_pose_chunk(pose_file,
                                      num_future_frames=chunk_frames - args.cond_frames,
                                      process_method=args.pose_process_method)

        if pose_tensor is not None:
            log.success(f"Loaded pose tensor from {pose_file} with shape {pose_tensor.shape}")

        # We had some sync problem so we just put a sleep here.
        time.sleep(.5)
        torch.cuda.synchronize()
        try:
            # Generate prediction using pipeline
            generated_video = pipe(
                poses=pose_tensor,
                video=chunk.unsqueeze(0),
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_conditional_frames=args.cond_frames,
                guidance=args.guidance,
                seed=args.seed,            )

            if generated_video is not None:
                # Save prediction
                pred_save_path = prediction_dir / video_name
                save_video_chunk(generated_video[0], pred_save_path, fps=args.fps)
                log.success(f"Saved prediction: {pred_save_path}")
            else:
                log.warning(f"Failed to generate video {video_name}")
                    
        except Exception as e:
            # Log the full exception with traceback
            log.error(f"{e}")
            traceback.print_exc()
        finally:
            torch.cuda.synchronize()
            time.sleep(.5)
            torch.cuda.synchronize()

    log.success(f"Finished processing all videos.")


if __name__ == "__main__":
    args = parse_args()
    try:
        pipe = setup_pipeline(args)
        process_video_evaluation(args, pipe)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
