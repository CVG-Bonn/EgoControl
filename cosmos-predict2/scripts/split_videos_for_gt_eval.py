from imaginaire.utils.io import save_image_or_video
import argparse
import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from tqdm import tqdm
from imaginaire.utils import log

FRAMES_PER_CHUNK = 45

def resize_frame_np(
        frame_rgb: np.ndarray,
        target_h: int,
        target_w: int,
) -> np.ndarray:
    """
    Resize frame using cv2 with aspect-aware letterbox approach.

    Args:
        frame_rgb: HxWx3 uint8 RGB frame
        target_h: Target height
        target_w: Target width

    Returns:
        HxWx3 uint8 RGB frame of size target_h x target_w
    """
    h, w = frame_rgb.shape[:2]

    # Letterbox approach
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Pad to target size
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding
    )


def save_video_chunk(chunk: torch.Tensor, save_path: Path, fps: int = 16) -> None:
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


def read_video_chunks(video_path: Path, target_h: int, target_w: int, frame_num: int) -> Iterator[torch.Tensor]:
    """
    Stream video frames in chunks of FRAMES_PER_CHUNK using cv2.

    Args:
        video_path: Path to the video file
        target_h: Target height for frames
        target_w: Target width for frames

    Yields:
        torch.Tensor: Video chunks of shape [C, T, H, W] in [-1, 1] range
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Failed to open video: {video_path}")
        return

    try:
        chunk_frames = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Resize frame
            frame_resized = resize_frame_np(frame_rgb, target_h, target_w)

            # Convert to tensor in [-1, 1] range
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor * 2.0 - 1.0  # Convert to [-1, 1]

            chunk_frames.append(frame_tensor)

            # When chunk is full, yield it
            if len(chunk_frames) == frame_num:
                # Stack to [T, C, H, W] then permute to [C, T, H, W]
                chunk_tensor = torch.stack(chunk_frames, dim=0).permute(1, 0, 2, 3)
                yield chunk_tensor
                chunk_frames = []

        # Yield last chunk if it has frames
        if chunk_frames:
            # Stack to [T, C, H, W] then permute to [C, T, H, W]
            chunk_tensor = torch.stack(chunk_frames, dim=0).permute(1, 0, 2, 3)
            yield chunk_tensor

    finally:
        cap.release()


def find_recording_videos(root_folder: Path) -> list[Path]:
    """
    Find all recording.mp4 files in subfolders of the root folder.

    Args:
        root_folder: Root directory to search

    Returns:
        List of paths to recording.mp4 files
    """
    return sorted(list(root_folder.rglob("recording_head.mp4")))


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
        "720": (960, 960),  # 720p 1:1
        "480": (480, 480),  # 480p 1:1
    }

    return resolution_map.get(args.resolution, (480, 480))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-to-World Evaluation Inference with Streaming Chunks")
    parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
        help="Root folder containing subfolders with recording.mp4 files",
    )
    parser.add_argument(
        "--resolution",
        choices=["480", "720"],
        default="720",
        type=str,
        help="Resolution of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--fps",
        choices=[16],
        default=16,
        type=int,
        help="FPS of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=FRAMES_PER_CHUNK,
        help="Number of frames per chunk (default: 45)",
    )
    return parser.parse_args()


def process_video_evaluation(args: argparse.Namespace) -> None:
    """
    Main processing function for video evaluation inference.

    Args:
        args: Command line arguments
        pipe: Initialized Video2World pipeline
    """
    root_folder = Path(args.root_folder)

    if not root_folder.exists():
        log.error(f"Root folder does not exist: {root_folder}")
        return

    # Find all recording.mp4 files
    video_files = find_recording_videos(root_folder)[:9]

    if not video_files:
        log.warning(f"No recording.mp4 files found in {root_folder}")
        return

    log.info(f"Found {len(video_files)} recording.mp4 files to process")

    if not video_files:
        log.warning(f"No recording.mp4 files found in {root_folder}")
        return

    # Get target resolution
    target_h, target_w = get_target_resolution(args)
    log.info(f"Target resolution: {target_h}x{target_w}")
    log.info(f"Chunk size: {args.frames} frames")

    # Create output directories
    ground_truth_dir = root_folder.parent / f"ground_truth_{args.frames}f_960"

    ground_truth_dir.mkdir(exist_ok=True)

    log.info(f"Ground truth videos will be saved to: {ground_truth_dir}")

    total_chunks_processed = 0

    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        log.info(f"Processing video: {video_path}")

        # Get relative path from root for naming
        rel_path = video_path.relative_to(root_folder)
        video_name = rel_path.parent.name  # Use parent folder name

        chunk_idx = 0
        processed_chunks_for_video = 0  # counts successfully attempted (non-skipped) chunks

        # Process chunks from this video
        for chunk in read_video_chunks(video_path, target_h, target_w, args.frames):
            chunk_frames = chunk.shape[1]
            log.info(f"Processing chunk {chunk_idx} for video {video_name} ({chunk_frames} frames)")

            # Skip chunks with fewer than NUM_CONDITIONING_FRAMES frames
            if chunk_frames < args.frames:
                log.warning(
                    f"Skipping chunk {chunk_idx} - only {chunk_frames} frames, need at least {args.frames}")
                chunk_idx += 1
                continue

            # Save ground truth chunk
            gt_save_path = ground_truth_dir / f"{video_name}_chunk_{chunk_idx:04d}.mp4"
            if os.path.exists(gt_save_path):
                log.info("Loaded existing ground truth chunk from disk.")
            else:
                save_video_chunk(chunk, gt_save_path, fps=args.fps)
                log.info(f"Saved ground truth: {gt_save_path}")

            processed_chunks_for_video += 1
            chunk_idx += 1
            total_chunks_processed += 1
    log.success(f"Finished processing all videos. Total chunks processed: {total_chunks_processed}")

if __name__ == "__main__":
    args = parse_args()
    try:
        process_video_evaluation(args)
    except Exception as e:
        log.error(f"An error occurred: {e}")
        raise
