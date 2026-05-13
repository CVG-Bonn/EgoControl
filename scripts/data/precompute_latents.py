# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0
#
# Precompute per-clip VAE latents for videos under a dataset root.
#
# What this script does:
# - Recursively finds videos named `recording_head.mp4` by default (configurable via `--video_name`).
# - Decodes frames in streaming mode with OpenCV (no full-video buffering in RAM).
# - Optionally resamples time to `--target_fps`; original FPS is resolved from `--orig_fps`,
#   then cv2 metadata, then ffprobe fallback.
# - Resizes each frame to `--height`/`--width` using `letterbox` (default), `center_crop`, or `stretch`.
# - Rounds spatial size down to multiples of 8 (minimum 8x8) for VAE compatibility.
# - Splits the processed stream into clips (`--frames_per_clip`, default 93), keeping the final short clip.
# - Encodes each clip with the Cosmos VAE tokenizer and saves `.pt` files beside each video in:
#   `latents_{height}x{width}_{target_fps}fps_{frames_per_clip}f/`
# - Stores per-clip metadata in each file: `start_frame`, `end_frame`, `num_frames`,
#   `height`, `width`, and `target_fps`.
# - Supports partial dataset runs with `--video_range` and resumable runs with `--skip_existing`.
#
# Example:
#   python -m scripts.data.precompute_latents \
#     --root_dir /path/to/dataset \
#     --video_name recording_head.mp4 \
#     --vae_path /path/to/vae.ckpt \
#     --height 480 --width 848 \
#     --target_fps 16 \
#     --orig_fps 30 \
#     --frames_per_clip 93 \
#     --resize_mode letterbox \
#     --dtype bf16 \
#     --video_range 0:-1 \
#     --skip_existing
#
# Dependency:
# - OpenCV (`opencv-python`) must be installed.

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm

# OpenCV for streaming decode
import cv2

from imaginaire.utils import log
from cosmos_predict2.tokenizers.tokenizer import VAE

DEFAULT_FRAMES_PER_CLIP = 93


def _round_to_multiple_of(x: int, m: int) -> int:
    return (x // m) * m


def _ensure_multiple_of_8(h: int, w: int) -> tuple[int, int]:
    # VAE spatial compression factor is 8; ensure divisibility
    h8 = _round_to_multiple_of(h, 8)
    w8 = _round_to_multiple_of(w, 8)
    # Avoid zero if user passes tiny values
    return max(h8, 8), max(w8, 8)


def list_target_videos(root: Path, video_name: str) -> list[Path]:
    # Find all files exactly matching the target name under root
    return sorted([p for p in root.rglob(video_name) if p.is_file()])


def resize_frame_np(
    frame_rgb: np.ndarray,
    target_h: int,
    target_w: int,
    mode: str,
) -> np.ndarray:
    """
    frame_rgb: HxWx3 uint8 RGB
    Returns HxWx3 uint8 RGB of size target_h x target_w
    """
    h, w = frame_rgb.shape[:2]

    if mode == "stretch":
        return cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    if mode == "letterbox":
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
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
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    if mode == "center_crop":
        scale = max(target_w / w, target_h / h)
        fill_w = max(1, int(round(w * scale)))
        fill_h = max(1, int(round(h * scale)))
        filled = cv2.resize(frame_rgb, (fill_w, fill_h), interpolation=cv2.INTER_CUBIC)
        top = max(0, (fill_h - target_h) // 2)
        left = max(0, (fill_w - target_w) // 2)
        return filled[top : top + target_h, left : left + target_w, :]

    raise ValueError(f"Unknown resize_mode: {mode}")


def _ffprobe_fps(video_path: Path) -> Optional[float]:
    """Get FPS using ffprobe, returns None on failure."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        fps_str = info["streams"][0]["r_frame_rate"]
        num, den = map(int, fps_str.split("/"))
        if den == 0:
            return None
        return num / den
    except Exception as e:
        log.warning(f"ffprobe FPS failed for {video_path}: {e}")
        return None


def detect_original_fps(video_path: Path, orig_fps_cli: Optional[float], cap: cv2.VideoCapture) -> Optional[float]:
    # 1) CLI override
    if orig_fps_cli and orig_fps_cli > 0:
        return float(orig_fps_cli)

    # 2) cv2 property (may be unreliable on some containers)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    if cap_fps and cap_fps > 1e-3:
        return float(cap_fps)

    # 3) ffprobe as fallback
    return _ffprobe_fps(video_path)


def iter_resampled_frames(
    cap: cv2.VideoCapture,
    video_path: Path,
    target_fps: Optional[float],
    orig_fps: Optional[float],
) -> Iterable[np.ndarray]:
    """
    Stream frames from cv2.VideoCapture with optional temporal resampling.
    Yields RGB uint8 frames one by one.
    """
    if not cap.isOpened():
        log.error(f"Failed to open video: {video_path}")
        return

    # Decide resampling schedule
    use_resample = target_fps and target_fps > 0
    if use_resample:
        if not orig_fps or orig_fps <= 0:
            log.warning(
                f"Could not determine original FPS for {video_path}; disabling temporal resampling. "
                f"Hint: pass --orig_fps explicitly."
            )
            use_resample = False

    if not use_resample:
        # Pass-through: yield every decoded frame
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_rgb
        return

    # Time-based schedule to avoid drifting and without needing total frame count
    step_in = 1.0 / float(orig_fps)  # time increment per input frame
    step_out = 1.0 / float(target_fps)  # desired time increment per output frame
    t_in = 0.0
    t_out = 0.0
    eps = 1e-9

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t_in += step_in
        # Yield duplicates if upsampling, or drop frames if downsampling
        produced = 0
        while t_out <= t_in + eps:
            yield frame_rgb
            t_out += step_out
            produced += 1
        # produced may be 0 if target_fps << orig_fps and current frame falls between output ticks


@torch.no_grad()
def encode_clip_latents(vae: VAE, clip_BCTHW: torch.Tensor) -> torch.Tensor:
    """
    clip_BCTHW: [1, C, T, H, W] in [-1, 1]
    Returns latents tensor (shape depends on VAE config), on CPU.
    """
    latents = vae.encode(clip_BCTHW)
    return latents.cpu()


def process_video_streaming(
    video_path: Path,
    vae: VAE,
    frames_per_clip: int,
    target_h: int,
    target_w: int,
    resize_mode: str,
    device: torch.device,
    skip_existing: bool,
    target_fps: Optional[float],
    orig_fps_cli: Optional[float],
) -> None:
    rel = str(video_path)
    log.info(f"Opening video (streaming): {rel}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Failed to open video: {rel}")
        return

    # Determine original FPS for resampling if needed
    orig_fps = detect_original_fps(video_path, orig_fps_cli, cap)
    if target_fps and target_fps > 0:
        if orig_fps and orig_fps > 0:
            log.info(f"Temporal resampling: original {orig_fps:.3f} -> target {target_fps:.3f} FPS")
        else:
            log.warning(f"Temporal resampling requested but original FPS not found; skipping resample")

    # Ensure 8x multiple for VAE; adjust target dims if needed
    adj_h, adj_w = _ensure_multiple_of_8(target_h, target_w)
    if (adj_h, adj_w) != (target_h, target_w):
        log.warning(f"Adjusted target size to multiples of 8: ({target_h},{target_w}) -> ({adj_h},{adj_w})")
        target_h, target_w = adj_h, adj_w

    # Prepare output folder beside the video
    latents_dir = video_path.parent / f"latents_{target_h}x{target_w}_{target_fps}fps_{frames_per_clip}f"
    latents_dir.mkdir(parents=True, exist_ok=True)

    # Iterate frames with resampling
    out_index = 0  # index in the resampled timeline
    clip_start_index = 0
    clip_frames: list[torch.Tensor] = []
    clip_number = 0

    def flush_clip(end_index_inclusive: int):
        nonlocal clip_frames, clip_start_index, out_index, clip_number
        if not clip_frames:
            return
        # Stack to [T, C, H, W] -> [C, T, H, W]
        clip_TCHW = torch.stack(clip_frames, dim=0)  # [T, C, H, W], float32 in [-1, 1]
        clip_CTHW = clip_TCHW.permute(1, 0, 2, 3).contiguous()
        clip_BCTHW = clip_CTHW.unsqueeze(0).to(device)

        add_idx = 0 if clip_number == 0 else 1
        out_path = latents_dir / f"{clip_number:06d}_{clip_start_index:06d}_{end_index_inclusive+add_idx:06d}.pt"
        if skip_existing and out_path.exists():
            log.info(f"Skipping existing latents: {out_path.name}")
            clip_frames = []
            clip_start_index = out_index
            return

        try:
            latents = encode_clip_latents(vae, clip_BCTHW)
            torch.save(
                {
                    "latents": latents,  # CPU tensor
                    "start_frame": clip_start_index,
                    "end_frame": end_index_inclusive,
                    "num_frames": end_index_inclusive - clip_start_index + 1,
                    "height": target_h,
                    "width": target_w,
                    "target_fps": float(target_fps) if target_fps else None,
                },
                out_path,
            )
            log.info(f"Saved latents: {out_path}")
        except Exception as e:
            log.error(f"Failed to encode/save clip [{clip_start_index}:{end_index_inclusive}] for {rel}: {e}")
        finally:
            clip_frames = []
            clip_start_index = out_index
            clip_number += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()

    try:
        for frame_rgb in iter_resampled_frames(cap, video_path, target_fps, orig_fps):
            # Resize on CPU using OpenCV (fast), keep memory minimal
            frame_resized = resize_frame_np(frame_rgb, target_h, target_w, resize_mode)
            # To tensor in [-1, 1]
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor * 2.0 - 1.0  # [-1, 1]
            clip_frames.append(frame_tensor)

            # When clip full, flush
            if len(clip_frames) == frames_per_clip:
                flush_clip(end_index_inclusive=out_index)
            out_index += 1

        # Flush last (possibly shorter) clip
        if len(clip_frames) > 0:
            flush_clip(end_index_inclusive=out_index - 1)

        log.success(f"Done: {rel} — processed {out_index} resampled frames")
    except Exception as e:
        log.error(f"Streaming processing failed for {rel}: {e}")
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-clip latents from videos (streaming, low-memory)")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing video folders")
    parser.add_argument(
        "--video_name",
        type=str,
        default="recording_head.mp4",
        help="Name of the video file inside each folder (default: recording_head.mp4)",
    )
    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--z_dim", type=int, default=16, help="Latent channels (z_dim) for VAE")
    parser.add_argument("--height", type=int, required=True, help="Target height (rounded to multiple of 8)")
    parser.add_argument("--width", type=int, required=True, help="Target width (rounded to multiple of 8)")
    parser.add_argument(
        "--resize_mode",
        type=str,
        choices=["letterbox", "center_crop", "stretch"],
        default="letterbox",
        help="How to fit aspect ratio",
    )
    parser.add_argument("--frames_per_clip", type=int, default=DEFAULT_FRAMES_PER_CLIP, help="Frames per clip")
    parser.add_argument(
        "--target_fps",
        type=float,
        default=None,
        help="Target FPS for processing (e.g., 16). If set, resamples from original FPS.",
    )
    parser.add_argument(
        "--orig_fps",
        type=float,
        default=None,
        help="Original FPS override (if detection from cv2/ffprobe is unreliable). Example: 30.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run encoding on (cuda or cpu)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip encoding clips that already have a saved latents file",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Compute dtype for AMP (bf16 recommended)",
    )
    parser.add_argument(
        "--video_range",
        type=str,
        default="0:-1",
        help="Range of videos to process, e.g. '0:-1' (default) or '10:20'",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        try:
            log.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            log.info("Using CUDA device")
    else:
        log.warning("Using CPU — encoding may be slow.")

    # Resolve dtype for AMP
    if args.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    # Initialize tokenizer (VAE)
    log.info(f"Loading VAE tokenizer: z_dim={args.z_dim}, checkpoint={args.vae_path}")
    vae = VAE(
        z_dim=args.z_dim,
        vae_pth=args.vae_path,
        load_mean_std=False,
        dtype=amp_dtype,
        device=device.type,
        is_amp=True,  # enable autocast
        temporal_window=16,
    )

    root = Path(args.root_dir)
    if not root.exists():
        log.error(f"Root directory does not exist: {root}")
        return

    video_files = list_target_videos(root, args.video_name)

    start_idx, end_idx = map(int, args.video_range.split(":"))
    video_files = video_files[start_idx:end_idx + 1] if end_idx != -1 else video_files[start_idx:]

    if not video_files:
        log.warning(f"No '{args.video_name}' files found under: {root}")
        return

    log.info(f"Found {len(video_files)} videos named '{args.video_name}', using range {args.video_range}")
    start_time = time.time()
    for vp in tqdm(video_files):
        try:
            process_video_streaming(
                vp,
                vae=vae,
                frames_per_clip=args.frames_per_clip,
                target_h=args.height,
                target_w=args.width,
                resize_mode=args.resize_mode,
                device=device,
                skip_existing=args.skip_existing,
                target_fps=args.target_fps,
                orig_fps_cli=args.orig_fps,
            )
        except Exception as e:
            log.error(f"Unexpected error while processing {vp}: {e}")
    end_time = time.time()
    elapsed = end_time - start_time
    log.info(f"Processed {len(video_files)} videos in {elapsed/60.0:.1f} minutes")
    log.success("All done.")


if __name__ == "__main__":
    main()
