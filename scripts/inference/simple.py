import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import traceback
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np


import time
import torch
import cv2
from megatron.core import parallel_state
from cosmos_predict2.pipelines.video2world_pose import Video2WorldPoseConditionedPipeline
from imaginaire.utils import log
from cosmos_predict2.data.pose_conditioned.dataset_utils import (
    load_pose_chunk,
    process_poses,
)

from scripts.inference.utils import (
    read_video, save_video_chunk, get_target_resolution,
    setup_pipeline, NUM_CONDITIONING_FRAMES, _DEFAULT_NEGATIVE_PROMPT
)

from scripts.inference.viz_utils import build_pose_video, make_viz_output_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--poses",
        type=str,
        help="Pose input: a folder of pose files, a comma-separated list, or a single pose file path.",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./",
        help="Output directory to save generated video.",
    )
    parser.add_argument(
        "--cond_frames",
        type=int,
        default=NUM_CONDITIONING_FRAMES,
    )
    parser.add_argument(
        "--pose_process_method",
        type=str,
        default="fullbody",
        choices=["fullbody", "head"],
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--model_size",
        default='2B',
    )
    parser.add_argument(
        "--resolution",
        default="480",
        type=str,
        help="Resolution of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--fps",
        default=16,
        type=int,
        help="FPS of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--dit_path",
        required=True,
        type=str,
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--disable_guardrail",
        action="store_true",
    )
    parser.add_argument(
        "--offload_guardrail",
        action="store_true",
    )
    parser.add_argument(
        "--disable_prompt_refiner",
        action="store_true",
    )
    parser.add_argument(
        "--offload_prompt_refiner",
        action="store_true",
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=35,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--viz_pose",
        action="store_true",
        help="Save a side-by-side visualization with generated video on the left and pose on the right.",
    )
    parser.add_argument("--autoregressive", action="store_true", help="Use autoregressive generation.")
    return parser.parse_args()


def resolve_pose_paths(poses_arg: str) -> list[str]:
    poses_arg = poses_arg.strip()
    if "," in poses_arg:
        return [pose.strip() for pose in poses_arg.split(",") if pose.strip()]

    pose_path = Path(poses_arg).expanduser()
    if pose_path.is_dir():
        poses = sorted(
            [
                str(path)
                for path in pose_path.iterdir()
                if path.is_file() and path.suffix in {".pt", ".pth"}
            ]
        )
        if not poses:
            raise FileNotFoundError(f"No pose files found in folder: {pose_path}")
        return poses

    return [poses_arg]


def pose_source_label(poses_arg: str, poses_paths: list[str]) -> str:
    poses_arg = poses_arg.strip()
    if "," in poses_arg:
        if not poses_paths:
            return "poses"
        first = Path(poses_paths[0]).stem
        last = Path(poses_paths[-1]).stem
        return first if first == last else f"{first}_to_{last}"

    pose_path = Path(poses_arg).expanduser()
    if pose_path.is_dir():
        return pose_path.name
    return pose_path.stem

def generate_video(args, pipe: Video2WorldPoseConditionedPipeline):
    """Generate video frames given input video frames and pose frames."""
    resolution = get_target_resolution(args)
    video = read_video(args.video, *resolution)

    num_frames = video.shape[1]

    poses_paths = resolve_pose_paths(args.poses)
    poses_label = pose_source_label(args.poses, poses_paths)

    long_video = []
    long_pose_video = [] if args.autoregressive and args.viz_pose else None

    # TODO fix the autoregressive inference
    print(f"Using {len(poses_paths)} pose files for generation.")
    if args.autoregressive:
        # Load all the poses at once
        poses = [torch.load(pose, weights_only=False)["poses"] for pose in poses_paths]
        pose = {
            'global_translations': np.concatenate([p['global_translations'] for p in poses], axis=0),
            'global_rotations_euler': np.concatenate([p['global_rotations_euler'] for p in poses], axis=0),
        }
        print(pose['global_translations'].shape)
        pose = process_poses(pose, num_future_frames=len(pose['global_translations']) - args.cond_frames, process_method=args.pose_process_method)
        pose = pose.reshape(pose.shape[0], -1)

        print(pose.shape)
        if pose.shape[0] % (num_frames - args.cond_frames) != 0:
            # Drop frames to make it divisible
            drop_frames = pose.shape[0] % (num_frames - args.cond_frames)
            pose = pose[:-drop_frames]

        poses = np.split(pose, pose.shape[0] // (num_frames - args.cond_frames))
        if len(poses) != len(poses_paths):
            log.warning(f"Number of pose chunks {len(poses)} does not match number of pose files {len(poses_paths)}. Using the minimum.")
            min_len = min(len(poses), len(poses_paths))
            poses = poses[:min_len]
            poses_paths = poses_paths[:min_len]

    for i, pose_path in enumerate(poses_paths):
        log.info(f"Processing pose file: {pose_path}")
        if args.autoregressive:
            pose = torch.from_numpy(poses[i]).to(pipe.device)
        else:
            pose = load_pose_chunk(pose_path, num_future_frames=num_frames-args.cond_frames, process_method=args.pose_process_method)

        if pose is not None:
            log.success("Pose loaded successfully. Shape : {}".format(pose.shape))
        else:
            log.error("Failed to load pose from {}".format(pose_path))
            return

        try:
            print(pose.shape, video.shape)
            print(video.dtype)
            generated_video = pipe(
                poses=pose,
                video=video.unsqueeze(0),
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_conditional_frames=args.cond_frames,
                guidance=args.guidance,
                seed=args.seed,
                num_sampling_step=args.steps,
            )

            if generated_video is not None:
                output_path=os.path.join(args.output, f"generated_from_{Path(pose_path).stem}_on_{Path(args.video).stem}.mp4")
                save_video_chunk(
                    generated_video[0],
                    Path(output_path),
                    fps=float(args.fps),
                )
                log.success(f"Generated video saved to {output_path}")
                if args.viz_pose:
                    pose_video = build_pose_video(
                        pose_path=pose_path,
                        num_frames=generated_video[0].shape[1],
                        cond_frames=args.cond_frames,
                        height=generated_video[0].shape[2],
                        width=generated_video[0].shape[3],
                    )
                    generated_cpu = generated_video[0].detach().float().cpu()
                    side_by_side = torch.cat([generated_cpu, pose_video.to(generated_cpu.dtype)], dim=3)
                    viz_output_path = make_viz_output_path(output_path)
                    save_video_chunk(
                        side_by_side,
                        Path(viz_output_path),
                        fps=float(args.fps),
                    )
                    log.success(f"Pose visualization saved to {viz_output_path}")
                    if args.autoregressive and long_pose_video is not None:
                        if i == 0:
                            long_pose_video.append(pose_video)
                        else:
                            long_pose_video.append(pose_video[:, args.cond_frames:])
            else:
                log.error("Video generation failed.")

            if args.autoregressive:
                if i == 0:
                    # Include also the conditional frames from the first chunk
                    long_video.append(generated_video[0])
                else:
                    long_video.append(generated_video[0][:, args.cond_frames:])
                video = generated_video[0][:, -args.cond_frames:]
                # Convert the video to [0, 255] range and uint8 type for the next iteration
                video = (video.clamp(-1, 1) + 1) / 2 * 255
                # The pipeline expects num_frames of input, but we only have cond_frames of real content.
                # Tile the conditioning frames up to at least num_frames, then slice to the exact length.
                # Tiling is preferred over zero-padding (see commented line below) because it fills the
                # "dummy" frames with plausible pixel values rather than black, avoiding generation bias.
                # Assumption: 4 * cond_frames >= num_frames.
                video = torch.cat([video, video, video, video], dim=1)[:, :num_frames]
                video = video.type(torch.uint8)
                video = video.to(pipe.device)

        except Exception as e:
            log.error("An error occurred during video generation:")
            log.error(traceback.format_exc())
        time.sleep(0.5)
        torch.cuda.synchronize()

    if args.autoregressive:
        long_video = torch.cat(long_video, dim=1)
        output_path=os.path.join(args.output, f"generated_from_{poses_label}_on_{Path(args.video).stem}_autoregressive.mp4")
        save_video_chunk(
            long_video,
            Path(output_path),
            fps=float(args.fps),
        )
        log.success(f"Generated long video saved to {output_path}")
        if args.viz_pose and long_pose_video:
            long_pose_video = torch.cat(long_pose_video, dim=1)
            long_video_cpu = long_video.detach().float().cpu()
            side_by_side = torch.cat([long_video_cpu, long_pose_video.to(long_video_cpu.dtype)], dim=3)
            viz_output_path = make_viz_output_path(output_path)
            save_video_chunk(
                side_by_side,
                Path(viz_output_path),
                fps=float(args.fps),
            )
            log.success(f"Pose visualization saved to {viz_output_path}")

def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()

    try:
        pipe = setup_pipeline(args)
        generate_video(args, pipe)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()