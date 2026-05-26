"""
Dataset class for pose-conditioned video generation.
Adapted from action_conditioned_dataset.py for 3D body pose control signals.
"""

import os
import random
import traceback
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from cosmos_predict2.data.pose_conditioned.dataset_utils import get_latent_num_frames, load_pose_chunk

from cosmos_predict2.data.action_conditioned.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
)

from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig


class PoseConditionedDataset(Dataset):
    def __init__(
            self,
            video_root=None,
            split="train",
            num_frames=93,
            num_conditioning_frames=5,
            video_size=(480, 480),  # 1:1 aspect ratio as specified
            load_poses=True,
            flatten_pose=True,
            debug=False,
            mode="train",
            use_precomputed_latents=False,
            latents_fps=16,
            num_pose_element=23,
            pose_process_method='fullbody', # 'fullbody' or 'head'
    ):
        """Dataset class for loading 3D body pose-conditioned data.

        This dataset loads video sequences with corresponding 3D body pose annotations
        for pose-conditioned video generation. Supports both video loading and precomputed latents.

        Args:
            video_root (str): Root path to video files
            split (str): Dataset split - 'train', 'val', or 'test'
            num_frames (int): Total number of frames to load per sequence (default: 93)
            num_conditioning_frames (int): Number of conditioning frames (default: 5)
            video_size (tuple): Target size (H, W) for video frames (default: (480, 480) for 1:1 aspect ratio)
            load_poses (bool): Whether to load poses or return fake ones
            normalize_pose (bool): Whether to normalize pose coordinates
            flatten_pose (bool): Whether to flatten pose tensor from [T, J, 3] to [T, J*3]
            debug (bool): If True, only loads subset of data
            mode (str): Dataset mode - 'train', 'val' or 'test'
            use_precomputed_latents (bool): Whether to load precomputed latents instead of videos
            latents_fps (int): FPS used for latent encoding (default: 16)

        The dataset expects:
        - If use_precomputed_latents=False: Video files in video_root/split/
        - If use_precomputed_latents=True: Video folders with latents_{H}x{W}_{FPS}fps subfolders
        - Pose annotation JSON files in annotation_root/split/ (if load_poses=True)
        - Each JSON contains pose_world: [T, J, 3] array of 3D joint coordinates
        """
        self.video_dir = os.path.join(video_root, split) if video_root else None
        self.split = split
        self.num_frames = num_frames
        self.num_conditioning_frames = num_conditioning_frames
        self.num_future_frames = num_frames - num_conditioning_frames  # 88 frames
        self.video_size = video_size
        self.load_poses = load_poses
        self.flatten_pose = flatten_pose
        self.mode = mode
        self.use_precomputed_latents = use_precomputed_latents
        self.fps = latents_fps
        self.wrong_number = 0  # Tracks number of invalid samples encountered
        self.num_pose_element = num_pose_element
        self.pose_process_method = pose_process_method

        # Latents folder name based on resolution and fps
        self.latents_folder_name = f"latents_{video_size[0]}x{video_size[1]}_{latents_fps}.0fps_{num_frames}f"
        # Pose folder name based on fps
        self.pose_folder_name = f"pose_{latents_fps}.0fps_{num_frames}f"

        # Video preprocessing transforms
        self.transform = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(video_size)),
            ]
        )

        self.samples = []

        if os.path.exists(self.video_dir):
            for item in os.listdir(self.video_dir):
                item_path = os.path.join(self.video_dir, item)
                if os.path.isdir(item_path):
                    latents_path = os.path.join(item_path, self.latents_folder_name)
                    if os.path.exists(latents_path) and os.path.isdir(latents_path):
                        self.samples.extend(sorted(glob(os.path.join(latents_path, "*.pt"))[:-1]))

        if debug:
            self.samples = self.samples[:20]

    def __len__(self):
        return len(self.samples)

    def _load_precomputed_latents(self, sample_path):
        """Load precomputed latent chunks for a sample."""
        try:
            latents = torch.load(sample_path, map_location='cpu')["latents"][0]  # Assuming shape [1, C, T, H, W]

            # Ensure we have the right number of (latent) frames
            if latents.shape[1] != get_latent_num_frames(self.num_frames):  # Assuming shape is [C, T, H, W]
                # Pad if necessary
                padding_frames = get_latent_num_frames(self.num_frames) - latents.shape[1]
                last_frame = latents[:, -1:].expand(-1, padding_frames, -1, -1)
                latents = torch.cat([latents, last_frame], dim=1)

            return latents

        except Exception as e:
            print(f"Error loading precomputed latents for {sample_path}: {e}")
            # Return dummy latents with expected shape [C, T, H, W]
            return torch.zeros(16, self.num_frames, self.video_size[0] // 8, self.video_size[1] // 8)

    def _get_fake_poses(self):
        """Generate fake pose data when poses are not available or load_poses=False."""
        if self.flatten_pose:
            # Return shape [T, J*6] for future frames only
            return np.ones((self.num_future_frames, self.num_pose_element * 6), dtype=np.float32)
        else:
            # Return shape [T, J, 6] for future frames only
            return np.ones((self.num_future_frames, self.num_pose_element, 6), dtype=np.float32)


    def _load_poses(self, sample_path):
        """Load precomputed pose data for a sample."""

        sample_path = Path(sample_path)
        poses_dir = os.path.join(sample_path.parent.parent, self.pose_folder_name)
        if not os.path.exists(poses_dir):
            raise FileNotFoundError(f"Pose directory not found: {poses_dir}")

        pose_path = os.path.join(poses_dir, f"{sample_path.stem}.pt")
        return load_pose_chunk(pose_path,
                                   num_future_frames=self.num_future_frames,
                                   process_method=self.pose_process_method)

    def __getitem__(self, index):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)
        try:
            sample_path = self.samples[index]

            # Load precomputed latents instead of video
            latents = self._load_precomputed_latents(sample_path)

            data = {
                "video": latents.float(),  # Target frames to predict
            }

            if self.load_poses:
                pose = self._load_poses(sample_path)
            else:
                pose = torch.from_numpy(self._get_fake_poses())

            data["fps"] = self.fps  # 16
            data["pose"] = pose.float()  # Shape: [88, J, 6]
            data["padding_mask"] = torch.zeros(1, self.video_size[0], self.video_size[1])
            data["image_size"] = torch.tensor(
                [self.video_size[0], self.video_size[1], self.video_size[0], self.video_size[1]])

            data["t5_text_embeddings"] = torch.zeros(
                CosmosTextEncoderConfig.NUM_TOKENS, CosmosTextEncoderConfig.EMBED_DIM, dtype=torch.bfloat16
            )  # Dummy embedding
            data["t5_text_mask"] = torch.ones(CosmosTextEncoderConfig.NUM_TOKENS, dtype=torch.int64)

            data["num_frames"] = self.num_frames

            # NOTE: __key__ is used to uniquely identify the sample, required for callback functions
            data["__key__"] = sample_path

            return data
        except Exception:
            # warnings.warn("FULL TRACEBACK:")  # noqa: B028
            # warnings.warn(traceback.format_exc())  # noqa: B028
            #warnings.warn(f"Error loading sample {self.samples[index]}, skipping to a new random sample.")
            #self.wrong_number += 1
            #print(self.wrong_number)

            return self[np.random.randint(len(self.samples))]


if __name__ == "__main__":
    # Example usage
    dataset = PoseConditionedDataset(
        video_root="DATA_PATH",
        split="val",
        num_frames=45,
        num_conditioning_frames=13,
        pose_process_method='fullbody',
        video_size=(480, 480),  # 1:1 aspect ratio
        load_poses=True,
        use_precomputed_latents=True,
        latents_fps=16,
        mode="train"
    )

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through the DataLoader
    for batch in dataloader:
        print(batch["video"].shape)
        print(batch["pose"].shape)
        break