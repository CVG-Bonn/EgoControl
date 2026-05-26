from pathlib import Path

import cv2
import numpy as np
import torch

from cosmos_predict2.data.pose_conditioned.dataset_utils import FULL_BODY_JOINTS

_POSE_YAW_DEG = 20.0
_POSE_PITCH_DEG = 10.0

_FULL_BODY_SKELETON = [
    ("Pelvis", "L5"),
    ("L5", "L3"),
    ("L3", "T12"),
    ("T12", "T8"),
    ("T8", "Neck"),
    ("Neck", "Head"),
    ("T8", "R_Shoulder"),
    ("R_Shoulder", "R_UpperArm"),
    ("R_UpperArm", "R_Forearm"),
    ("R_Forearm", "R_Hand"),
    ("T8", "L_Shoulder"),
    ("L_Shoulder", "L_UpperArm"),
    ("L_UpperArm", "L_Forearm"),
    ("L_Forearm", "L_Hand"),
    ("Pelvis", "R_UpperLeg"),
    ("R_UpperLeg", "R_LowerLeg"),
    ("R_LowerLeg", "R_Foot"),
    ("R_Foot", "R_Toe"),
    ("Pelvis", "L_UpperLeg"),
    ("L_UpperLeg", "L_LowerLeg"),
    ("L_LowerLeg", "L_Foot"),
    ("L_Foot", "L_Toe"),
]
_FULL_BODY_SKELETON_EDGES = [
    (FULL_BODY_JOINTS.index(start), FULL_BODY_JOINTS.index(end))
    for start, end in _FULL_BODY_SKELETON
]


def load_raw_pose_translations(pose_path: str) -> np.ndarray:
    pose_data = torch.load(pose_path, map_location="cpu", weights_only=False)["poses"]
    translations = pose_data["global_translations"]
    if isinstance(translations, torch.Tensor):
        translations = translations.cpu().numpy()
    if translations.ndim == 4 and translations.shape[2] == 1:
        translations = translations.squeeze(2)
    return translations


def select_future_translations(translations: np.ndarray, future_frames: int) -> np.ndarray:
    if future_frames <= 0:
        return translations[:0]
    if translations.shape[0] >= future_frames:
        return translations[-future_frames:]
    pad_frames = future_frames - translations.shape[0]
    pad = np.repeat(translations[-1:], pad_frames, axis=0)
    return np.concatenate([translations, pad], axis=0)


def project_pose_xy(translations: np.ndarray, width: int, height: int, margin: float = 0.05) -> np.ndarray:
    yaw = np.deg2rad(-_POSE_YAW_DEG)
    pitch = np.deg2rad(_POSE_PITCH_DEG)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)

    rot_yaw = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rot_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, -sin_pitch],
            [0.0, sin_pitch, cos_pitch],
        ],
        dtype=np.float32,
    )
    rot = rot_pitch @ rot_yaw
    rotated = translations.astype(np.float32).reshape(-1, 3) @ rot.T
    rotated = rotated.reshape(translations.shape)
    xz = rotated[..., [0, 2]]

    min_xy = xz.min(axis=(0, 1))
    max_xy = xz.max(axis=(0, 1))
    span = max_xy - min_xy
    span[span == 0] = 1.0
    scale = min((width * (1 - 2 * margin)) / span[0], (height * (1 - 2 * margin)) / span[1])
    center = (min_xy + max_xy) / 2.0
    centered = xz - center
    coords = np.empty_like(centered)
    coords[..., 0] = centered[..., 0] * scale + width / 2.0
    coords[..., 1] = -centered[..., 1] * scale + height / 2.0
    return coords


def render_pose_frames(
    translations: np.ndarray,
    width: int,
    height: int,
    edges: list[tuple[int, int]],
    joint_radius: int = 3,
    bone_thickness: int = 2,
) -> np.ndarray:
    coords = project_pose_xy(translations, width, height)
    frames = np.zeros((coords.shape[0], height, width, 3), dtype=np.uint8)
    for t in range(coords.shape[0]):
        frame = frames[t]
        joints = coords[t]
        for start, end in edges:
            pt1 = (int(round(joints[start, 0])), int(round(joints[start, 1])))
            pt2 = (int(round(joints[end, 0])), int(round(joints[end, 1])))
            cv2.line(frame, pt1, pt2, (0, 255, 0), bone_thickness)
        for joint in joints:
            pt = (int(round(joint[0])), int(round(joint[1])))
            cv2.circle(frame, pt, joint_radius, (255, 255, 255), -1)
    return frames


def build_pose_video(
    pose_path: str,
    num_frames: int,
    cond_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    translations = load_raw_pose_translations(pose_path)
    future_frames = max(0, num_frames - cond_frames)
    translations = select_future_translations(translations, future_frames)
    if future_frames > 0:
        pose_frames = render_pose_frames(translations, width, height, _FULL_BODY_SKELETON_EDGES)
    else:
        pose_frames = np.zeros((0, height, width, 3), dtype=np.uint8)
    if cond_frames > 0:
        blank = np.zeros((cond_frames, height, width, 3), dtype=np.uint8)
        pose_frames = np.concatenate([blank, pose_frames], axis=0)
    if pose_frames.shape[0] < num_frames:
        if pose_frames.shape[0] == 0:
            pad = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
            pose_frames = pad
        else:
            pad_count = num_frames - pose_frames.shape[0]
            pad = np.repeat(pose_frames[-1:], pad_count, axis=0)
            pose_frames = np.concatenate([pose_frames, pad], axis=0)
    elif pose_frames.shape[0] > num_frames:
        pose_frames = pose_frames[:num_frames]
    pose_tensor = torch.from_numpy(pose_frames).permute(3, 0, 1, 2).float().div(255.0)
    return pose_tensor * 2 - 1


def make_viz_output_path(output_path: str) -> str:
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_viz{path.suffix}"))
