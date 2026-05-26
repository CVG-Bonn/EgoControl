from pathlib import Path
import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from cosmos_predict2.data.action_conditioned.dataset_utils import euler2rotm, rotm2euler

FULL_BODY_JOINTS = ['Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head', 'R_Shoulder', 'R_UpperArm', 'R_Forearm',
              'R_Hand', 'L_Shoulder', 'L_UpperArm', 'L_Forearm', 'L_Hand', 'R_UpperLeg', 'R_LowerLeg', 'R_Foot',
              'R_Toe', 'L_UpperLeg', 'L_LowerLeg', 'L_Foot', 'L_Toe']


def get_latent_num_frames(num_pixel_frames: int) -> int:
    return 1 + (num_pixel_frames - 1) // 4


def get_pixel_num_frames(num_latent_frames: int) -> int:
    return (num_latent_frames - 1) * 4 + 1


def get_velocity(translations, rotations_euler, num_future_frames, accum=False):
    """
    Compute the velocity (delta) of the poses given translations and rotations in Euler angles of a single joint.
    The velocity is computed as the relative transformation between consecutive frames by default.
    :param translations: Selected joint translations of shape (num_frames, 1, 3)
    :param rotations_euler: Selected joint rotations in Euler angles of shape (num_frames, 1, 3)
    :param num_future_frames:
    :return: Pose velocity of shape (num_future_frames, 1, 6) where 6 = 3 for translation and 3 for rotation
    """
    translations = translations[-(num_future_frames + 1):]
    rotations_euler = rotations_euler[-(num_future_frames + 1):]

    velocity = np.zeros((translations.shape[0] - 1, 1, 6))  # 3 for translation and 3 for rotation

    if accum:
        first_translation = translations[0]
        first_rotation = rotations_euler[0]
        first_rotmatrix = euler2rotm(first_rotation)  # 0 is because we are only using head

        for i in range(1, translations.shape[0]):
            current_translation = translations[i]
            current_rotation = rotations_euler[i]
            current_rotmatrix = euler2rotm(current_rotation)

            rel_translation = np.dot(first_rotmatrix.T, (current_translation - first_translation))
            rel_rotmatrix = first_rotmatrix.T @ current_rotmatrix
            rel_rotation = rotm2euler(rel_rotmatrix)

            velocity[i - 1, 0, :3] = rel_translation  # 0 is because we are only using head
            velocity[i - 1, 0, 3:] = rel_rotation
    else:
        for i in range(1, translations.shape[0]):
            prev_translation = translations[i - 1]
            prev_rotation = rotations_euler[i - 1]
            prev_rotmatrix = euler2rotm(prev_rotation)

            current_translation = translations[i]
            current_rotation = rotations_euler[i]
            current_rotmatrix = euler2rotm(current_rotation)

            rel_translation = np.dot(prev_rotmatrix.T, (current_translation - prev_translation))
            rel_rotmatrix = prev_rotmatrix.T @ current_rotmatrix
            rel_rotation = rotm2euler(rel_rotmatrix)

            velocity[i - 1, 0, :3] = rel_translation  # 0 is because we are only using head
            velocity[i - 1, 0, 3:] = rel_rotation

    return velocity


def process_body_poses(poses, num_future_frames, root_joint='Pelvis', relative_joints=None, use_head_vel=True):
    assert relative_joints is not None, "relative_joints must be provided"

    # Head is always included
    if use_head_vel:
        head_vel = pose_single_joint(poses, num_future_frames, joint='Head')
    # Root joint velocity
    root_vel = pose_single_joint(poses, num_future_frames, joint=root_joint)

    global_translations = poses["global_translations"].squeeze(2)[-num_future_frames:]
    global_rotations_euler = poses["global_rotations_euler"][-num_future_frames:]

    # Exclude root and head from the relative joints
    excluded = [root_joint, 'Head'] if use_head_vel else [root_joint]
    relative_joints_idxs = [FULL_BODY_JOINTS.index(joint) for joint in relative_joints if joint not in excluded]

    root_idx = FULL_BODY_JOINTS.index(root_joint)

    # Compute the relative poses of the 'relative_joints' with respect to the root joint 'Pelvis'
    relative_poses = np.zeros((min(root_vel.shape[0], num_future_frames), len(relative_joints_idxs), 6))  # 3 for translation and 3 for rotation
    for i in range(root_vel.shape[0]):
        root_translation = global_translations[i, root_idx, :]
        root_rotation = global_rotations_euler[i, root_idx, :]
        root_rotmatrix = euler2rotm(root_rotation)

        for idx in relative_joints_idxs:
            rel_translation = np.dot(root_rotmatrix.T, (global_translations[i, idx] - root_translation))
            rel_rotmatrix = root_rotmatrix.T @ euler2rotm(global_rotations_euler[i, idx])
            rel_rotation = rotm2euler(rel_rotmatrix)

            relative_poses[i, relative_joints_idxs.index(idx), :3] = rel_translation
            relative_poses[i, relative_joints_idxs.index(idx), 3:] = rel_rotation

    if use_head_vel:
        final_poses = np.concatenate([head_vel, root_vel, relative_poses], axis=1)
    else:
        final_poses = np.concatenate([root_vel, relative_poses], axis=1)

    return final_poses


def pose_single_joint(poses, num_future_frames, joint='Head', accum=False):
    global_translations = poses["global_translations"].squeeze(2)
    global_rotations_euler = poses["global_rotations_euler"]

    # Find indices of selected joints in part_names
    selected_indices = FULL_BODY_JOINTS.index(joint)

    selected_translations = global_translations[:, selected_indices, :]
    selected_rotations_euler = global_rotations_euler[:, selected_indices, :]

    pose_cond = get_velocity(selected_translations, selected_rotations_euler, num_future_frames, accum=accum)

    return pose_cond


def process_poses(poses, num_future_frames, process_method='global'):
    if process_method == 'fullbody':
        return process_body_poses(poses, num_future_frames, relative_joints=FULL_BODY_JOINTS)
    elif process_method == 'head':
        return pose_single_joint(poses, num_future_frames)


def load_pose_chunk(pose_path, num_future_frames=88, process_method='head'):
    # Find the pose file corresponding to the chosen latent file
    if not os.path.exists(pose_path):
        # Try adjusting the index in the filename if not found
        pose_path_obj = Path(pose_path)
        file_name = pose_path_obj.name
        parts = file_name.split("_")
        last_part = int(parts[-1].split(".")[0]) - 1
        new_file_name = f"{'_'.join(parts[:-1])}_{last_part:06d}.pt"
        pose_path = pose_path_obj.parent / new_file_name
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

    # Load the pose data
    poses = torch.load(pose_path, map_location='cpu', weights_only=False)["poses"]

    poses = process_poses(poses, num_future_frames, process_method)

    num_frames = poses.shape[0]
    target_frames = num_future_frames  # + 1

    if num_frames < target_frames:
        pad_amount = target_frames - num_frames
        # create pad spec dynamically
        pad_width = [(0, 0)] * poses.ndim
        pad_width[0] = (0, pad_amount)  # pad along first axis only
        poses = np.pad(poses, pad_width, mode='constant', constant_values=0)

    return torch.from_numpy(poses).reshape(poses.shape[0], -1)