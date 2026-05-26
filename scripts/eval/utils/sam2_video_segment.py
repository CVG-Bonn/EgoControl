"""
This script runs SAM2 video tracking segmentation, takes as input the points defined in a XML file.
You will need to setup SAM2 dependencies, configs and checkpoints.
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET

from sam2.build_sam import build_sam2_video_predictor


# -----------------------------
# Annotation loaders
# -----------------------------
def _ensure_labels(labels: Optional[np.ndarray], n: int) -> np.ndarray:
    if labels is None:
        return np.ones((n,), dtype=np.int32)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError("labels must be 1D of same length as points")
    return labels.astype(np.int32)


# parse a CVAT "x1,y1;x2,y2;..." string to Nx2 float32 points
def _parse_cvat_points_attr(points_str: str) -> np.ndarray:
    pts: List[List[float]] = []
    for token in (points_str or "").strip().split(";"):
        token = token.strip()
        if not token:
            continue
        xy = token.split(",")
        if len(xy) != 2:
            continue
        try:
            x = float(xy[0].strip())
            y = float(xy[1].strip())
        except ValueError:
            continue
        pts.append([x, y])
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


# load a single XML file mapping image-name stems -> (points_list, labels_list)
def load_annotations_xml(path: Path) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Parse CVAT-like XML with:
      <annotations>
        ...
        <image name="video_stem.png" ...>
          <points points="x1,y1;x2,y2" .../>
          <points points="x1,y1" .../>
          ...
        </image>
        ...
      </annotations>
    Returns:
      dict[stem] = (points_list, labels_list)
        - points_list: list of per-object Nx2 float32 arrays
        - labels_list: list of per-object N int32 arrays (all ones)
    """
    tree = ET.parse(path)
    root = tree.getroot()

    ann_map: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    for img in root.findall(".//image"):
        img_name = img.get("name")
        if not img_name:
            continue
        stem = Path(img_name).stem  # drop .png, should match video stem

        obj_points_list: List[np.ndarray] = []
        obj_labels_list: List[np.ndarray] = []

        for pe in img.findall("./points"):
            pts_attr = pe.get("points", "")
            pts = _parse_cvat_points_attr(pts_attr)
            if pts.shape[0] == 0:
                continue
            lbls = _ensure_labels(None, len(pts))
            obj_points_list.append(pts)
            obj_labels_list.append(lbls)

        if obj_points_list:
            if stem in ann_map:
                # merge if multiple <image> with same name appear
                prev_pts, prev_lbls = ann_map[stem]
                prev_pts.extend(obj_points_list)
                prev_lbls.extend(obj_labels_list)
            else:
                ann_map[stem] = (obj_points_list, obj_labels_list)

    if not ann_map:
        raise ValueError(f"No valid points found in XML {path}")
    return ann_map

# -----------------------------
# Tracking pipeline
# -----------------------------
def track_video_with_points(
    predictor,
    video_path: Path,
    points_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    seed_frame_idx: int = 0,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = False,
) -> np.ndarray:
    """
    Run SAM 2 tracking on a video given per-object seed point prompts on a frame.

    Returns a boolean array of shape (T, O, H, W) where T is frames, O objects.
    """
    # Initialize predictor state and get video metadata
    state = predictor.init_state(
        str(video_path),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=False,
    )

    num_frames = int(state["num_frames"])  # total frames
    H = int(state["video_height"])  # original video height
    W = int(state["video_width"])   # original video width
    num_objects = len(points_list)

    # Add objects' prompts on the seed frame
    for obj_id, (pts, lbls) in enumerate(zip(points_list, labels_list), start=1):
        if pts.size == 0:
            continue
        # SAM2 expects pixel coords when normalize_coords=True (internally divides by W,H)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=seed_frame_idx,
            obj_id=obj_id,
            points=pts.tolist(),
            labels=lbls.tolist(),
            clear_old_points=True,
            normalize_coords=True,
            box=None,
        )

    # Preallocate output array (bool)
    masks = np.zeros((num_frames, num_objects, H, W), dtype=np.bool_)

    # Track across all frames (forward from seed)
    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
        # video_res_masks: Tensor [O, 1, H, W] logits
        with torch.no_grad():
            bin_masks = (video_res_masks > 0).squeeze(1).to("cpu").numpy().astype(np.bool_)
        masks[frame_idx, : len(obj_ids)] = bin_masks

    return masks


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch track SAM 2 from point prompts on frame 0.")
    p.add_argument("--videos_dir", required=True, type=str, help="Directory containing input videos (.mp4)")
    # CHANGED: make dir optional, add XML as required
    p.add_argument("--annotations_xml", required=True, type=str, help="CVAT-like XML with per-image 'points' prompts; image name stem must match video stem")
    p.add_argument("--output_dir", required=True, type=str, help="Directory to store output NPZ masks")
    p.add_argument("--video_ext", default=".mp4", type=str, help="Video file extension to look for")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # Model settings
    p.add_argument("--model_config", default="configs/sam2.1/sam2.1_hiera_s.yaml", type=str, help="Path to SAM 2 config YAML")
    p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_small.pt", type=str, help="Path to SAM 2 checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Compute device")
    p.add_argument("--offload_video_to_cpu", action="store_true", help="Offload loaded video frames to CPU to save VRAM")
    p.add_argument("--offload_state_to_cpu", action="store_true", help="Offload model state between frames to CPU to save VRAM (slower)")
    return p.parse_args()


def main():
    args = parse_args()

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build predictor
    predictor = build_sam2_video_predictor(
        config_file=args.model_config,
        ckpt_path=args.checkpoint,
        device=args.device,
        mode="eval",
        apply_postprocessing=True,
        vos_optimized=False,
    )

    # Build map from stem -> video path
    video_map: Dict[str, Path] = {}
    for p in videos_dir.glob(f"*{args.video_ext}"):
        video_map[p.stem] = p

    # Prefer XML-driven annotations
    ann_xml_path = Path(args.annotations_xml)
    try:
        stem_to_ann = load_annotations_xml(ann_xml_path)
    except Exception as e:
        print(f"[ERROR] Failed to load XML annotations: {e}")
        return

    # Iterate XML images and process matching videos; seed at frame 0
    for stem in tqdm(sorted(stem_to_ann.keys()), desc="Processing XML annotations"):
        video_path = video_map.get(stem)
        if video_path is None:
            print(f"[WARN] No matching video for image stem '{stem}'; expected {stem}{args.video_ext}")
            continue

        out_path = out_dir / f"{stem}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] Output exists: {out_path}")
            continue

        points_list, labels_list = stem_to_ann[stem]

        try:
            masks = track_video_with_points(
                predictor=predictor,
                video_path=video_path,
                points_list=points_list,
                labels_list=labels_list,
                seed_frame_idx=0,  # first frame of each video
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )
        except Exception as e:
            print(f"[ERROR] Tracking failed for {video_path.name}: {e}")
            continue

        try:
            np.savez_compressed(out_path, masks=masks)
            print(f"[OK] Saved {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save {out_path}: {e}")


if __name__ == "__main__":
    main()
