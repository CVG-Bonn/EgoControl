"""
This script evaluate the mIoU between segmentation masks of the body arms computed by Sam2 on the ground-truth and on the generated frames.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

@dataclass
class VideoMetrics:
    file: str
    n_frames: int
    height: int
    width: int
    frame_iou_union: List[float]  # length == n_frames; NaN where both GT and pred empty
    video_iou_union_over_time: float
    binary_result: float


def find_mask_array(arrs: Dict[str, np.ndarray], preferred_key: Optional[str]) -> np.ndarray:
    keys = list(arrs.keys())
    if preferred_key and preferred_key in arrs:
        return arrs[preferred_key]
    for k in ("masks", "mask", "labels", "arr_0"):
        if k in arrs:
            return arrs[k]
    # fallback: first array-like entry with ndim >= 3 and last two dims look like image dims
    for k in keys:
        a = arrs[k]
        if isinstance(a, np.ndarray) and a.ndim >= 3 and a.shape[-1] >= 4 and a.shape[-2] >= 4:
            return a
    raise KeyError("Could not find a mask array. Tried keys: 'masks','mask','labels','arr_0'.")


def load_masks(npz_path: str, key: Optional[str] = None, threshold: Optional[float] = None) -> np.ndarray:
    """
    Returns a boolean array with shape (T, H, W).
    Multi-instance arrays (T, N, H, W) are collapsed to union.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        arr = find_mask_array(data, key)
        a = np.array(arr)

    if a.ndim < 3:
        raise ValueError(f"Mask array must have at least 3 dims, got shape {a.shape} in {npz_path}")

    # Convert to boolean
    if a.dtype == np.bool_:
        b = a
    else:
        if threshold is None:
            threshold = 0.0
        b = a > threshold

    # Collapse multi-instance axis to union -> (T, H, W)
    if b.ndim == 4:
        b = np.any(b, axis=1)

    return b.astype(bool, copy=False)


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    if union == 0:
        return 1.0  # both empty
    return float(inter / union)


def evaluate_video(gt: np.ndarray, pr: np.ndarray, file: str) -> VideoMetrics:
    """gt and pr are both (T, H, W) bool arrays."""
    if gt.shape != pr.shape:
        raise ValueError(f"Shape mismatch for {file}: gt {gt.shape} vs pred {pr.shape}.")

    T, H, W = gt.shape

    frame_iou_union: List[float] = []
    binary_result = 0

    for t in range(T):
        gt_empty = not gt[t].any()
        pr_empty = not pr[t].any()

        if gt_empty and pr_empty:
            frame_iou_union.append(float('nan'))
        else:
            frame_iou_union.append(iou_binary(gt[t], pr[t]))

        if (gt[t].any() and pr[t].any()) or (gt_empty and pr_empty):
            binary_result += 1

    video_iou_union = iou_binary(gt, pr)

    return VideoMetrics(
        file=file,
        n_frames=T,
        height=H,
        width=W,
        frame_iou_union=[float(x) for x in frame_iou_union],
        video_iou_union_over_time=video_iou_union,
        binary_result=binary_result / T if T > 0 else 1.0,
    )


def evaluate_folders(
    gt_folder: str,
    pred_folder: str,
    key: Optional[str],
    threshold: Optional[float],
    strict: bool,
) -> Tuple[List[VideoMetrics], List[str]]:
    files = sorted(f for f in os.listdir(gt_folder) if f.endswith('.npz'))
    results: List[VideoMetrics] = []
    warnings: List[str] = []

    for f in files:
        gt_path = os.path.join(gt_folder, f)
        pr_path = os.path.join(pred_folder, f)
        if not os.path.exists(pr_path):
            msg = f"Missing prediction for {f}; skipping."
            if strict:
                raise FileNotFoundError(msg)
            warnings.append(msg)
            continue
        try:
            gt_masks = load_masks(gt_path, key=key, threshold=threshold)[13:]
            pr_masks = load_masks(pr_path, key=key, threshold=threshold)[13:]
            if not gt_masks.any() and not pr_masks.any():
                continue
            results.append(evaluate_video(gt_masks, pr_masks, f))
        except Exception as e:
            if strict:
                raise
            warnings.append(f"Failed on {f}: {e}")

    return results, warnings


def summarize(results: Sequence[VideoMetrics]) -> Dict:
    if not results:
        return {
            "num_videos": 0,
            "mean_video_iou_union": float('nan'),
            "mean_binary_result": float('nan'),
            "per_frame_mean_iou_union": [],
        }

    mean_video_iou_union = float(np.mean([r.video_iou_union_over_time for r in results]))
    mean_binary_result = float(np.mean([r.binary_result for r in results]))

    # Per-frame nanmean across videos (only when all videos share the same frame count)
    T = results[0].n_frames
    if all(r.n_frames == T for r in results):
        arr = np.array([r.frame_iou_union for r in results], dtype=np.float64)  # (V, T)
        per_frame_mean_iou_union = [float(x) for x in np.nanmean(arr, axis=0)]
    else:
        per_frame_mean_iou_union = []

    return {
        "num_videos": len(results),
        "mean_video_iou_union": mean_video_iou_union,
        "mean_binary_result": mean_binary_result,
        "per_frame_mean_iou_union": per_frame_mean_iou_union,
    }


def format_table(results: Sequence[VideoMetrics]) -> str:
    headers = ["file", "T", "HxW", "IoU(video-union)", "binary"]
    rows = [
        [r.file, str(r.n_frames), f"{r.height}x{r.width}",
         f"{r.video_iou_union_over_time:.4f}", f"{r.binary_result:.4f}"]
        for r in results
    ]
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(row):
        return "  ".join(s.ljust(w) for s, w in zip(row, widths))

    lines = [fmt_row(headers), fmt_row(["-" * len(h) for h in headers])]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate video segmentation IoU given GT and prediction .npz folders.")
    p.add_argument('gt_folder', nargs='?', help='Folder with ground-truth .npz files')
    p.add_argument('pred_folder', nargs='?', help='Folder with prediction .npz files (matching filenames)')
    p.add_argument('--key', default=None, help="Array key inside .npz (default: auto-detect)")
    p.add_argument('--threshold', type=float, default=None, help='Threshold for binarizing non-bool masks (default: >0)')
    p.add_argument('--strict', action='store_true', help='Fail on missing/malformed files (default: skip with warning)')
    p.add_argument('--out_json', default=None, help='Optional path to write per-video metrics and summary as JSON')

    args = p.parse_args(argv)

    if not args.gt_folder or not args.pred_folder:
        print("Error: gt_folder and pred_folder are required.", file=sys.stderr)
        p.print_usage(sys.stderr)
        return 2

    if not os.path.isdir(args.gt_folder):
        print(f"gt_folder not found or not a directory: {args.gt_folder}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.pred_folder):
        print(f"pred_folder not found or not a directory: {args.pred_folder}", file=sys.stderr)
        return 2

    results, warnings = evaluate_folders(
        args.gt_folder, args.pred_folder,
        key=args.key, threshold=args.threshold, strict=args.strict,
    )

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
        print()

    if not results:
        print("No results to report.")
        return 1

    print(format_table(results))
    summary = summarize(results)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    if args.out_json:
        payload = {"videos": [asdict(r) for r in results], "summary": summary}
        with open(args.out_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON to {args.out_json}")

    if summary["per_frame_mean_iou_union"]:
        plt.figure()
        plt.plot(summary["per_frame_mean_iou_union"])
        plt.title("Per-frame Mean IoU (Union)")
        plt.xlabel("Frame Index")
        plt.ylabel("Mean IoU")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()

    print(np.mean(summary["per_frame_mean_iou_union"]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())