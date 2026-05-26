"""
Video Evaluation Script for Computing DreamSim, SSIM, LPIPS, FID and FVD metrics
between generated videos and ground truth videos.
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from scripts.eval.utils.video_evaluator import VideoEvaluator, log
    from scripts.eval.utils.video_utils import find_video_pairs
except ImportError:
    from utils.video_evaluator import VideoEvaluator, log
    from utils.video_utils import find_video_pairs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated videos against ground truth using SSIM, LPIPS, DreamSim and FID metrics"
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=-1,
        help="Number of video pairs to evaluate (default: all available pairs)"
    )
    parser.add_argument(
        "--gen_folder",
        type=str,
        required=True,
        help="Path to folder containing generated videos"
    )

    parser.add_argument(
        "--gt_folder",
        type=str,
        required=True,
        help="Path to folder containing ground truth videos"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results (JSON format)"
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to evaluate per video (default: all frames)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing frames (default: 8)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation (default: cuda)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="ssim,lpips,dreamsim,fid",
        help="Comma-separated list of metrics to compute: ssim, lpips, dreamsim, fid"
    )
    # New: conditioning frames
    parser.add_argument(
        "--cond_frames",
        type=int,
        default=13,
        help="Number of initial conditioning frames to discard for frame-wise metrics (SSIM/LPIPS/DreamSim/FID)."
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Target resolution for evaluation in HxW format (e.g., 256x256). If not set, uses min height/width of the pair."
    )
    parser.add_argument(
        "--frame_range",
        type=str,
        default="0:-1",
        help="Frame range for per frame metrics evaluation, start:end format (0-indexed, end exclusive). If not set, uses all frames."
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Configure logging
    if args.verbose:
        try:
            log.remove()
            log.add(lambda msg: print(msg, end=""), level="DEBUG")
        except:
            # Fallback for basic logging
            import logging
            logging.getLogger().setLevel(logging.DEBUG)

    log.info(f"Starting video evaluation at {args.resolution}...")
    if args.resolution is not None:
        resolution = args.resolution.split('x')
        if len(resolution) != 2 or not all(r.isdigit() for r in resolution):
            log.error(f"Invalid resolution format: {args.resolution}. Use HxW format, e.g., 256x256.")
            return
        resolution = (int(resolution[0]), int(resolution[1]))
    else:
        resolution = None

    log.info(f"Generated videos folder: {args.gen_folder}")
    log.info(f"Ground truth videos folder: {args.gt_folder}")

    # Parse and validate metrics
    allowed_metrics = {"ssim", "lpips", "dreamsim", "fid", "fvd"}
    metrics_to_compute = {m.strip().lower() for m in args.metrics.split(",") if m.strip()}
    metrics_to_compute = metrics_to_compute & allowed_metrics
    if not metrics_to_compute:
        log.error("No valid metrics selected. Choose from: ssim, lpips, dreamsim, fid, fvd")
        return

    # Check for FVD mutual exclusivity
    if "fvd" in metrics_to_compute and len(metrics_to_compute) > 1:
        log.error("FVD computation must be done alone. Cannot combine with other metrics (ssim, lpips, dreamsim, fid).")
        return

    log.info(f"Requested metrics: {sorted(metrics_to_compute)}")

    try:
        evaluator = VideoEvaluator(device=args.device, batch_size=args.batch_size)
        video_pairs = find_video_pairs(args.gen_folder, args.gt_folder, args.num_videos)
        if not video_pairs:
            log.error("No matching video pairs found. Please check folder contents and naming.")
            return

        # Handle FVD separately (dataset-level metric)
        if "fvd" in metrics_to_compute:
            log.info("Computing FVD (dataset-level metric)...")
            gen_paths = [pair[0] for pair in video_pairs]
            gt_paths = [pair[1] for pair in video_pairs]

            fvd_score = evaluator.compute_fvd(gen_paths, gt_paths)

            # Prepare FVD-only results
            new_results = {
                'summary': {
                    'total_pairs': len(video_pairs),
                    'evaluated_pairs': len(video_pairs),
                    'aggregate_metrics': {},
                    'dataset_metrics': {
                        'fvd': fvd_score
                    }
                },
                'individual_results': {
                    f"pair_{i+1}_{Path(gen_path).stem}": {
                        'generated_video': gen_path,
                        'ground_truth_video': gt_path,
                        'metrics': {}
                    } for i, (gen_path, gt_path) in enumerate(video_pairs)
                }
            }

            # Merge with existing results if file exists
            merged_results = new_results
            if os.path.exists(args.output_path):
                try:
                    with open(args.output_path, 'r') as f:
                        existing = json.load(f)
                except Exception as e:
                    log.warning(f"Failed to read existing output file, will overwrite: {e}")
                    existing = None

                if isinstance(existing, dict):
                    merged_individual = existing.get('individual_results', {})
                    # Update or add individual results
                    for pair_name, entry in new_results['individual_results'].items():
                        if pair_name not in merged_individual:
                            merged_individual[pair_name] = entry
                        else:
                            # Preserve existing pair data and metrics
                            merged_individual[pair_name]['generated_video'] = entry['generated_video']
                            merged_individual[pair_name]['ground_truth_video'] = entry['ground_truth_video']
                            # PRESERVE existing metrics - don't reset to empty dict
                            if 'metrics' not in merged_individual[pair_name]:
                                merged_individual[pair_name]['metrics'] = {}
                            # Keep all existing metrics since FVD doesn't add per-pair metrics

                    # Preserve existing aggregate metrics completely
                    existing_aggregate = existing.get('summary', {}).get('aggregate_metrics', {})

                    # Dataset metrics: update FVD, keep others from existing
                    existing_dataset = existing.get('summary', {}).get('dataset_metrics', {})
                    merged_dataset = dict(existing_dataset)
                    merged_dataset['fvd'] = fvd_score

                    merged_results = {
                        'summary': {
                            'total_pairs': len(merged_individual),
                            'evaluated_pairs': len(merged_individual),
                            'aggregate_metrics': existing_aggregate,  # Keep existing aggregate metrics
                            'dataset_metrics': merged_dataset
                        },
                        'individual_results': merged_individual
                    }

            # Save FVD results
            with open(args.output_path, 'w') as f:
                json.dump(merged_results, f, indent=2)

            # Print FVD summary
            log.info("\n" + "=" * 60)
            log.info("FVD EVALUATION SUMMARY")
            log.info("=" * 60)
            log.info(f"Total video pairs: {len(video_pairs)}")
            log.info(f"FVD Score: {fvd_score:.4f}")
            log.info(f"\nResults saved to: {args.output_path}")
            return

        # Handle other metrics (existing code path)
        # Aggregate per-pair metrics (exclude FID which is dataset-level)
        all_results = {}
        per_pair_metric_names = [m for m in ("ssim", "lpips", "dreamsim") if m in metrics_to_compute]
        aggregated_metrics = {m: [] for m in per_pair_metric_names}

        start_frame, end_frame = args.frame_range.split(':')
        start_frame = int(start_frame)
        end_frame = int(end_frame)

        for i, (gen_path, gt_path) in enumerate(video_pairs):
            log.info(f"\nEvaluating pair {i + 1}/{len(video_pairs)}")
            try:
                results = evaluator.evaluate_videos(
                    gen_path, gt_path, args.max_frames, metrics=metrics_to_compute, cond_frames=args.cond_frames,
                    resolution=resolution,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
                pair_name = f"pair_{i + 1}_{Path(gen_path).stem}"
                all_results[pair_name] = {
                    'generated_video': gen_path,
                    'ground_truth_video': gt_path,
                    'metrics': results
                }

                # Aggregate per-pair metrics
                for metric in per_pair_metric_names:
                    if results.get(metric) is not None:
                        aggregated_metrics[metric].append(results[metric])

            except Exception as e:
                log.error(f"Error evaluating pair {gen_path} vs {gt_path}: {e}")
                continue

        # Compute aggregate statistics for per-pair metrics
        aggregate_stats = {}
        for metric, values in aggregated_metrics.items():
            if values:
                aggregate_stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            else:
                aggregate_stats[metric] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}

        # Compute dataset-level FID once if requested
        dataset_fid = None

        if "fid" in metrics_to_compute:
            log.info("Computing dataset-level FID...")
            try:
                dataset_fid = evaluator.fid_fn.compute().item()
            except Exception as e:
                log.error(f"Failed to compute dataset-level FID: {e}")
                dataset_fid = 0.0

        # Prepare this-run results
        new_results = {
            'summary': {
                'total_pairs': len(video_pairs),
                'evaluated_pairs': len(all_results),
                'aggregate_metrics': aggregate_stats,
                'dataset_metrics': {
                    # Only set those computed in this run; others will be merged from file if present
                    **({'fid': dataset_fid} if dataset_fid is not None else {}),
                }
            },
            'individual_results': all_results
        }

        # If output exists, merge and update
        merged_results = new_results
        if os.path.exists(args.output_path):
            try:
                with open(args.output_path, 'r') as f:
                    existing = json.load(f)
            except Exception as e:
                log.warning(f"Failed to read existing output file, will overwrite: {e}")
                existing = None

            if isinstance(existing, dict):
                merged_individual = existing.get('individual_results', {})
                # Update or add per-pair metrics only for requested ones
                for pair_name, entry in new_results['individual_results'].items():
                    if pair_name not in merged_individual:
                        merged_individual[pair_name] = entry
                    else:
                        merged_individual[pair_name]['generated_video'] = entry['generated_video']
                        merged_individual[pair_name]['ground_truth_video'] = entry['ground_truth_video']
                        merged_metrics = merged_individual[pair_name].get('metrics', {})
                        for k, v in entry['metrics'].items():
                            merged_metrics[k] = v
                        merged_individual[pair_name]['metrics'] = merged_metrics

                # Recompute aggregate stats over merged individuals for available per-pair metrics
                recomputed_agg = {}
                for metric_name in {"ssim", "lpips", "dreamsim"}:
                    vals = []
                    for pair in merged_individual.values():
                        m = pair.get('metrics', {}).get(metric_name, None)
                        if isinstance(m, (int, float)):
                            vals.append(m)
                    if vals:
                        recomputed_agg[metric_name] = {
                            'mean': float(np.mean(vals)),
                            'std': float(np.std(vals)),
                            'min': float(np.min(vals)),
                            'max': float(np.max(vals)),
                            'count': len(vals)
                        }
                    else:
                        recomputed_agg[metric_name] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}

                # Dataset metrics: update only those computed now; keep others from existing
                existing_dataset = existing.get('summary', {}).get('dataset_metrics', {})
                merged_dataset = dict(existing_dataset)
                for k, v in new_results['summary']['dataset_metrics'].items():
                    merged_dataset[k] = v

                merged_results = {
                    'summary': {
                        'total_pairs': len(merged_individual),
                        'evaluated_pairs': len(merged_individual),
                        'aggregate_metrics': recomputed_agg,
                        'dataset_metrics': merged_dataset
                    },
                    'individual_results': merged_individual
                }
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        return

    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(merged_results, f, indent=2)

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    log.info(f"Total video pairs: {merged_results['summary']['total_pairs']}")
    log.info(f"Successfully evaluated: {merged_results['summary']['evaluated_pairs']}")
    for metric, stats in merged_results['summary']['aggregate_metrics'].items():
        if stats['count'] > 0:
            log.info(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                     f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
        else:
            log.info(f"{metric.upper()}: Not computed")
    dm = merged_results['summary'].get('dataset_metrics', {})
    if 'fid' in dm:
        log.info(f"DATASET FID: {dm['fid']:.4f}")
    log.info(f"\nDetailed results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
