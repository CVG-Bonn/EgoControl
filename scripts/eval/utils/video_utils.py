from pathlib import Path
from typing import List, Tuple

try:
    from .video_evaluator import log
except ImportError:
    from video_evaluator import log


def find_video_pairs(gen_folder: str, gt_folder: str, num_videos: int) -> List[Tuple[str, str]]:
    """
    Find matching video pairs between generated and ground truth folders.
    Deterministically orders pairs by filename stem (alphabetical).
    """
    gen_path = Path(gen_folder)
    gt_path = Path(gt_folder)

    if not gen_path.exists():
        raise FileNotFoundError(f"Generated videos folder not found: {gen_folder}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth videos folder not found: {gt_folder}")

    # Get all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    gen_files = {f.stem: f for f in gen_path.iterdir()
                 if f.suffix.lower() in video_extensions}
    gt_files = {f.stem: f for f in gt_path.iterdir()
                if f.suffix.lower() in video_extensions}

    # Find common names
    common_names = set(gen_files.keys()) & set(gt_files.keys())

    if not common_names:
        log.warning("No matching video pairs found between folders")
        # Try alternative matching strategies
        log.info("Attempting alternative matching strategies...")

        # Strategy 1: Remove common prefixes/suffixes
        gen_stems = set()
        for name in gen_files.keys():
            # Remove common prefixes like "generated_", "gen_", etc.
            clean_name = name
            for prefix in ["generated_", "gen_", "output_"]:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
            # Remove common suffixes like "_generated", "_gen", etc.
            for suffix in ["_generated", "_gen", "_output"]:
                if clean_name.endswith(suffix):
                    clean_name = clean_name[:-len(suffix)]
            gen_stems.add((name, clean_name))

        gt_stems = set()
        for name in gt_files.keys():
            # Remove common prefixes like "gt_", "truth_", etc.
            clean_name = name
            for prefix in ["gt_", "truth_", "target_"]:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
            # Remove common suffixes
            for suffix in ["_gt", "_truth", "_target"]:
                if clean_name.endswith(suffix):
                    clean_name = clean_name[:-len(suffix)]
            gt_stems.add((name, clean_name))

        # Find matches with cleaned names
        for gen_name, gen_clean in gen_stems:
            for gt_name, gt_clean in gt_stems:
                if gen_clean == gt_clean:
                    common_names.add(gen_name)
                    # Map the cleaned name back
                    gen_files[gen_name] = gen_files[gen_name]
                    gt_files[gen_name] = gt_files[gt_name]

    # Deterministic ordering by stem
    names_sorted = sorted(common_names)
    pairs = [(str(gen_files[name]), str(gt_files[name])) for name in names_sorted]

    # Respect num_videos: <= 0 means all
    if num_videos is not None and num_videos > 0:
        pairs = pairs[:num_videos]

    log.info(f"Found {len(pairs)} matching video pairs (sorted)")
    return pairs
