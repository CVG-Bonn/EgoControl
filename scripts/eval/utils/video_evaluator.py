import os
import warnings
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from torcheval.metrics import FrechetInceptionDistance

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision.io import read_video
    import numpy as np
    from skimage.metrics import structural_similarity as ssim

    # Try importing loguru, fallback to basic logging
    try:
        from loguru import logger as log
    except ImportError:
        import logging

        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)

    # Try importing LPIPS, will install if needed
    try:
        import lpips
    except ImportError:
        log.warning("LPIPS not installed. Please install with: pip install lpips")
        lpips = None
    # DreamSim optional import
    try:
        from dreamsim import dreamsim as dreamsim_fn
    except ImportError:
        log.warning("DreamSim not installed. Please install with: pip install dreamsim")
        dreamsim_fn = None

    # FVD import
    try:
        from scripts.eval.utils.fvd import compute_fvd
        fvd_available = True
    except ImportError as e:
        print(f"Required dependencies not available: {e}")
        log.warning("FVD module not available. FVD calculation will be disabled.")
        fvd_available = False
        compute_fvd = None
    torch_available = True

except ImportError as e:
    print(f"Required dependencies not available: {e}")
    print("Please install with: pip install torch torchvision scikit-image lpips scipy")
    torch_available = False


    # Create dummy classes for missing imports
    class DummyTorch:
        Tensor = object

        def cuda(self): return None


    torch = DummyTorch()


    class DummyLogger:
        def info(self, msg): print(f"INFO: {msg}")

        def warning(self, msg): print(f"WARNING: {msg}")

        def error(self, msg): print(f"ERROR: {msg}")

        def debug(self, msg): print(f"DEBUG: {msg}")

        def success(self, msg): print(f"SUCCESS: {msg}")


    log = DummyLogger()
    lpips = None
    scipy_available = False
    # Ensure dreamsim placeholder is present when torch missing
    dreamsim_fn = None


class VideoEvaluator:
    """Video evaluation class for computing various metrics between generated and ground truth videos."""

    def __init__(self, device: str = "cuda", batch_size: int = 8):
        """
        Initialize the VideoEvaluator.

        Args:
            device: Device to use for computation ('cuda' or 'cpu')
            batch_size: Batch size for processing frames
        """
        if not torch_available:
            raise ImportError(
                "Required dependencies not available. Please install torch, torchvision, scikit-image, lpips, scipy")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        # Initialize LPIPS model
        if lpips is not None:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            log.info("LPIPS model loaded successfully")
        else:
            self.lpips_model = None
            log.warning("LPIPS model not available")
        # Initialize DreamSim model
        if dreamsim_fn is not None:
            try:
                self.dreamsim_model, self.dreamsim_preprocess = dreamsim_fn(pretrained=True, device=self.device)
                self.dreamsim_model.eval()
                log.info("DreamSim model loaded successfully")
            except Exception as e:
                log.warning(f"Failed to initialize DreamSim: {e}")
                self.dreamsim_model, self.dreamsim_preprocess = None, None
        else:
            self.dreamsim_model, self.dreamsim_preprocess = None, None
            log.warning("DreamSim model not available")
        # Initialize Inception model for FID
        self.fid_fn = FrechetInceptionDistance(feature_dim=2048).to(self.device)
        log.info("FID model loaded from torcheval")

        log.info(f"VideoEvaluator initialized on device: {self.device}")

    def load_video(self, video_path: str, max_frames: Optional[int] = None) -> torch.Tensor:
        """
        Load video from file and return as tensor.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load (None for all frames)

        Returns:
            Video tensor of shape (T, C, H, W) in range [0, 1]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            # Read video using torchvision
            video, audio, info = read_video(video_path, pts_unit='sec')

            # Convert from (T, H, W, C) to (T, C, H, W) and normalize to [0, 1]
            video = video.permute(0, 3, 1, 2).float() / 255.0

            # Limit frames if specified
            if max_frames is not None and video.shape[0] > max_frames:
                video = video[:max_frames]

            log.debug(f"Loaded video {video_path}: shape {video.shape}")
            return video

        except Exception as e:
            log.error(f"Error loading video {video_path}: {e}")
            raise

    def resize_video(self, video: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Resize video frames to target size.

        Args:
            video: Video tensor of shape (T, C, H, W)
            target_size: Target (height, width)

        Returns:
            Resized video tensor
        """
        T, C, H, W = video.shape
        target_h, target_w = target_size

        if H == target_h and W == target_w:
            return video

        # Resize using bilinear interpolation
        video_resized = F.interpolate(
            video.view(-1, C, H, W),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )

        return video_resized.view(T, C, target_h, target_w)

    def compute_ssim(self, video1: torch.Tensor, video2: torch.Tensor) -> float:
        """
        Compute Structural Similarity Index (SSIM) between two videos.

        Args:
            video1: First video tensor (T, C, H, W)
            video2: Second video tensor (T, C, H, W)

        Returns:
            Average SSIM score across all frames
        """
        assert video1.shape == video2.shape, f"Video shapes must match: {video1.shape} vs {video2.shape}"

        # Convert to numpy and compute SSIM for each frame
        video1_np = video1.cpu().numpy()
        video2_np = video2.cpu().numpy()

        ssim_scores = []
        for t in range(video1_np.shape[0]):
            # Convert from CHW to HWC for skimage
            frame1 = np.transpose(video1_np[t], (1, 2, 0))
            frame2 = np.transpose(video2_np[t], (1, 2, 0))

            # Compute SSIM (multichannel=True for RGB)
            score = ssim(frame1, frame2, multichannel=True, channel_axis=2, data_range=1.0)
            ssim_scores.append(score)

        return float(np.mean(ssim_scores))

    def compute_lpips(self, video1: torch.Tensor, video2: torch.Tensor) -> float:
        """
        Compute Learned Perceptual Image Patch Similarity (LPIPS) between two videos.

        Args:
            video1: First video tensor (T, C, H, W)
            video2: Second video tensor (T, C, H, W)

        Returns:
            Average LPIPS score across all frames
        """
        if self.lpips_model is None:
            log.warning("LPIPS model not available, returning 0.0")
            return 0.0

        assert video1.shape == video2.shape, f"Video shapes must match: {video1.shape} vs {video2.shape}"

        # Move to device and normalize to [-1, 1] range for LPIPS
        video1 = video1.to(self.device) * 2.0 - 1.0
        video2 = video2.to(self.device) * 2.0 - 1.0

        lpips_scores = []

        # Process frames in batches to manage memory
        T = video1.shape[0]
        for i in range(0, T, self.batch_size):
            end_idx = min(i + self.batch_size, T)
            batch1 = video1[i:end_idx]
            batch2 = video2[i:end_idx]

            with torch.no_grad():
                batch_scores = self.lpips_model(batch1, batch2)
                lpips_scores.extend(batch_scores.cpu().numpy().flatten())

        return float(np.mean(lpips_scores))

    def compute_dreamsim(self, video1: torch.Tensor, video2: torch.Tensor) -> float:
        """
        Compute DreamSim distance between two videos (lower is better).
        Falls back gracefully if API variants differ.
        """
        if self.dreamsim_model is None or self.dreamsim_preprocess is None:
            log.warning("DreamSim model not available, returning 0.0")
            return 0.0
        assert video1.shape == video2.shape, f"Video shapes must match: {video1.shape} vs {video2.shape}"

        to_pil = transforms.ToPILImage()
        T = video1.shape[0]
        scores: List[float] = []

        # Ensure on CPU for PIL conversion, preprocessing will move to device after
        video1_cpu = video1.detach().cpu()
        video2_cpu = video2.detach().cpu()

        for i in range(0, T, self.batch_size):
            end_idx = min(i + self.batch_size, T)
            # Preprocess frames with DreamSim's preprocess (may produce center-crop or TenCrop)
            imgs1 = [self.dreamsim_preprocess(to_pil(video1_cpu[t])) for t in range(i, end_idx)]
            imgs2 = [self.dreamsim_preprocess(to_pil(video2_cpu[t])) for t in range(i, end_idx)]

            batch1 = torch.stack(imgs1, dim=0)
            batch2 = torch.stack(imgs2, dim=0)

            # Handle possible TenCrop outputs: (B, N, C, H, W) -> (B*N, C, H, W)
            crops = None
            if batch1.ndim == 5:
                B, N, C, H, W = batch1.shape
                batch1 = batch1.view(B * N, C, H, W)
                batch2 = batch2.view(B * N, C, H, W)
                crops = (B, N)

            batch1 = batch1.to(self.device)
            batch2 = batch2.to(self.device)

            with torch.no_grad():
                try:
                    scores_tensor = self._dreamsim_distance(batch1, batch2)
                except Exception as e:
                    log.error(f"DreamSim computation failed: {e}")
                    return 0.0

                # If we flattened TenCrop, reshape back and average crops
                if crops is not None:
                    B, N = crops
                    try:
                        scores_tensor = scores_tensor.view(B, N).mean(dim=1)
                    except Exception as e:
                        log.warning(f"Failed to aggregate TenCrop scores, using first crop: {e}")
                        scores_tensor = scores_tensor.view(B, N)[:, 0]

                scores.extend(scores_tensor.detach().cpu().numpy().flatten().tolist())

        return float(np.mean(scores)) if scores else 0.0

    def _dreamsim_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Robustly compute DreamSim distances between batches x and y.
        Tries different APIs and normalizes the returned object to a 1D tensor.
        """
        out = None

        # Try direct call
        try:
            out = self.dreamsim_model(x, y)
        except Exception as e:
            log.debug(f"DreamSim direct call failed: {e}")

        # Try explicit pairwise_distance
        if out is None:
            try:
                out = self.dreamsim_model.pairwise_distance(x, y)
            except Exception as e:
                log.debug(f"DreamSim pairwise_distance failed: {e}")

        # Fallback: compute embeddings and L2 distance
        if out is None:
            try:
                embed_fn = getattr(self.dreamsim_model, "embed", None) or getattr(self.dreamsim_model, "encode", None)
                if embed_fn is None:
                    raise AttributeError("DreamSim model has no embed/encode method")
                emb1 = embed_fn(x)
                emb2 = embed_fn(y)
                out = torch.norm(emb1 - emb2, dim=1)
            except Exception as e:
                raise RuntimeError(f"DreamSim embeddings fallback failed: {e}")

        # Normalize output to a tensor
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):
            # pick first tensor value
            out = next((v for v in out.values() if torch.is_tensor(v)), None)
        if not torch.is_tensor(out):
            out = torch.as_tensor(out, device=x.device)

        return out

    def evaluate_videos(self, gen_video_path: str, gt_video_path: str,
                        max_frames: Optional[int] = None,
                        metrics: Optional[set] = None,
                        cond_frames: int = 0,
                        resolution: Optional[Tuple[int, int]] = None,
                        start_frame: int = 0,
                        end_frame: int = -1,
                        ) -> Dict[str, float]:
        """
        Evaluate a pair of videos (generated vs ground truth).
        Only compute metrics requested via the 'metrics' set.
        For frame-wise metrics (ssim, lpips, dreamsim, fid), drop the first `cond_frames` frames.
        """
        metrics = set(metrics or {"ssim", "lpips", "dreamsim", "fid"})
        log.info(f"Evaluating: {gen_video_path} vs {gt_video_path}")

        # Load and align videos
        gen_video = self.load_video(gen_video_path, max_frames)
        gt_video = self.load_video(gt_video_path, max_frames)
        min_frames = min(gen_video.shape[0], gt_video.shape[0])
        gen_video = gen_video[:min_frames]
        gt_video = gt_video[:min_frames]
        gen_h, gen_w = gen_video.shape[2], gen_video.shape[3]
        gt_h, gt_w = gt_video.shape[2], gt_video.shape[3]
        if resolution is None:
            target_h = min(gen_h, gt_h)
            target_w = min(gen_w, gt_w)
        else:
            target_h, target_w = resolution
        gen_video = self.resize_video(gen_video, (target_h, target_w))
        gt_video = self.resize_video(gt_video, (target_h, target_w))

        # Apply conditioning cut only for frame-wise metrics and FID
        total_frames = gen_video.shape[0]
        cut = 0 if cond_frames <= 0 else min(cond_frames, total_frames)
        eval_gen_video = gen_video[cut:] if cut > 0 else gen_video
        eval_gt_video = gt_video[cut:] if cut > 0 else gt_video

        # Extract features based on requested metrics
        if "fid" in metrics:
            try:
                self.fid_fn.update(eval_gt_video, is_real=True)
                self.fid_fn.update(eval_gen_video, is_real=False)
            except Exception as e:
                log.warning(f"Inception features not available for FID: {e}")

        # For per frame metrics (ssim, lpips, dreamsim), compute only on the specified frame range
        eval_gt_video = eval_gt_video[start_frame:end_frame]
        eval_gen_video = eval_gen_video[start_frame:end_frame]
        if cut > 0:
            log.debug(f"Dropping first {cut} conditioning frames for frame-wise metrics/FID (total {total_frames})")

        # Compute requested per-pair metrics on eval_* (post-cut)
        results = {}
        if "ssim" in metrics:
            try:
                if eval_gen_video.shape[0] == 0:
                    raise ValueError("No frames left after cond_frames cut for SSIM")
                results['ssim'] = self.compute_ssim(eval_gen_video, eval_gt_video)
                log.info(f"SSIM: {results['ssim']:.4f}")
            except Exception as e:
                log.error(f"Error computing SSIM: {e}")
                results['ssim'] = 0.0
        if "lpips" in metrics:
            try:
                if eval_gen_video.shape[0] == 0:
                    raise ValueError("No frames left after cond_frames cut for LPIPS")
                results['lpips'] = self.compute_lpips(eval_gen_video, eval_gt_video)
                log.info(f"LPIPS: {results['lpips']:.4f}")
            except Exception as e:
                log.error(f"Error computing LPIPS: {e}")
                results['lpips'] = 0.0
        if "dreamsim" in metrics:
            try:
                if eval_gen_video.shape[0] == 0:
                    raise ValueError("No frames left after cond_frames cut for DreamSim")
                results['dreamsim'] = self.compute_dreamsim(eval_gen_video, eval_gt_video)
                log.info(f"DreamSim: {results['dreamsim']:.4f}")
            except Exception as e:
                log.error(f"Error computing DreamSim: {e}")
                results['dreamsim'] = 0.0

        return results

    def compute_fvd(self, gen_video_paths: List[str], gt_video_paths: List[str]) -> float:
        """
        Compute Fréchet Video Distance (FVD) between two sets of videos.
        This is a dataset-level metric that requires multiple videos.

        Args:
            gen_video_paths: List of paths to generated videos
            gt_video_paths: List of paths to ground truth videos

        Returns:
            FVD score (lower is better)
        """
        if not fvd_available or compute_fvd is None:
            log.warning("FVD computation not available, returning 0.0")
            return 0.0

        if len(gen_video_paths) != len(gt_video_paths):
            log.warning(f"Mismatched video counts: {len(gen_video_paths)} gen vs {len(gt_video_paths)} gt")

        try:
            log.info(f"Computing FVD on {len(gen_video_paths)} video pairs...")
            fvd_score = compute_fvd(gt_video_paths, gen_video_paths, self.device)
            return float(fvd_score)
        except Exception as e:
            log.error(f"Error computing FVD: {e}")
            return 0.0
