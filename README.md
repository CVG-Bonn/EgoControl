<div align="center">
<h1>EgoControl: Controllable Egocentric Video Generation via 3D Full-Body Poses</h1>

[**Enrico Pallotta***](https://pallottaenrico.github.io/)<sup>1,2</sup> &nbsp;·&nbsp;
[**Sina Mokhtarzadeh Azar***](https://scholar.google.com/citations?user=kojTGo8AAAAJ&hl=en)<sup>1,2</sup> &nbsp;·&nbsp;
[**Lars Doorenboos**](https://scholar.google.com/citations?user=i2LqZCwAAAAJ&hl=en)<sup>1,2</sup>

[**Serdar Ozsoy**](https://scholar.google.com/citations?user=6jXE6SYAAAAJ&hl=en)<sup>1,2</sup> &nbsp;·&nbsp;
[**Umar Iqbal**](https://www.umariqbal.info/)<sup>3</sup> &nbsp;·&nbsp;
[**Juergen Gall**](https://scholar.google.de/citations?user=1CLaPMEAAAAJ)<sup>1,2</sup>

<sup>1</sup>University of Bonn &nbsp;&nbsp; <sup>2</sup>Lamarr Institute for Machine Learning and Artificial Intelligence &nbsp;&nbsp; <sup>3</sup>NVIDIA

&ast;Equal Contribution

<h3>⭐ CVPR 2026</h3>

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org/abs/2511.18173)
[![Project Page Badge](https://img.shields.io/badge/Project-EgoControl-Green)](https://cvg-bonn.github.io/EgoControl/)
[![Hugging Face Badge](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/PallottaEnrico/EgoControl)

</div>

---

> Official implementation of **EgoControl**, a method for controllable egocentric video generation conditioned on 3D full-body poses.

---

## 📋 Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Downloading Checkpoints](#downloading-checkpoints)
  - [Dataset](#dataset)
- [Inference](#inference)
  - [Simple Inference](#simple-inference)
  - [Evaluation Inference](#evaluation-inference)
- [Evaluation](#evaluation)
  - [Visual Quality Metrics](#visual-quality-metrics-ssim-lpips-dreamsim-fid-fvd)
  - [Body Control Accuracy](#body-control-accuracy-miou)
- [Training](#training) 
- [Citation](#citation)

---

## Getting Started

### Installation

> _We follow the original setup guide of cosmos-predict2; refer to [that guide](cosmos-predict2/documentations/setup.md) for additional details._

**System Requirements:**
- NVIDIA GPU with Ampere architecture (RTX 30xx, A100) or newer
- NVIDIA driver compatible with CUDA 12.6
- Linux x86-64
- glibc ≥ 2.31 (e.g. Ubuntu ≥ 22.04)
- Python 3.10

We **highly recommend** using [uv](https://docs.astral.sh/uv/getting-started/installation/).

**Install uv:**
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

**Option A — new environment:**
```shell
uv sync --extra cu126
source .venv/bin/activate
```

**Option B — active environment (e.g. conda):**
```shell
uv sync --extra cu126 --active --inexact
```

If you run scripts directly without installing the package, add the local sources to `PYTHONPATH`:
```shell
export PYTHONPATH="$PWD/cosmos-predict2:$PYTHONPATH"
```

---

### Downloading Checkpoints

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with **Read** permission.
2. Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli):
   ```shell
   uv tool install -U "huggingface_hub[cli]"
   ```
3. Log in:
   ```shell
   hf auth login
   ```
4. Accept the [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B).

**Download the base cosmos-predict2 2B weights:**
```shell
python cosmos-predict2/scripts/download_checkpoints.py --model_types video2world --model_sizes 2B --resolution 480
```

**Download the EgoControl fine-tuned checkpoint from [Hugging Face](https://huggingface.co/PallottaEnrico/EgoControl):**
```shell
hf download PallottaEnrico/EgoControl egocontrol-480p.pt --local-dir ./cosmos-predict2/checkpoints/
```

---

### Dataset

Follow the [Nymeria](https://github.com/facebookresearch/nymeria_dataset) repo to download the dataset and store it as:

```
nymeria_root/
└── train/
    └── video_name_01/
        └── recording_head.mp4   # video from the main RGB camera
```

We recommend using [`scripts/data/precompute_latents.py`](scripts/data/precompute_latents.py) to precompute and cache latents for faster training. It stores a subfolder `latents_{height}x{width}_{target_fps}fps_{frames_per_clip}f/` inside each `video_name_XX/` directory.

The current [dataset implementation](cosmos-predict2/cosmos_predict2/data/pose_conditioned/pose_conditioned_dataset.py) assumes this structure. You must also process Nymeria poses and place `.pt` pose files in the same video folder as the latent files.

Feel free to modify the dataset class to adapt to a different pipeline.

---

## Inference

All inference scripts share a common `--dit_path` argument pointing to the downloaded EgoControl checkpoint.

First:
```angular2html
cd cosmos-predict2
```
### Simple Inference

Use [`scripts/inference/simple.py`](scripts/inference/simple.py) to generate a video from a single input clip and one or more pose files.

**Single pose file:**
```shell
python ../scripts/inference/simple.py \
  --video ../examples/videos/20231113_s0_patricia_gutierrez_act3_0fk89s_chunk_0014.mp4 \
  --poses ../examples/poses/000080_003600_003644.pt \
  --dit_path path/to/egocontrol-480p.pt \
  --output ../outputs/ \
  --viz_pose
```
> `--viz_pose` saves a side-by-side video of the generation and the rendered pose.

**Folder of pose files** (one output video per pose file):
```shell
python ../scripts/inference/simple.py \
  --video ../examples/videos/20231113_s0_patricia_gutierrez_act3_0fk89s_chunk_0014.mp4 \
  --poses ../examples/poses/ \
  --dit_path path/to/egocontrol-480p.pt \
  --output ../outputs/
```

**Autoregressive generation** (chain multiple pose files into a longer video):
```shell
python ../scripts/inference/simple.py \
  --video ../examples/videos/20231113_s0_patricia_gutierrez_act3_0fk89s_chunk_0014.mp4 \
  --poses ../examples/autoregressive/ \
  --dit_path path/to/egocontrol-480p.pt \
  --output ../outputs/ar/ \
  --autoregressive
```

---

### Evaluation Inference

Use [`scripts/inference/evaluation_inference.py`](scripts/inference/evaluation_inference.py) to run inference over a folder of ground-truth video chunks. Output videos are saved in a sibling directory named after `--exp_name`, preserving original filenames for easy metric computation.

Pose files are looked up automatically from `--pose_root` using the convention:  
`<video>_chunk_<id>.mp4` → `<pose_root>/<video>/pose_<fps>fps_<frames>f/00<id>_*.pt`

**Single-GPU:**
```shell
python ../scripts/inference/evaluation_inference.py \
  --gt_folder path/to/gt_val_chunks/ \
  --pose_root path/to/dataset/val/ \
  --dit_path path/to/egocontrol-480p.pt \
  --exp_name my_experiment \
  --skip 16
```
> `--skip 16` subsamples the evaluation set by skipping every 16 chunks.

**Multi-GPU** (context parallelism):
```shell
torchrun --nproc_per_node=4 ../scripts/inference/evaluation_inference.py \
  --gt_folder path/to/gt_val_chunks/ \
  --pose_root path/to/dataset/val/ \
  --dit_path path/to/egocontrol-480p.pt \
  --exp_name my_experiment \
  --num_gpus 4
```

---

## Evaluation

We provide two evaluation scripts under [`scripts/eval/`](scripts/eval/).

### Visual Quality Metrics (SSIM, LPIPS, DreamSim, FID, FVD)

[`scripts/eval/visual_eval.py`](scripts/eval/visual_eval.py) computes frame-level and dataset-level visual quality metrics between generated and ground-truth videos. FVD must be run separately from the other metrics.

**Frame-level metrics (SSIM, LPIPS, DreamSim) + FID:**
```shell
python scripts/eval/visual_eval.py \
  --gen_folder path/to/generated/ \
  --gt_folder path/to/gt_chunks/ \
  --metrics ssim,lpips,dreamsim,fid \
  --cond_frames 13 \
  --output_path results.json
```

**Dataset-level FVD:**
```shell
python scripts/eval/visual_eval.py \
  --gen_folder path/to/generated/ \
  --gt_folder path/to/gt_chunks/ \
  --metrics fvd \
  --output_path results.json
```

Results are saved to a JSON file and merged across runs, so you can compute metrics incrementally.

---

### Body Control Accuracy (mIoU)

[`scripts/eval/body_control_eval.py`](scripts/eval/body_control_eval.py) evaluates how well generated videos follow the input body poses by computing the mean IoU between SAM2 segmentation masks of body/arms in ground-truth and generated frames. Masks should be pre-computed as `.npz` files (one per video).

First set up [SAM2](https://github.com/facebookresearch/sam2). We provide a segmentation script with manually annotated input points:
```shell
python scripts/eval/utils/sam2_video_segment.py \
  --videos_dir path/to/mp4_videos/ \
  --annotations_xml scripts/eval/segmentations/sam2_input_points.xml \
  --output_dir path/to/output_segmentations/
```

Then compute mIoU:
```shell
python scripts/eval/body_control_eval.py \
  path/to/gt_masks/ \
  path/to/pred_masks/ \
  --out_json body_control_results.json
```

---
## Training
A config file is needed to define the experiment configuration (dataset, model, parameters ...).
You can find an example [here](cosmos-predict2/cosmos_predict2/configs/pose_conditioned/experiment/fullbody_control.py) defining an experiment named "predict2_video2world_2b_pose_control".

To run the experiment use the following command:
```shell
cd cosmos-predict2

torchrun --nproc_per_node=4 --nnodes $SLURM_NNODES -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  -- experiment=predict2_video2world_2b_pose_control
```

Note: this command might vary based on your server setup.

## Citation

```bibtex
@InProceedings{Pallotta_2026_CVPR,
    author    = {Pallotta, Enrico and Azar, Sina Mokhtarzadeh and Doorenbos, Lars and Ozsoy, Serdar and Iqbal, Umar and Gall, Juergen},
    title     = {EgoControl: Controllable Egocentric Video Generation via 3D Full-Body Poses},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {4269-4279}
}
```
