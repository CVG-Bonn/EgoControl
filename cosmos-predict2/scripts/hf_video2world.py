#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate",
#   "cosmos-guardrail",
#   "diffusers>=0.34.0",
#   "transformers",
# ]
# [tool.uv]
# exclude-newer = "2025-08-15T00:00:00Z"
# override-dependencies = ["peft>=0.15.0"]
# ///

"""Example of Cosmos-Predict2 Video2World inference using Hugging Face diffusers."""

import argparse
import pathlib
import textwrap

import diffusers
import torch

ROOT = pathlib.Path(__file__).parents[1]
SEPARATOR = "-" * 20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=str, help="Output directory")
    parser.add_argument("--prompt", type=str, required=True, help="Path to prompt text file")
    parser.add_argument("--negative_prompt", type=str, help="Path to negative prompt text file")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--video", type=str, help="Path to video")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Predict2-2B-Video2World",
        help="Model name or path (Cosmos-Predict2: https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959)",
    )
    parser.add_argument("--revision", type=str, help="Model revision (branch name, tag name, or commit id)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--steps",
        type=int,
        default=35,
        help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7,
        help="Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2005.11487). Guidance scale is enabled by setting `guidance_scale > 1`.",
    )
    parser.add_argument("--height", type=int, default=704, help="The height in pixels of the generated image.")
    parser.add_argument("--width", type=int, default=1280, help="The width in pixels of the generated image.")
    parser.add_argument("--frames", type=int, default=93, help="The number of frames in the generated video.")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second of the generated video.")
    args = parser.parse_args()

    prompt = open(args.prompt).read()
    if args.negative_prompt is not None:
        negative_prompt = open(args.negative_prompt).read()
    else:
        negative_prompt = open(f"{ROOT}/prompts/video2world/negative/default.txt").read()

    if args.image is not None:
        image = diffusers.utils.load_image(args.image)
    else:
        image = None
    if args.video is not None:
        video = diffusers.utils.load_video(args.video)
    else:
        video = None

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(SEPARATOR)
        print("Prompt:")
        print(textwrap.indent(prompt.rstrip(), "  "))
        print("Negative Prompt:")
        print(textwrap.indent(negative_prompt.rstrip(), "  "))
        print(SEPARATOR)

    pipe = diffusers.Cosmos2VideoToWorldPipeline.from_pretrained(
        args.model,
        revision=args.revision,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    print("Generating video...")
    output = pipe(
        image=image,
        video=video,
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        fps=args.fps,
    ).frames[0]
    diffusers.utils.export_to_video(output, str(output_dir / "output.mp4"), fps=args.fps)
    print(f"Saved video to: {output_dir / 'output.mp4'}")


if __name__ == "__main__":
    main()
