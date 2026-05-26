from __future__ import annotations

from typing import Optional

import torch

from .tokenizer import TokenizerInterface


class PrecomputedLatentTokenizer(TokenizerInterface):
    """
    Passthrough tokenizer for precomputed video latents.

    - encode() is identity (optionally normalize if your stored latents are unnormalized).
    - decode() optionally uses the VAE decoder to visualize latents.
    - spatial/temporal compression factors are set to 1 because the inputs are already latents.
    - get_latent_num_frames / get_pixel_num_frames are identity mappings.

    IMPORTANT:
    Your stored latents must match the model’s expected latent space:
      [B, C=16, T_latent, H_latent, W_latent]
    where (H_latent, W_latent) = (H_pixel / VAE_spatial_factor, W_pixel / VAE_spatial_factor)
    and T_latent = T_pixel / VAE_temporal_factor.
    """
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        latents = state # Skip forward pass through VAE encoder
        num_frames = latents.shape[2]
        if num_frames == 1:
            return (latents - self.model.img_mean.type_as(latents)) / self.model.img_std.type_as(latents)
        else:
            return (latents - self.model.video_mean[:, :, :num_frames].type_as(latents)) / self.model.video_std[
                                                                                           :, :, :num_frames
                                                                                           ].type_as(latents)
