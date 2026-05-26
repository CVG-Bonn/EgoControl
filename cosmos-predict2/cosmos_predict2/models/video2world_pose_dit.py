import torch
import torch.nn as nn
from einops import rearrange

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from imaginaire.utils.graph import create_cuda_graph
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic Transformer-style sinusoidal positional encoding.
    Works for arbitrary sequence lengths (no learned parameters).
    """
    def __init__(self, dim, max_len=1000):
        super().__init__()
        # Create a (1, max_len, dim) buffer of positional encodings
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        """
        x: (B, T, D) tensor
        Returns: x + positional_encoding[:T]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoseConditionedMinimalV1LVGDiT(MinimalV1LVGDiT):
    def __init__(self, *args, **kwargs):
        assert "pose_dim" in kwargs, "pose_dim must be provided"
        pose_dim = kwargs["pose_dim"]
        self.pose_dim = pose_dim
        del kwargs["pose_dim"]
        self.use_pose_cross_attn = kwargs.pop("use_pose_cross_attn", True)
        self.use_head_adaln = kwargs.pop("use_head_adaln", False)
        self.use_body_adaln = kwargs.pop("use_body_adaln", True)
        use_head_velocity = kwargs.pop("use_head_velocity", True)
        use_cross_attn_proj = kwargs.pop("use_cross_attn_proj", True)
        all_velocity = kwargs.pop("all_velocity", False)
        self.no_scale = kwargs.pop("no_scale", False)
        self.pose_channels = kwargs.pop("pose_channels", 1024)

        super().__init__(*args, **kwargs)

        if self.use_head_adaln or self.use_body_adaln:
            if self.use_head_adaln:
                pose_dim1 = 1
            elif self.use_body_adaln:
                pose_dim1 = pose_dim[1]
            pose_scaler = torch.ones((pose_dim[0], pose_dim1, pose_dim[2]), dtype=torch.bfloat16)

            if use_head_velocity:
                num_velocity_joints = 1 if pose_dim1 == 1 else 2  # 1 for head only, 2 for head + root
            else:
                num_velocity_joints = 1

            if all_velocity:
                num_velocity_joints = pose_dim1

            pose_scaler[:, :num_velocity_joints, :3] *= 20.0  # Scale translation
            pose_scaler[:, :num_velocity_joints, 3:] *= 10.0  # Scale rotation
            pose_scaler = pose_scaler.reshape(-1)  # Flatten

            if not self.no_scale:
                self.pose_scaler = nn.Parameter(pose_scaler, requires_grad=True)  # Learnable scaling

            p_dim = pose_dim[0] * pose_dim1 * pose_dim[2]

            self.pose_embedder_B_D = Mlp(
                in_features=p_dim,
                hidden_features=self.model_channels * 4,
                out_features=self.model_channels,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
            self.pose_embedder_B_3D = Mlp(
                in_features=p_dim,
                hidden_features=self.model_channels * 4,
                out_features=self.model_channels * 3,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )


        # Add projection for pose tokens
        if self.use_pose_cross_attn:
            pose_dim_per_frame = pose_dim[1] * pose_dim[2]

            self.pose_to_token = nn.Sequential(
                nn.Linear(pose_dim_per_frame, self.pose_channels),  # no flattening over time
                nn.GELU(),
                nn.LayerNorm(self.pose_channels),
            )

            if use_cross_attn_proj:
                self.crossattn_proj = nn.Linear(self.pose_channels, self.pose_channels)  # Adjust dimensions as needed
            else:
                self.crossattn_proj = None
            # Small random initialization for weights
            # nn.init.xavier_uniform_(self.pose_to_tokens[0].weight, gain=0.01)
            # nn.init.zeros_(self.pose_to_tokens[0].bias)
            # nn.init.xavier_uniform_(self.crossattn_proj.weight, gain=0.01)
            # nn.init.zeros_(self.crossattn_proj.bias)
        self.pose_pos_enc = SinusoidalPositionalEncoding(dim=self.pose_channels, max_len=256)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        use_cuda_graphs: bool = False,
        pose: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

        if self.use_pose_cross_attn:
            # Generate pose tokens and append to crossattn_emb
            pose_tokens = self.pose_to_token(pose)
            pose_tokens = self.pose_pos_enc(pose_tokens)
            if crossattn_emb is None:
                crossattn_emb = pose_tokens
            else:
                if crossattn_emb.size(-1) != pose_tokens.size(-1):
                    crossattn_emb = crossattn_emb.to(self.crossattn_proj.weight.dtype)  # Ensure dtype matches
                    crossattn_emb = self.crossattn_proj(crossattn_emb)
                crossattn_emb = torch.cat([crossattn_emb, pose_tokens], dim=1)

        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        assert not (self.training and use_cuda_graphs), "CUDA Graphs are supported only for inference"
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if self.crossattn_proj is not None:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

        # Ensure pose is provided
        assert pose is not None, "pose must be provided"
        pose = rearrange(pose, "b t d -> b 1 (t d)")

        if self.use_head_adaln or self.use_body_adaln:
            if self.use_head_adaln:
                pose = rearrange(pose, "b 1 (t j c) -> b t j c", j=self.pose_dim[1], c=self.pose_dim[2])
                head_pose = pose[:, :, :1, :].reshape(pose.shape[0], 1, -1)  # [B, 1, head_dim]
                if not self.no_scale:
                    pose = head_pose * self.pose_scaler.to(pose.device)  # Apply scaling
                else:
                    pose = head_pose
            else:
                if not self.no_scale:
                    pose = pose * self.pose_scaler.to(pose.device)  # Apply scaling

            pose_emb_B_D = self.pose_embedder_B_D(pose)
            pose_emb_B_3D = self.pose_embedder_B_3D(pose)

            # NOTE: add pose embedding to the timestep embedding and adaln_lora
            t_embedding_B_T_D = t_embedding_B_T_D + pose_emb_B_D
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + pose_emb_B_3D

        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        if use_cuda_graphs:
            shapes_key = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        for block in blocks:
            x_B_T_H_W_D = block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs)
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp