from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from cosmos_predict2.conditioner import DataType, TextCondition
from cosmos_predict2.pipelines.video2world import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
from cosmos_predict2.configs.base.config_video2world import Video2WorldPipelineConfig
from cosmos_predict2.utils.context_parallel import cat_outputs_cp, split_inputs_cp
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.utils import log, misc



class Video2WorldPoseConditionedPipeline(Video2WorldActionConditionedPipeline):
    """Pose-conditioned Video2World pipeline.
    
    This pipeline extends the action-conditioned pipeline to handle pose conditioning.
    It automatically converts pose data to the expected action format for compatibility.
    """
    
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)

    @staticmethod
    def from_config(
        config: Video2WorldPipelineConfig,
        dit_path: str = "",
        use_text_encoder: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_ema_to_reg: bool = False,
        load_prompt_refiner: bool = False,
    ) -> Any:
        """Create pose-conditioned pipeline from config."""
        config.resize_online = False  # Disable online resizing for pose conditioning
        # Create a pose-conditioned pipe
        pipe = Video2WorldPoseConditionedPipeline(device=device, torch_dtype=torch_dtype)
        
        # Use the parent class initialization but with our instance
        parent_pipe = Video2WorldActionConditionedPipeline.from_config(
            config=config,
            dit_path=dit_path,
            use_text_encoder=use_text_encoder,
            device=device,
            torch_dtype=torch_dtype,
            load_ema_to_reg=load_ema_to_reg,
            load_prompt_refiner=load_prompt_refiner,
        )
        
        # Copy all attributes from parent pipe
        for attr_name in dir(parent_pipe):
            if not attr_name.startswith('__') and hasattr(parent_pipe, attr_name):
                if attr_name in ["__call__","_get_data_batch_input","get_data_and_condition"]:
                    continue
                setattr(pipe, attr_name, getattr(parent_pipe, attr_name))
        
        return pipe

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None) -> None:
        if self.data_batch[self.input_video_key].shape[1] == 3:
            # Call the parent normalization if 3-channel input
            super()._normalize_video_databatch_inplace(data_batch)
        else:
            # If not 3-channel, means we are working with latents directly, no normalization needed
            pass


    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        poses: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        num_latent_conditional_frames: int = 1,
    ):
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.
            negative_prompt (str): Negative prompt.
            num_latent_conditional_frames (int, optional): The number of latent conditional frames. Defaults to 1.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = video.shape

        self.batch_size = 1
        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            # NOTE: we don't use text embeddings for pose conditional video2world
            "t5_text_embeddings": torch.zeros(
                self.batch_size,
                CosmosTextEncoderConfig.NUM_TOKENS,
                CosmosTextEncoderConfig.EMBED_DIM,
                dtype=torch.bfloat16,
            ).cuda(),
            "fps": torch.ones((self.batch_size,)) * 16,  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(self.batch_size, 1, H, W),  # Padding mask (assumed no padding here)
            "num_conditional_frames": num_latent_conditional_frames,  # Specify number of conditional frames
            "pose": poses,
        }

        # Handle negative prompts for classifier-free guidance
        if negative_prompt:
            data_batch["neg_t5_text_embeddings"] = self.encode_prompt(negative_prompt).to(dtype=self.torch_dtype)

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        return data_batch

    # This is called only during inference so we use videos and not latents
    @torch.no_grad()
    def __call__(
            self,
            video: torch.Tensor,
            poses: torch.Tensor,
            prompt: str = "",
            negative_prompt: str = "",
            num_conditional_frames: int = 1,
            guidance: float = 7.0,
            num_sampling_step: int = 35,
            seed: int = 0,
            solver_option: str = "2ab",
    ) -> torch.Tensor | None:

        log.success("Using the __call__ from pose conditioning pipeline")
        num_latent_conditional_frames = self.tokenizer.get_latent_num_frames(num_conditional_frames)

        # transform first frame and poses to tensor
        vid_input = video
        poses_tensor = poses.to(dtype=torch.bfloat16)[None, ...]

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input,
            poses_tensor,
            prompt,
            negative_prompt,
            num_latent_conditional_frames=num_latent_conditional_frames,
        )

        # preprocess
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_video_key
        n_sample = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            self.tokenizer.get_latent_num_frames(_T),
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=True)

        log.info("Starting video generation...")

        x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),  # noqa: RUF005
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * self.scheduler.config.sigma_max
        )

        # Split the input data and condition for model parallelism, if context parallelism is enabled.
        if self.dit.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

        # ------------------------------------------------------------------ #
        # Sampling loop driven by `RectifiedFlowAB2Scheduler`
        # ------------------------------------------------------------------ #
        scheduler = self.scheduler

        # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)

        # Bring the initial latent into the precision expected by the scheduler
        sample = x_sigma_max.to(dtype=torch.float32)

        x0_prev: torch.Tensor | None = None

        for i, _ in tqdm(enumerate(scheduler.timesteps)):
            # Current noise level (sigma_t).
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)

            # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
            # pass a 1-D tensor broadcastable to any later shape handling.
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with conditional and unconditional branches
            x0_pred = x0_fn(sample, sigma_in)

            # Scheduler step updates the noisy sample and returns the cached x0.
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        # Final clean pass at sigma_min.
        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        samples = x0_fn(sample, sigma_in)

        # Merge context-parallel chunks back together if needed.
        if self.dit.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        # Decode
        video = self.decode(samples)  # shape: (B, C, T, H, W), possibly out of [-1, 1]

        # Run video guardrail on the generated video and apply postprocessing
        if self.video_guardrail_runner is not None:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            # Clamp to safe range before normalization
            video = video.clamp(-1.0, 1.0)
            video_normalized = (video + 1) / 2  # [0, 1]

            # Convert tensor to NumPy frames for guardrail processing
            video_squeezed = video_normalized.squeeze(0)  # (C, T, H, W)
            frames = (video_squeezed * 255).clamp(0, 255).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)

            # Run guardrail
            processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
            if processed_frames is None:
                return None
            else:
                log.success("Passed guardrail on generated video")

            # Convert processed frames back to tensor format
            processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
            processed_video = processed_video * 2 - 1  # back to [-1, 1]
            processed_video = processed_video.unsqueeze(0)

            video = processed_video.to(video.device, dtype=video.dtype)

        log.success("Video generation completed successfully")
        return video