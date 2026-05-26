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

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from imaginaire.model import ImaginaireModel
from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


@dataclass
class _LossRecord:
    iter_count: int = 0
    loss: float = 0

    def reset(self) -> None:
        self.iter_count = 0
        self.loss = 0

    def get_stat(self) -> tuple[float, float]:
        if self.iter_count > 0:
            loss = self.loss / self.iter_count
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        else:
            loss = torch.ones(1)
        iter_count = self.iter_count
        self.reset()
        return loss.tolist(), iter_count


class LossLog(Callback):
    def __init__(
            self,
            logging_iter_multipler: int = 1,
            enable_tensorboard: bool = True,
    ) -> None:
        super().__init__()
        self.logging_iter_multipler = logging_iter_multipler
        self.name = self.__class__.__name__
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE

        self.train_video_log = _LossRecord()
        self.val_video_log = _LossRecord()
        self.summary_writer = None

    def _init_tensorboard_writer(self):
        """Initialize TensorBoard writer if enabled and not already initialized."""
        if (self.enable_tensorboard and
                self.summary_writer is None and
                hasattr(self, 'config') and
                distributed.is_rank0()):
            # Store TensorBoard logs in the same directory as stdout.log
            tensorboard_log_dir = os.path.join(self.config.job.path_local, "tensorboard")
            os.makedirs(tensorboard_log_dir, exist_ok=True)

            self.summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)
            log.info(f"TensorBoard logging initialized at: {tensorboard_log_dir}")

    def _log_to_tensorboard(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard if available."""
        if self.summary_writer is not None and distributed.is_rank0():
            self.summary_writer.add_scalar(tag, value, step)

    def _close_tensorboard_writer(self):
        """Close TensorBoard writer if it exists."""
        if self.summary_writer is not None:
            self.summary_writer.close()
            self.summary_writer = None

    def on_before_backward(
            self,
            model: ImaginaireModel,
            loss: torch.Tensor,
            iteration: int = 0,
    ):
        # Initialize TensorBoard writer if needed
        self._init_tensorboard_writer()

        # Log this loss for aligning the curve with diffsyncstudio
        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0 and distributed.is_rank0():
            info = {
                "train_loss_step": loss.detach().item(),
            }

            # Log to TensorBoard
            self._log_to_tensorboard("train/loss_step", loss.detach().item(), iteration)

    def on_before_optimizer_step(
            self,
            model_ddp,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            grad_scaler: torch.amp.GradScaler,
            iteration: int = 0,
    ) -> None:
        """Log learning rate to TensorBoard before optimizer step."""
        # Initialize TensorBoard writer if needed
        self._init_tensorboard_writer()

        # Log learning rate at regular intervals
        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0 and distributed.is_rank0():
            # Get the current learning rate from the scheduler
            current_lr = scheduler.get_last_lr()

            # Handle different scheduler types - some return lists, some single values
            if isinstance(current_lr, (list, tuple)) and len(current_lr) > 0:
                # If multiple parameter groups, log the first one (most common case)
                lr_value = current_lr[0]
                self._log_to_tensorboard("train/learning_rate", lr_value, iteration)

                # If multiple parameter groups, log them separately
                if len(current_lr) > 1:
                    for i, lr_val in enumerate(current_lr):
                        self._log_to_tensorboard(f"train/learning_rate_group_{i}", lr_val, iteration)
            elif isinstance(current_lr, (int, float)):
                self._log_to_tensorboard("train/learning_rate", current_lr, iteration)

    def on_training_step_end(
            self,
            model: ImaginaireModel,
            data_batch: dict[str, torch.Tensor],
            output_batch: dict[str, torch.Tensor],
            loss: torch.Tensor,
            iteration: int = 0,
    ):
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            log.critical(
                f"Unstable loss {loss} at iteration {iteration} with is_image_batch: {model.is_image_batch(data_batch)}",
                rank0_only=False,
            )

        if not skip_update_due_to_unstable_loss:
            _loss = output_batch["loss"].detach().mean(dim=0)

            self.train_video_log.iter_count += 1
            self.train_video_log.loss += _loss

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            world_size = dist.get_world_size()
            loss, iter_count = self.train_video_log.get_stat()
            iter_count *= world_size

            if distributed.is_rank0():
                info = {}
                if iter_count > 0:
                    info[f"train@{self.logging_iter_multipler}/loss"] = loss

                    # Log averaged loss to TensorBoard
                    self._log_to_tensorboard(f"train/loss_avg@{self.logging_iter_multipler}", loss, iteration)

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """Close TensorBoard writer when training ends."""
        self._close_tensorboard_writer()

    def on_validation_start(
            self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        """Reset validation loss accumulator at the start of validation."""
        self.val_video_log.reset()

    def on_validation_step_end(
            self,
            model: ImaginaireModel,
            data_batch: dict[str, torch.Tensor],
            output_batch: dict[str, torch.Tensor],
            loss: torch.Tensor,
            iteration: int = 0,
    ) -> None:
        """Accumulate validation loss for each validation step."""
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            log.critical(
                f"Unstable validation loss {loss} at iteration {iteration}",
                rank0_only=False,
            )

        if not skip_update_due_to_unstable_loss:
            _loss = output_batch["loss"].detach().mean(dim=0)

            self.val_video_log.iter_count += 1
            self.val_video_log.loss += _loss

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """Compute and log average validation loss at the end of validation."""
        world_size = dist.get_world_size()
        loss, iter_count = self.val_video_log.get_stat()
        iter_count *= world_size

        if distributed.is_rank0():
            if iter_count > 0:
                # Log to console
                log.info(f"Validation at iteration {iteration}: val_loss = {loss:.4f}")

                # Log to TensorBoard
                self._log_to_tensorboard("val/loss", loss, iteration)