import os
from typing import Optional, Any, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchvision.utils import make_grid
from einops import rearrange

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .utils import count_params, pl_on_train_tart
from diffusers import AutoencoderKL


class VAETrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        pretrained_model_path = config.pretrained_model_path
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, subfolder="vae")
        count_params(self.vae)
        if config.precision == 16:
            self.data_dtype = torch.float16
        elif config.precision == 32:
            self.data_dtype = torch.float32
        self.loss_fn = nn.MSELoss()

    @rank_zero_only
    def on_train_start(self):
        pl_on_train_tart(self)

    def forward(self, batch, mode="train"):
        image = batch["image"].to(self.vae.dtype)
        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * 0.18215
        latents = 1.0 / 0.18215 * latents  # introduce analytical error
        decoded = self.vae.decode(latents).sample
        loss = self.loss_fn(image, decoded)
        if mode == "train":
            return loss
        elif mode == "validation":
            return loss, decoded

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, decoded = self(batch, mode="validation")
        self.log(
            "train/loss",
            loss,
            batch_size=len(batch["image"]),
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": loss, "image": batch["image"], "decoded": decoded}

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss, decoded = self(batch, mode="validation")
        self.log(
            "validation/loss",
            loss,
            batch_size=len(batch["image"]),
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": loss, "image": batch["image"], "decoded": decoded}

    def configure_optimizers(self) -> Any:
        lr = self.learning_rate
        params = [{"params": self.vae.parameters()}]
        print(
            f"Initialize optimizer with: lr: {lr}, weight_decay: {self.config.weight_decay}, eps: {self.config.adam_epsilon}"
        )
        opt = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_epsilon,
        )
        return opt

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        optimizer_idx: int,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )


class VAEImageLogger(Callback):
    def __init__(self, train_batch_frequency, val_batch_frequency):
        super().__init__()
        self.batch_freq = {
            "train": train_batch_frequency,
            "validation": val_batch_frequency,
        }

    @rank_zero_only
    def _wandb_image(self, pl_module, results, batch_idx, split):
        print(f"Log images to wandb at: {split}/{batch_idx}")
        raw_image = results["image"]
        recontstruct_image = results["decoded"]
        batch_size = raw_image.shape[0]
        with torch.no_grad():
            # 2 x B x 3 x H x W
            grids = torch.stack([raw_image, recontstruct_image])
            grids = torch.clamp((grids + 1.0) / 2.0, min=0.0, max=1.0)
            grids = rearrange(grids, "g b c h w -> c (g h) (b w)", g=2)
            # 4 pairs in a group
            split = batch_size // 4
            groups = torch.tensor_split(grids, split, dim=2)

            def reshape_for_grid(group): return rearrange(
                group, "c (g h) (b w) -> (g b) c h w", g=2, b=4
            )
            groups = [make_grid(reshape_for_grid(group), nrow=4)
                      for group in groups]
        pl_module.logger.log_image(
            key=f"image-{split}/{batch_idx}",
            images=groups,
            step=batch_idx,
        )

    def check_freq(self, batch_dix, split="train"):
        return (batch_dix + 1) % self.batch_freq[split] == 0

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        split = "train"
        if self.check_freq(batch_idx, split=split):
            self._wandb_image(pl_module, outputs, batch_idx, split=split)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        split = "validation"
        if self.check_freq(batch_idx, split=split):
            self._wandb_image(pl_module, outputs, batch_idx, split=split)
