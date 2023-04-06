import os
import re
import inspect
from omegaconf import OmegaConf
from typing import Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
import torchvision
import torch.distributed as torchdist
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from einops import rearrange

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from diffusers import AutoencoderKL, UNet2DConditionModel, ConfigMixin
from transformers import CLIPTextModel, CLIPTokenizer, CanineModel, T5EncoderModel
from ..model import (
    MaskMSELoss,
    UNet2DMultiConditionModel,
    CharEmbedder,
    CharEmbedderConfig,
    CharTokenizer,
    convert_single_cond_unet,
    convert_fourchannel_unet,
)
from typing import List, Dict
from .utils import (
    count_params,
    pl_on_train_tart,
    module_requires_grad,
    get_obj_from_str,
)

FIXED_COMMIT_IDS = {
    "CompVis/stable-diffusion-v1-4": "3857c45b7d4e78b3ba0f39d4d7f50a2a05aa23d4",
    "runwayml/stable-diffusion-inpainting": "caac1048f28756b68042add4670bec6f4ae314f8",
}


class CharInpaintModelWrapper:
    def __init__(self):
        pass

    def __new__(cls, config):
        if config.source == "raw":
            print(f"Initailize model from {config.pretrained_model_path}")
            FIXED_COMMIT_ID = FIXED_COMMIT_IDS[config.pretrained_model_path]
            model = CharInpaintTrainer(config=config)
            if "char_embedder" in config:  # multi condition UNet
                old_unet = UNet2DConditionModel.from_pretrained(
                    config.pretrained_model_path,
                    subfolder="unet",
                    revision=FIXED_COMMIT_ID,
                )
                # move weights in old_unet to new unet
                if "inpainting" in config.pretrained_model_path:
                    print("Converting unet from inpainting diffusion checkpoint")
                    convert_single_cond_unet(old_unet, model.unet)
                else:
                    print("Converting unet from raw diffusion checkpoint")
                    convert_fourchannel_unet(old_unet, model.unet)
                del old_unet
        elif config.source == "inpainting-trained":
            print(
                f"Resume inpainting-trained model from {config.pretrained_model_path}"
            )
            model = CharInpaintTrainer.load_from_checkpoint(
                config.pretrained_model_path, map_location="cpu"
            )
            print("Pretrained model config:")
            print(model.config)
            if config.freeze_char_embedder:
                model.freeze_char_embedder()
        return model

    @classmethod
    def load_from_checkpoint(cls, path):
        return CharInpaintTrainer.load_from_checkpoint(path)


class CharInpaintTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        FIXED_COMMIT_ID = FIXED_COMMIT_IDS.get(
            config.pretrained_model_path, None)
        pretrained_model_path = config.pretrained_model_path
        if self.config.vae is not None and self.config.vae.get(
            "pretrained_model_path", False
        ):
            print("Initialize vae from finetuned...")
            if os.path.exists(self.config.vae.get("pretrained_model_path")):
                self.vae = AutoencoderKL.from_pretrained(
                    self.config.vae.get("pretrained_model_path"),
                )
            else:
                self.vae = AutoencoderKL.from_pretrained(
                    pretrained_model_path,
                    subfolder="vae",
                    revision=FIXED_COMMIT_ID,
                )
            self.NORMALIZER = self.config.vae.get("normalizer", 0.18215)
        else:
            print("Initialize vae from pretrained...")
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_path,
                subfolder="vae",
                revision=FIXED_COMMIT_ID,
            )
            self.NORMALIZER = 0.18215  # original normalizer
        self.noise_scheduler = get_obj_from_str(config.noise_scheduler).from_config(
            pretrained_model_path,
            subfolder="scheduler",
            revision=FIXED_COMMIT_ID,
        )

        self.tokenizer = (
            None
            if "tokenizer" not in config
            else (
                # we may set max_length for tokenizer
                CLIPTokenizer.from_pretrained(
                    pretrained_model_path,
                    subfolder="tokenizer",
                    revision=FIXED_COMMIT_ID,
                    **config.tokenizer,
                )
            )
        )
        self.text_encoder = (
            None
            if "tokenizer" not in config
            else (
                CLIPTextModel.from_pretrained(
                    pretrained_model_path,
                    subfolder="text_encoder",
                    revision=FIXED_COMMIT_ID,
                )
            )
        )

        if 'char_tokenizer' in config:  # if we add character embedding
            if config.char_tokenizer.get("target", "CharTokenizer") == "CharTokenizer":
                print("Initialize CharTokenizer")
                target_cls = CharTokenizer
                self.char_tokenizer = (
                    CharTokenizer.from_pretrained(
                        config.char_tokenizer.pretrained_path,
                        **config.char_tokenizer,
                    )
                    if "char_tokenizer" in config
                    else (
                        CharTokenizer.from_pretrained(
                            config.char_tokenizer.pretrained_path,
                            **config.char_tokenizer,
                            cliptokenizer=self.tokenizer,
                        )
                    )
                )
                self.char_embedder = (
                    CharEmbedder(CharEmbedderConfig(**config.char_embedder))
                    if "char_embedder" in config
                    else self.text_encoder
                )
            else:
                tokenizer_name = config.char_tokenizer.target
                print("Loading ", tokenizer_name)

                target_cls = get_obj_from_str(config.char_tokenizer.target)
                self.char_tokenizer = target_cls.from_pretrained(
                    config.char_tokenizer.pretrained_path, **config.char_tokenizer
                )
                if "canine" in config.char_tokenizer.target.lower():
                    self.char_embedder = CanineModel.from_pretrained(
                        config.char_tokenizer.pretrained_path
                    )
                elif "byt5" in config.char_tokenizer.target.lower():
                    self.char_embedder = T5EncoderModel.from_pretrained(
                        config.char_tokenizer.pretrained_path
                    ).encoder
                else:
                    raise ValueError("Unkown char tokenizer")

            if "char_embedder" in config:
                self.char_embedder.requires_grad_(True)
            else:
                self.char_embedder.requires_grad_(False)

        if 'char_tokenizer' in config:
            self.unet = UNet2DMultiConditionModel(
                **OmegaConf.to_container(config.unet))
        else:
            oldunet = UNet2DConditionModel.from_pretrained(
                pretrained_model_path,
                subfolder="unet",
                revision=FIXED_COMMIT_ID,
            )
            oldconfig = dict(oldunet.config)
            if oldconfig['in_channels'] != 9:
                print("Converting 4-channel unet to 9-channel unet...")
                oldconfig['in_channels'] = 9
                self.unet = UNet2DConditionModel(**oldconfig)
                convert_fourchannel_unet(oldunet, self.unet)
            else:
                print("Reusing 9-channel unet...")
                self.unet = oldunet

        if config.loss_type == "MaskMSELoss":
            self.loss_fn = MaskMSELoss(alpha=config.loss_alpha)
        else:
            self.loss_fn = nn.MSELoss()

        if self.config.get("optimize_vae", False):
            self.vae.requires_grad_(True)
        else:
            self.vae.requires_grad_(False)
        if self.text_encoder:
            self.text_encoder.requires_grad_(False)

        if config.precision == 16:
            self.data_dtype = torch.float16
        elif config.precision == 32:
            self.data_dtype = torch.float32

        if config.get('only_char', False):
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            # self.unet.
        self.count_params()  # report param number

    def count_params(self):
        count_params(self.vae)
        count_params(self.unet)
        if self.text_encoder:
            count_params(self.text_encoder)
        if hasattr(self, 'char_embedder'):
            count_params(self.char_embedder)

    #########################################################################
    # pytorch_lightning training related code
    #########################################################################

    def on_train_start(self) -> None:
        pl_on_train_tart(self)

    def on_fit_start(self) -> None:
        # synchronize between different gpus
        if torchdist.is_initialized():
            print(f"Fit synchronize on rank: {torchdist.get_rank()}")
            torchdist.barrier()
            torch.cuda.synchronize()

    def configure_optimizers(self):
        lr = self.learning_rate
        params = [{"params": self.unet.parameters()}]
        if hasattr(self, 'char_embedder') and module_requires_grad(self.char_embedder):
            print("Optimize char embedder together with unet")
            params.append({"params": self.char_embedder.parameters()})
        if self.config.get("optimize_vae", False):
            print("Optimize vae together with unet")
            params.append({"params": self.vae.parameters()})

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
        # avoid gradient exploding
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def shared_step(self, batch, batch_idx, stage="train"):
        loss = self(batch)
        if len(loss) == 1:
            loss = loss[0]
            self.log(
                f"{stage}_traj/loss",
                loss,
                batch_size=len(batch["image"]),
                prog_bar=True,
                sync_dist=True,
            )
        else:
            loss, mse_loss, ocr_loss = loss
            self.log(
                f"{stage}_traj/loss",
                loss,
                batch_size=len(batch["image"]),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_traj/mse_loss",
                mse_loss,
                batch_size=len(batch["image"]),
                sync_dist=True,
            )
            if self.config.get("optimize_vae", False):
                self.log(
                    f"{stage}_traj/reconstruct_loss",
                    ocr_loss,
                    batch_size=len(batch["image"]),
                    prog_bar=False,
                    sync_dist=True,
                )
            else:
                self.log(
                    f"{stage}_traj/ocr_loss",
                    ocr_loss,
                    batch_size=len(batch["image"]),
                    prog_bar=True,
                    sync_dist=True,
                )
        return loss

    def training_step(self, batch, batch_idx):
        # global rank, no need to manually save
        loss = self.shared_step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="valid")
        return loss

    # @rank_zero_only
    # def on_validation_start(self) -> None:
    #     # save before validation
    #     path = os.path.join(self.config['ckpt_dir'], "last.ckpt")
    #     print(f"Manually saving to {path}")
    #     self.trainer.save_checkpoint(path)

    #########################################################################
    # Training related code
    #########################################################################

    @torch.no_grad()
    def prepare_input(self, batch):
        image = batch["image"].to(self.vae.dtype)
        mask = batch["mask"].to(self.vae.dtype)
        masked_image = batch["masked_image"].to(self.vae.dtype)
        resolution = image.shape[-1]
        mask = (
            torch.nn.functional.interpolate(
                mask, size=(resolution // 8, resolution // 8)
            )
            .to(memory_format=torch.contiguous_format)
            .float()
        )
        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.NORMALIZER
        masked_image_latents = self.vae.encode(
            masked_image).latent_dist.sample()
        masked_image_latents = masked_image_latents * self.NORMALIZER

        chars = batch["chars"]
        style_chars = batch['style']
        encoder_hidden_states = self.prepare_condition_hidden_states(
            chars, style_chars)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (bsz,), dtype=torch.long
        ).to(self.device)
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps)
        latent_inputs = torch.cat(
            [noisy_latents, mask, masked_image_latents], dim=1)
        latent_inputs = self.noise_scheduler.scale_model_input(
            latent_inputs, timesteps)
        return (noise, timesteps, latent_inputs, mask, encoder_hidden_states)

    def calc_loss(self, noise_pred, noise, mask):
        if type(self.loss_fn) == MaskMSELoss:
            loss = self.loss_fn(
                noise_pred, noise, mask.repeat(1, 4, 1, 1).to(self.device)
            )
        else:
            loss = self.loss_fn(noise_pred, noise)
        return loss

    def forward(self, batch):
        (
            noise,
            timesteps,
            latent_inputs,
            mask,
            encoder_hidden_states,
        ) = self.prepare_input(batch)
        noise_pred = self.unet(latent_inputs, timesteps,
                               encoder_hidden_states).sample
        mse_loss = self.calc_loss(noise_pred, noise, mask)
        loss = mse_loss
        if self.config.get("optimize_vae", False):
            latent_inputs = latent_inputs[:, :4, ...]
            assert latent_inputs.shape == noise_pred.shape
            pred_x0 = torch.cat(
                [
                    self.noise_scheduler.step(
                        noise_pred[i: i +
                                   1], timesteps[i], latent_inputs[i: i + 1]
                    ).pred_original_sample
                    for i in range(len(batch["chars"]))
                ]
            )
            pred_x0 = 1.0 / self.NORMALIZER * pred_x0
            pred_image_x0 = self.convert_latent2image(pred_x0, tocpu=False)
            reconstruct_loss = nn.MSELoss()(batch["image"], pred_image_x0)
            loss = mse_loss + reconstruct_loss
            return (loss, mse_loss, reconstruct_loss)
        return (loss,)

    #########################################################################
    # Sampling related code
    #########################################################################

    @torch.no_grad()
    def prepare_condition_hidden_states(
        self,
        chars: List,
        style_chars: List,
        do_classifier_free_guidance=False,
        num_sample_per_image=1,
    ):
        text_hidden_states = None
        char_hidden_states = None
        bs_size = len(chars)
        if self.text_encoder is not None:
            input_ids = self.tokenizer(
                style_chars, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            text_hidden_states = self.text_encoder(input_ids.to(self.device))[0]
        if hasattr(self, 'char_embedder'):
            char_input_ids = self.char_tokenizer(
                chars,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=(hasattr(self, 'char_tokenizer')
                                    and self.char_tokenizer.cls_token_id is not None),
            ).input_ids
            char_hidden_states = self.char_embedder(
                char_input_ids.to(self.device))[0]

        if num_sample_per_image != 1:
            text_hidden_states = expand_hidden_states(
                text_hidden_states, num_sample_per_image
            )
            if char_hidden_states is not None:
                char_hidden_states = expand_hidden_states(
                    char_hidden_states, num_sample_per_image
                )

        if do_classifier_free_guidance:
            uncond_tokens = [" "] * bs_size
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=(hasattr(self, 'char_tokenizer')
                                    and self.char_tokenizer.cls_token_id is not None),
            ).input_ids
            uncond_embeddings = self.text_encoder(
                uncond_input.to(self.device))[0]
            if num_sample_per_image != 1:
                uncond_embeddings = expand_hidden_states(
                    uncond_embeddings, num_sample_per_image
                )
            text_hidden_states = torch.cat(
                [uncond_embeddings, text_hidden_states])

            if char_hidden_states is not None:
                uncond_input = self.char_tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=(hasattr(
                        self, 'char_tokenizer') and self.char_tokenizer.cls_token_id is not None),
                ).input_ids
                uncond_embeddings = self.char_embedder(
                    uncond_input.to(self.device))[0]
                if num_sample_per_image != 1:
                    uncond_embeddings = expand_hidden_states(
                        uncond_embeddings, num_sample_per_image
                    )
                char_hidden_states = torch.cat(
                    [uncond_embeddings, char_hidden_states])
        encoder_hidden_states = {
            "text": text_hidden_states,
            "char": char_hidden_states,
        }
        if char_hidden_states is None:
            # only text encoder is used, which corresponds to no character input
            encoder_hidden_states = encoder_hidden_states['text']
        return encoder_hidden_states

    @torch.no_grad()
    def sample_loop(
        self,
        latents: torch.Tensor,
        mask: torch.Tensor,
        masked_image_latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        do_classifier_free_guidance: bool = False,
        guidance_scale: float = 7.5,
        return_intermediates: Optional[bool] = False,
        extra_step_kwargs: Optional[Dict] = {},  # we may provide generator here
        **kwargs,
    ):
        intermediates = []  # for now we only return predicted image
        res = {}
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = torch.cat(
                [latent_model_input, mask, masked_image_latents], dim=1
            )
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t
            ).to(dtype=self.unet.dtype)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=encoder_hidden_states
            ).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            scheduler_res = self.noise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            )
            latents = scheduler_res.prev_sample
            if return_intermediates:
                intermediates.append(scheduler_res.pred_original_sample)

        latents = 1 / self.NORMALIZER * latents  # scale back
        res["latents"] = latents
        if len(intermediates) != 0:
            intermediates = [1 / self.NORMALIZER * x for x in intermediates]
            res["intermediates"] = intermediates
        return res

    # convert to image space
    def convert_latent2image(self, latent, tocpu=True):
        image = self.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        if tocpu:
            image = image.cpu()
        return image

    # single prompt for single image, leave the expansion outside sample function
    @torch.no_grad()
    def sample(
        self,
        batch: Dict[str, Union[torch.Tensor, List[str]]],
        guidance_scale: float = 7.5,
        num_sample_per_image: int = 1,
        num_inference_steps: int = 50,
        eta: float = 0.0,  # for DDIM
        generator: Optional[torch.Generator] = None,
        return_intermediates: Optional[bool] = False,
        **kwargs,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        # prepare input
        mask = batch["mask"].to(self.vae.dtype).to(self.device)
        masked_image = batch["masked_image"].to(self.vae.dtype).to(self.device)
        resolution = masked_image.shape[-1]
        mask = (
            torch.nn.functional.interpolate(
                mask, size=(resolution // 8, resolution // 8)
            )
            .to(memory_format=torch.contiguous_format)
            .float()
        )
        masked_image_latents = self.vae.encode(
            masked_image).latent_dist.sample()
        masked_image_latents = masked_image_latents * self.NORMALIZER
        if num_sample_per_image > 1:
            mask = expand_hidden_states(mask, num_sample_per_image)
            masked_image_latents = expand_hidden_states(
                masked_image_latents, num_sample_per_image
            )

        if "latents" in batch:  # ! maybe delete this?
            image = batch["latents"].to(self.vae.dtype).to(self.device)
            image_latents = self.vae.encode(image).latent_dist.sample()
            image_latents = image_latents * self.NORMALIZER
            if image_latents.shape != masked_image_latents.shape:
                image_latents = expand_hidden_states(
                    image_latents, num_sample_per_image
                )
        else:
            image_latents = torch.randn(
                masked_image_latents.shape,
                generator=generator,
                device=masked_image_latents.device,
                dtype=masked_image_latents.dtype,
            )

        chars = batch["chars"]
        style_chars = batch['style']
        encoder_hidden_states = self.prepare_condition_hidden_states(
            chars,
            style_chars,
            do_classifier_free_guidance,
            num_sample_per_image=num_sample_per_image,
        )

        # prepare for classifier free guidance
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )

        # set up timesteps tensors
        self.noise_scheduler.set_timesteps(
            num_inference_steps, device=self.vae.device)
        timesteps = self.noise_scheduler.timesteps

        # scale latent input
        image_latents = image_latents * self.noise_scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        # get diffusion prediction
        latent_results = self.sample_loop(
            image_latents,
            mask,
            masked_image_latents,
            encoder_hidden_states,
            timesteps,
            do_classifier_free_guidance,
            guidance_scale,
            return_intermediates=return_intermediates,
            extra_step_kwargs=extra_step_kwargs,
            **kwargs,
        )

        image_results = {}
        images = self.convert_latent2image(latent_results["latents"])
        image_results["images"] = images
        if return_intermediates:
            intermediate_images = [
                self.convert_latent2image(x) for x in latent_results["intermediates"]
            ]
            image_results["intermediate_images"] = intermediate_images
        return image_results

    @torch.no_grad()
    def log_images(self, batch, generation_kwargs, stage="train", cat_gt=False):
        image_results = dict()
        if (
            stage == "train" or stage == "validation" or stage == "valid"
        ):  # one prompt for each image, so we can just provide original batch to sample()
            assert type(batch["chars"][0]) == str
            num_sample_per_image = generation_kwargs.get(
                "num_sample_per_image", 1)
            sample_results = self.sample(batch, **generation_kwargs)
            # create (caption, image) dict
            for i, caption in enumerate(batch["chars"]):
                raw_image = batch["image"][i].cpu()  # 3 * size * size
                raw_image = (raw_image / 2 + 0.5).clamp(0., 1.)
                # convert 1 * size * size to 3 * size * size
                raw_mask = (batch["masked_image"][i].cpu() /
                            2 + 0.5).clamp(0.0, 1.0)
                # num_sample_per_image * 3 * size * size
                sample_res = sample_results["images"][
                    i * num_sample_per_image: (i + 1) * num_sample_per_image
                ]
                #! for now we do not considier intermediates, they are too much
                if cat_gt:
                    image_results[f"{i}-{caption}"] = torch.cat(
                        [torch.stack([raw_image, raw_mask]), sample_res]
                    )
                else:
                    # don't cat raw image and mask to result
                    image_results[f"{i}-{caption}"] = sample_res

        return image_results


#########################################################################
# Image logger
#########################################################################
class CharInpaintImageLogger(Callback):
    def __init__(
        self,
        train_batch_frequency,
        valid_batch_frequency,
        generation_kwargs,
        metric_callback=None,
        disable_wandb=False,
    ):
        super().__init__()
        self.batch_freq = {
            "train": train_batch_frequency,
            "valid": valid_batch_frequency,
        }
        self.generation_kwargs = OmegaConf.to_container(generation_kwargs)
        self.metric_callback = metric_callback
        self.disable_wandb = disable_wandb

    def get_log_dir(self, pl_module):
        if isinstance(pl_module.logger, WandbLogger):
            return pl_module.logger.experiment.dir
        elif isinstance(pl_module.logger, TensorBoardLogger):
            return pl_module.logger.log_dir

    def logger_log_image(self, pl_module, captions, images, global_step, split):
        if isinstance(pl_module.logger, WandbLogger) and not self.disable_wandb:
            pl_module.logger.log_image(
                key=f"{split}_img/{global_step}",
                images=images,
                caption=captions,
                step=global_step,
            )
        elif isinstance(pl_module.logger, TensorBoardLogger):
            big_grid = make_grid(
                torch.stack(images),
                padding=3,
                pad_value=50,
            )
            pl_module.logger.experiment.add_image(
                f"{split}_img",
                big_grid,
                global_step=global_step,
            )
            pl_module.logger.experiment.add_text(
                f"{split}_img_caption",
                " | ".join(captions),
                global_step=global_step,
            )

    @rank_zero_only
    def save_image(self, pl_module, images, global_step, split):
        print(f"Log images at: {split}/{global_step}")
        all_image_grids = []
        all_captions = []
        for k in images:
            grid = make_grid(images[k])
            all_captions.append(f"{k}")
            all_image_grids.append(grid)

        path = os.path.join(
            self.get_log_dir(pl_module), split + "_img", str(global_step)
        )
        os.makedirs(path, exist_ok=True)
        for caption, grid in zip(all_captions, all_image_grids):
            img = ToPILImage()(grid)
            img.save(os.path.join(path, caption + ".png"))

        self.logger_log_image(
            pl_module, all_captions,
            all_image_grids, global_step, split
        )

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (
            self.check_frequency(batch_idx, split)
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
        ):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                generation_samples = pl_module.log_images(
                    batch,
                    generation_kwargs=self.generation_kwargs,
                    stage=split,
                    cat_gt=True,
                )
            self.save_image(pl_module, generation_samples,
                            pl_module.global_step, split)
            if is_train:
                pl_module.train()

            return generation_samples  # we may use it in the future

    def check_frequency(self, batch_idx, split="train"):
        if split == "train":
            if ((batch_idx + 1) % self.batch_freq[split]) == 0:  # avoid batch 0
                return True
        else:
            if (batch_idx % self.batch_freq[split.split("_")[0]]) == 0:
                return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # make sure the random seed is same
        self.generation_kwargs["generator"] = torch.Generator(
            device=pl_module.device
        ).manual_seed(self.generation_kwargs["seed"])
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloadr_idx
    ):
        if trainer.state.stage == "sanity_check":
            return
        # make sure the random seed is same
        self.generation_kwargs["generator"] = torch.Generator(
            device=pl_module.device
        ).manual_seed(self.generation_kwargs["seed"])
        self.log_img(pl_module, batch, batch_idx, split="valid")


#########################################################################
# Utility code
#########################################################################
@torch.no_grad()
def expand_hidden_states(a: torch.Tensor, num_sample_per_image=1):
    origin_size = a.shape
    repeat_size = [1] * len(origin_size)
    repeat_size[1] = num_sample_per_image
    a = a.repeat(repeat_size)
    a = a.view(origin_size[0] * num_sample_per_image, *origin_size[1:])
    return a
