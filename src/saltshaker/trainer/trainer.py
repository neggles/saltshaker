import gc
import itertools
import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EMAModel,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup, get_scheduler
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from saltshaker.settings import Settings

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
except ImportError:
    torch_xla = None
    xm = None

logger = get_logger(__name__, log_level="INFO")


class StableDiffusionTrainer(nn.Module):
    def __init__(
        self,
        accelerator: Accelerator,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        train_dataloader: DataLoader,
        noise_scheduler: DDPMScheduler,
        lr_scheduler: LambdaLR,
        optimizer: Optimizer,
        weight_dtype: torch.dtype,
        settings: Settings,
        ema_model: Optional[EMAModel] = None,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.noise_scheduler = noise_scheduler
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.weight_dtype = weight_dtype

        self.settings = settings
        self.ema_model = ema_model

        self.max_train_steps = settings.max_train_samples // settings.batch_size
        self.train_text_encoder = settings.train_text_encoder
        self.extended_mode_chunks = settings.extended_mode_chunks
        self.clip_penultimate = settings.clip_penultimate
        self.device = accelerator.device

        # Set some flags
        self._use_ema = self.ema_model is not None
        self._v_prediction = self.noise_scheduler.config["prediction_type"] == "v_prediction"

        # and some locals
        self.global_step = 0
        self.first_epoch = 0

        # Set up the progress bar
        if accelerator.is_main_process:
            self.progress = tqdm.tqdm(
                range(self.global_step, self.max_train_steps),
                desc="Total Steps",
                leave=False,
                dynamic_ncols=True,
                smoothing=min(0.3, 20.0 / len(self.train_dataloader)),
            )

    def progress_text(self, val: dict, **kwargs) -> None:
        if self.accelerator.is_main_process:
            self.progress.set_postfix(val, **kwargs)
        if met is not None:
            logger.info(met.short_metrics_report())
            self.settings.output_dir.joinpath(f"metrics-{self.global_step}.txt").write_text(
                met.metrics_report()
            )

    def progress_step(self, val: int = 1) -> None:
        if self.accelerator.is_main_process:
            self.progress.update(val)

    def save_pipeline(self):
        unet = self.accelerator.unwrap_model(self.unet)
        text_encoder = (
            self.accelerator.unwrap_model(self.text_encoder) if self.train_text_encoder else self.text_encoder
        )
        if self.settings.use_ema:
            self.ema_model.store(unet.parameters())
            self.ema_model.copy_to(unet.parameters())
        pipeline: StableDiffusionPipeline = StableDiffusionPipeline(
            vae=self.vae,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=self.tokenizer,
            safety_checker=None,
            scheduler=PNDMScheduler.from_pretrained(self.settings.model_name_or_path, subfolder="scheduler"),
            feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            requires_safety_checker=False,
        )
        save_dir = self.settings.output_dir.joinpath(
            self.settings.project_name, "pipeline", f"step_{self.global_step}"
        )
        logger.info(
            f"Saving pipeline checkpoint for step {self.global_step} to"
            + f" output_dir/{save_dir.relative_to(self.settings.output_dir)}"
        )
        if self.accelerator.is_main_process:
            pipeline.save_pretrained(save_dir, safe_serialization=True)
        if self.settings.use_ema:
            self.ema_model.restore(unet.parameters())
        del pipeline
        gc.collect()

    def encode_text(self, examples):
        input_ids = [x for x in examples["tokens"]]

        if self.extended_mode_chunks < 2:
            for i, x in enumerate(input_ids):
                for j, y in enumerate(x):
                    input_ids[i][j] = [
                        self.tokenizer.bos_token_id,
                        *y,
                        *np.full((self.tokenizer.model_max_length - len(y) - 1), self.tokenizer.eos_token_id),
                    ]

            if self.clip_penultimate:
                input_ids = [
                    self.text_encoder.text_model.final_layer_norm(
                        self.text_encoder(
                            torch.asarray(input_id).to(self.device),
                            output_hidden_states=True,
                        )["hidden_states"][-2]
                    )[0]
                    for input_id in input_ids
                ]
            else:
                input_ids = [
                    self.text_encoder(
                        torch.asarray(input_id).to(self.device),
                        output_hidden_states=True,
                    ).last_hidden_state[0]
                    for input_id in input_ids
                ]
        else:
            max_length = self.tokenizer.model_max_length
            max_standard_tokens = max_length - 2
            max_chunks = self.extended_mode_chunks
            max_len = (
                np.ceil(max(len(x) for x in input_ids) / max_standard_tokens).astype(int).item()
                * max_standard_tokens
            )
            if max_len > max_standard_tokens:
                z = None
                for i, x in enumerate(input_ids):
                    if len(x) < max_len:
                        input_ids[i] = [*x, *np.full((max_len - len(x)), self.tokenizer.eos_token_id)]
                batch_t = torch.tensor(input_ids)
                chunks = [
                    batch_t[:, i : i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)
                ]
                for chunk in chunks:
                    chunk = torch.cat(
                        (
                            torch.full((chunk.shape[0], 1), self.tokenizer.bos_token_id),
                            chunk,
                            torch.full((chunk.shape[0], 1), self.tokenizer.eos_token_id),
                        ),
                        1,
                    )
                    if z is None:
                        if self.clip_penultimate:
                            z = self.text_encoder.text_model.final_layer_norm(
                                self.text_encoder(chunk.to(self.device), output_hidden_states=True)[
                                    "hidden_states"
                                ][-2]
                            )
                        else:
                            z = self.text_encoder(
                                chunk.to(self.device), output_hidden_states=True
                            ).last_hidden_state
                    else:
                        if self.clip_penultimate:
                            z = torch.cat(
                                (
                                    z,
                                    self.text_encoder.text_model.final_layer_norm(
                                        self.text_encoder(chunk.to(self.device), output_hidden_states=True)[
                                            "hidden_states"
                                        ][-2]
                                    ),
                                ),
                                dim=-2,
                            )
                        else:
                            z = torch.cat(
                                (
                                    z,
                                    self.text_encoder(
                                        chunk.to(self.device), output_hidden_states=True
                                    ).last_hidden_state,
                                ),
                                dim=-2,
                            )
                input_ids = z
            else:
                for i, x in enumerate(input_ids):
                    input_ids[i] = [
                        self.tokenizer.bos_token_id,
                        *x,
                        *np.full((self.tokenizer.model_max_length - len(x) - 1), self.tokenizer.eos_token_id),
                    ]
                if self.clip_penultimate:
                    input_ids = self.text_encoder.text_model.final_layer_norm(
                        self.text_encoder(
                            torch.asarray(input_ids).to(self.device), output_hidden_states=True
                        )["hidden_states"][-2]
                    )
                else:
                    input_ids = self.text_encoder(
                        torch.asarray(input_ids).to(self.device), output_hidden_states=True
                    ).last_hidden_state

        input_ids = torch.stack(tuple(input_ids))
        return input_ids

    def compute_snr(self, timesteps: Tensor) -> Tensor:
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def backprop(self, loss):
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            if self.train_text_encoder:
                self.accelerator.clip_grad_norm_(
                    itertools.chain(self.unet.parameters(), self.text_encoder.parameters()), 0.7071
                )
            else:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def train(self):
        for epoch in range(self.first_epoch, self.settings.epochs):
            self.unet.train()
            if self.train_text_encoder:
                self.text_encoder.train()
            for num, batch in enumerate(self.train_dataloader):
                if (
                    self.settings.resume_from_checkpoint is not None
                    and self.global_step < self.settings.resume_from_steps
                ):
                    self.progress_step()
                    self.global_step += 1
                    continue

                with self.accelerator.autocast():
                    # do the step and measure execution time
                    b_start = time.perf_counter()
                    with self.accelerator.accumulate(self.unet):
                        if self.train_text_encoder:
                            with self.accelerator.accumulate(self.text_encoder):
                                loss = self.train_step(batch)
                        else:
                            loss = self.train_step(batch)
                    b_end = time.perf_counter()

                    if self._use_ema:
                        self.ema_model.step(self.unet.parameters())

                    if self.accelerator.sync_gradients:
                        # compute metrics
                        seconds_per_step = b_end - b_start
                        steps_per_second = 1 / seconds_per_step
                        rank_images_per_second = self.settings.batch_size * steps_per_second
                        world_images_per_second = rank_images_per_second * self.accelerator.num_processes
                        samples_seen = (
                            self.global_step * self.settings.batch_size * self.accelerator.num_processes
                        )

                        # log metrics
                        train_loss = self.accelerator.gather_for_metrics(loss)
                        train_loss = train_loss.mean() / self.accelerator.gradient_accumulation_steps
                        train_lr = self.lr_scheduler.get_last_lr()
                        metrics = {
                            "train/loss": train_loss,
                            "train/lr": train_lr[0],
                            "train/epoch": epoch,
                            "train/step": self.global_step,
                            "train/samples_seen": samples_seen,
                            "perf/rank_sps": rank_images_per_second,
                            "perf/world_sps": world_images_per_second,
                        }
                        if len(train_lr) > 1:  # means we're training the text encoder
                            metrics["train/te_lr"] = train_lr[1]

                        # update progress bar and global step
                        self.progress_step()
                        self.global_step += 1
                        self.progress_text(metrics)
                        self.accelerator.log(metrics, step=self.global_step)

                # save checkpoint if needed
                if not (self.global_step % self.settings.checkpoint_steps):
                    self.accelerator.wait_for_everyone()
                    self.accelerator.save_state(
                        output_dir=self.settings.output_dir.joinpath(
                            "checkpoints", f"e{epoch}_step_{self.global_step}"
                        )
                    )

                # save pipeline if needed
                if not (self.global_step % self.settings.save_steps):
                    self.save_pipeline()

            # save state and pipeline at the end of each epoch
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(
                output_dir=self.settings.output_dir.joinpath("checkpoints", f"e{epoch}_step_final")
            )
            self.save_pipeline()

    def train_step(self, batch: Tensor) -> Tensor:
        # make latents out of the images
        pixel_values = batch["pixel_values"]
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents.mul(0.18215)

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the embedding for conditioning
        encoder_hidden_states = self.encode_text(batch)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # is this...
        loss = F.mse_loss(noise_pred, target, reduction="mean")

        # backprop and update
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.unet.parameters(), self.settings.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return loss


# get_cosine_with_hard_restarts_schedule_with_warmup_and_scaling
def get_cosine_restart_scheduler_scaled(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 1,
    last_epoch: int = -1,
    max_scale: float = 1.0,
    min_scale: float = 0.0,
) -> LambdaLR:
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        if progress >= 1.0:
            return 0.0
        return max(
            0.0,
            0.5
            * (
                max_scale
                + min_scale
                + (max_scale - min_scale) * math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))
            ),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def remap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def rescale(x, min_scale: float = 0.0, max_scale: float = 1.0) -> float:
    part1 = x
    part2 = max_scale - min_scale
    part3 = 1.0 - 0.0
    part4 = min_scale

    an_value = part1 * part2 / part3 + part4
    # dont need the div by 1
    an_value = part1 * part2 + part4

    an_value = (x * (max_scale - min_scale)) + min_scale

    pass
