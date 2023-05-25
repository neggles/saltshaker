import copy
import gc
import io
import itertools
import json
import math
import random
import sys
import time
from os import PathLike
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EMAModel,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL.PngImagePlugin import PngInfo
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from saltshaker.settings import TrainSettings


class StableDiffusionTrainer:
    def __init__(
        self,
        accelerator: Accelerator,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        train_dataloader: DataLoader,
        noise_scheduler: DDPMScheduler,
        lr_scheduler: LambdaLR,
        optimizer: Optimizer,
        weight_dtype: torch.dtype,
        config: TrainSettings,
        ema: Optional[EMAModel] = None,
    ):
        self.accelerator = accelerator
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.ema = ema
        self.train_dataloader = train_dataloader
        self.noise_scheduler = noise_scheduler
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.weight_dtype = weight_dtype

        self.config = config
        self.config.output_path = Path(self.config.output_path)

        if accelerator.is_main_process:
            self.progress_bar = tqdm.tqdm(
                range(self.config.epochs * len(self.train_dataloader)),
                desc="Total Steps",
                leave=False,
                dynamic_ncols=True,
                smoothing=min(0.3, 20.0 / len(self.train_dataloader)),
            )

        class Logger:
            def __init__(self, outpath: PathLike, run_name: str):
                if not isinstance(outpath, Path):
                    outpath = Path(outpath)
                self.run_name = run_name
                self.logfile = outpath.joinpath(f"train_{run_name}.log")
                self.fp = outpath.open("w")

            def log(self, data, step: Optional[int] = None):
                if "images" in data.keys():
                    image_dir = self.logfile.parent.joinpath(self.run_name, "images", f"step_{step}")
                    image_dir.mkdir(parents=True, exist_ok=True)

                    index = 0
                    for image, caption in data["images"]:
                        metadata = PngInfo()
                        metadata.add_text("SD_TRAINING_RUN", config.run_name)
                        metadata.add_text("SD_PROMPT", caption)
                        metadata.add_text("SD_STEP_NUMBER", str(step))

                        image_path = image_dir.joinpath(f"sample_{index}.png")
                        image.save(image_path, "PNG", pnginfo=metadata)
                        index += 1
                else:
                    self.fp.write(f"[step {step}]: " if step else "[log]: ")
                    self.fp.write(json.dumps(data))
                    self.fp.write("\n")
                    self.fp.flush()

            def close(self) -> None:
                self.fp.close()
                self.fp = None

        self.run = Logger(self.config.output_path)
        self.global_step = 0

    def save_checkpoint(self):
        unet = self.accelerator.unwrap_model(self.unet)
        text_encoder = self.text_encoder
        if self.config.train_te:
            text_encoder = self.accelerator.unwrap_model(self.text_encoder)
        if self.config.use_ema:
            self.ema.store(unet.parameters())
            self.ema.copy_to(unet.parameters())
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=self.vae,
            unet=unet,
            tokenizer=self.tokenizer,
            scheduler=PNDMScheduler.from_pretrained(self.config.model_name_or_path, subfolder="scheduler"),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        print(f"Saving model (step: {self.global_step})...")
        pipeline.save_pretrained(
            self.config.output_path.joinpath(self.config.run_name, "checkpoints", f"step_{self.global_step}"),
            safe_serialization=True,
        )
        if self.config.use_ema:
            self.ema.restore(unet.parameters())
        del pipeline
        gc.collect()

    def sample(self, prompt: str) -> None:
        # get prompt from random batch
        text_encoder = self.text_encoder
        if self.config.train_te:
            text_encoder = self.accelerator.unwrap_model(self.text_encoder)

        def fake_safety_checker(clip_input, images):
            return images, []

        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=self.vae,
            unet=self.accelerator.unwrap_model(self.unet),
            tokenizer=self.tokenizer,
            scheduler=PNDMScheduler.from_pretrained(
                self.config.model_name_or_path,
                subfolder="scheduler",
            ),
            safety_checker=fake_safety_checker,  # don't load the real safety checker to save memory
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        ).to(self.accelerator.device)
        # inference
        with torch.no_grad():
            with torch.autocast("cuda", enabled=self.config.fp16):
                images = ((pipeline(prompt).images[0], prompt) for _ in range(self.config.sample_count))
                # log images under single caption
                self.run.log({"images": images}, step=self.global_step)

        # cleanup so we don't run out of memory
        del pipeline
        gc.collect()

    def encode(self, captions) -> torch.Tensor:
        # id rather die than refactor this code
        if self.config.extended_mode_chunks < 2:
            max_length = self.tokenizer.model_max_length - 2
            input_ids = [
                self.tokenizer(
                    [example],
                    truncation=True,
                    return_length=True,
                    return_overflowing_tokens=False,
                    padding=False,
                    add_special_tokens=False,
                    max_length=max_length,
                ).input_ids
                for example in captions
                if example is not None
            ]
        else:
            max_length = self.tokenizer.model_max_length
            max_chunks = self.config.extended_mode_chunks
            input_ids = [
                self.tokenizer(
                    [example],
                    truncation=True,
                    return_length=True,
                    return_overflowing_tokens=False,
                    padding=False,
                    add_special_tokens=False,
                    max_length=(max_length * max_chunks) - (max_chunks * 2),
                ).input_ids[0]
                for example in captions
                if example is not None
            ]

        text_encoder = (
            self.text_encoder
            if not self.config.train_te
            else self.accelerator.unwrap_model(self.text_encoder)
        )

        if self.config.extended_mode_chunks < 2:
            attn = copy.deepcopy(input_ids)
            for i, x in enumerate(input_ids):
                for j, y in enumerate(x):
                    input_ids[i][j] = [
                        self.tokenizer.bos_token_id,
                        *y,
                        *np.full(
                            (min(self.tokenizer.model_max_length - len(y) - 1, 1)),
                            self.tokenizer.eos_token_id,
                        ),
                        *np.full(
                            (max(self.tokenizer.model_max_length - len(y) - 2, 0)),
                            self.tokenizer.pad_token_id,
                        ),
                    ]
                    attn[i][j] = [
                        *np.full(len(y) + 2, 1),
                        *np.full(self.tokenizer.model_max_length - len(y) - 2, 0),
                    ]

            if self.config.clip_penultimate:
                input_ids = [
                    text_encoder.text_model.final_layer_norm(
                        text_encoder(
                            torch.asarray(input_id).to(self.accelerator.device),
                            output_hidden_states=True,
                            attention_mask=torch.asarray(attn).to(self.accelerator.device),
                        )["hidden_states"][-2]
                    )[0]
                    for (input_id, attn) in zip(input_ids, attn)
                ]
            else:
                input_ids = [
                    text_encoder(
                        torch.asarray(input_id).to(self.accelerator.device),
                        output_hidden_states=True,
                        attention_mask=torch.asarray(attn).to(self.accelerator.device),
                    ).last_hidden_state[0]
                    for (input_id, attn) in zip(input_ids, attn)
                ]
        else:
            max_standard_tokens = max_length - 2
            max_chunks = self.config.extended_mode_chunks
            max_len = (
                np.ceil(max(len(x) for x in input_ids) / max_standard_tokens).astype(int).item()
                * max_standard_tokens
            )
            if max_len > max_standard_tokens:
                z = None
                for i, x in enumerate(input_ids):
                    if len(x) < max_len:
                        input_ids[i] = [
                            *x,
                            *np.full(min(max_len - len(x), 1), self.tokenizer.eos_token_id),
                            *np.full(max(max_len - len(x) - 1, 0), self.tokenizer.pad_token_id),
                        ]
                batch_t = torch.tensor(input_ids)
                chunks = [
                    batch_t[:, i : i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)
                ]
                for chunk in chunks:
                    chunk = torch.cat(
                        (
                            torch.full((chunk.shape[0], 1), self.tokenizer.bos_token_id),
                            chunk,
                            torch.full((chunk.shape[0], 1), self.tokenizer.pad_token_id),
                        ),
                        1,
                    )
                    attn = torch.asarray(
                        [
                            list(
                                map(
                                    lambda x: 0 if x.detach().item() == self.tokenizer.pad_token_id else 1,
                                    [x for x in sc],
                                )
                            )
                            for sc in chunk
                        ]
                    )
                    if z is None:
                        if self.config.clip_penultimate:
                            z = text_encoder.text_model.final_layer_norm(
                                text_encoder(
                                    chunk.to(self.accelerator.device),
                                    output_hidden_states=True,
                                    attention_mask=torch.asarray(attn).to(self.accelerator.device),
                                )["hidden_states"][-2]
                            )
                        else:
                            z = text_encoder(
                                chunk.to(self.accelerator.device),
                                output_hidden_states=True,
                                attention_mask=torch.asarray(attn).to(self.accelerator.device),
                            ).last_hidden_state
                    else:
                        if self.config.clip_penultimate:
                            z = torch.cat(
                                (
                                    z,
                                    text_encoder.text_model.final_layer_norm(
                                        text_encoder(
                                            chunk.to(self.accelerator.device),
                                            output_hidden_states=True,
                                            attention_mask=torch.asarray(attn).to(self.accelerator.device),
                                        )["hidden_states"][-2]
                                    ),
                                ),
                                dim=-2,
                            )
                        else:
                            z = torch.cat(
                                (
                                    z,
                                    text_encoder(
                                        chunk.to(self.accelerator.device),
                                        output_hidden_states=True,
                                        attention_mask=torch.asarray(attn).to(self.accelerator.device),
                                    ).last_hidden_state,
                                ),
                                dim=-2,
                            )
                input_ids = z
            else:
                attn = copy.deepcopy(input_ids)
                for i, x in enumerate(input_ids):
                    input_ids[i] = [
                        self.tokenizer.bos_token_id,
                        *x,
                        *np.full(
                            (min(self.tokenizer.model_max_length - len(x) - 1, 1)),
                            self.tokenizer.eos_token_id,
                        ),
                        *np.full(
                            (max(self.tokenizer.model_max_length - len(x) - 2, 0)),
                            self.tokenizer.pad_token_id,
                        ),
                    ]
                    attn[i] = [
                        *np.full(len(x) + 2, 1),
                        *np.full(self.tokenizer.model_max_length - len(x) - 2, 0),
                    ]
                if self.config.clip_penultimate:
                    input_ids = text_encoder.text_model.final_layer_norm(
                        text_encoder(
                            torch.asarray(input_ids).to(self.accelerator.device),
                            output_hidden_states=True,
                            attention_mask=torch.asarray(attn).to(self.accelerator.device),
                        )["hidden_states"][-2]
                    )
                else:
                    input_ids = text_encoder(
                        torch.asarray(input_ids).to(self.accelerator.device),
                        output_hidden_states=True,
                        attention_mask=torch.asarray(attn).to(self.accelerator.device),
                    ).last_hidden_state
        return torch.stack(tuple(input_ids))

    def backprop(self, loss):
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            if self.config.train_te:
                self.accelerator.clip_grad_norm_(
                    itertools.chain(self.unet.parameters(), self.text_encoder.parameters()), 0.7071
                )
            else:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def sub_step(self, batch: dict, epoch: int) -> torch.Tensor:
        latents = list(map(lambda x: torch.load(io.BytesIO(x)), batch["latents"]))
        # Make sure we have latents of all the same size
        # (this should be true unless there is a db or preprocessing error)
        latent_sizes = {}
        for idx, lat in enumerate(latents):
            if lat.size() in latent_sizes:
                latent_sizes[lat.size()] = (idx, latent_sizes[lat.size()][1] + 1)
            else:
                latent_sizes[lat.size()] = (idx, 1)
        largest_latent = max(list(latent_sizes.items()), key=lambda x: x[1][1])[1][0]

        for idx, lat in enumerate(latents):
            if lat.size() != latents[largest_latent].size():
                print(
                    f"ERROR: Uneven latent size found at step {self.global_step} ({lat.size()} ->"
                    " {latents[largest_latent].size()})! Replacing..."
                )

                latents[idx] = latents[largest_latent].clone()
                batch["captions"][idx] = batch["captions"][largest_latent]

        # Finally stack our latents of the same guaranteed size
        latents = torch.stack(latents).to(self.accelerator.device, dtype=torch.float32)

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Encode captions with respect to extended mode and penultimate options
        encoder_hidden_states = self.encode(batch["captions"])

        # Predict the noise residual and compute loss
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Pew pew
        if self.noise_scheduler.config["prediction_type"] == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Invalid prediction type: {self.noise_scheduler.config.prediction_type}")

        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        self.backprop(loss)

        del latents
        return self.accelerator.gather_for_metrics(loss).mean()

    def step(self, batch: dict, epoch: int) -> dict:
        with self.accelerator.accumulate(self.unet), self.accelerator.autocast():
            loss = self.sub_step(batch, epoch)
        if self.accelerator.sync_gradients:
            # Update EMA
            if self.config.use_ema:
                self.ema.step(self.unet.parameters())

        return {
            "train/loss": loss.detach().item(),
            "train/lr": self.lr_scheduler.get_last_lr()[0],
        }

    def train(self) -> None:
        self.logs = {"train/loss": 0.33}
        ema_loss_decay = min(1, 20.0 / len(self.train_dataloader))
        if not self.config.log_loss_ema:
            ema_loss_decay = 1
        for epoch in range(self.config.epochs):
            self.unet.train()
            if self.config.train_te:
                self.text_encoder.train()
            for _, batch in enumerate(self.train_dataloader):
                step_start = time.perf_counter()
                if self.global_step < self.config.resume_steps:
                    if self.accelerator.is_main_process:
                        self.progress_bar.update(1)

                    self.global_step += 1
                    self.lr_scheduler.step()
                    del batch["captions"]
                    del batch["latents"]
                    del batch
                    continue

                logs = self.step(batch, epoch)

                self.global_step += 1

                if self.accelerator.is_main_process:
                    rank_samples_per_second = self.config.batch_size * (
                        1 / (time.perf_counter() - step_start)
                    )
                    world_samples_per_second = rank_samples_per_second * self.accelerator.num_processes
                    logs.update(
                        {
                            "perf/rank_sps": rank_samples_per_second,
                            "perf/world_sps": world_samples_per_second,
                            "train/epoch": epoch,
                            "train/step": self.global_step,
                            "train/samples_seen": self.global_step
                            * self.accelerator.num_processes
                            * self.config.batch_size,
                        }
                    )
                    # smooth loss over 5% of an epoch

                    ema_loss = (
                        self.logs["train/loss"] * (1 - ema_loss_decay) + logs["train/loss"] * ema_loss_decay
                    )
                    self.logs.update(logs)
                    self.logs["train/loss"] = ema_loss

                    # Output GPU RAM to flush tqdm
                    if not hasattr(self, "report_idx"):
                        self.report_idx = 1
                    else:
                        self.report_idx += 1
                    if self.report_idx % 10 == 0:
                        self.run.log(self.logs, step=self.global_step)
                    if self.report_idx % 1000 == 0:
                        self.accelerator.print(f"\nLOSS: {ema_loss}", file=sys.stderr)
                        sys.stderr.flush()

                    self.progress_bar.update(1)
                    self.progress_bar.set_postfix(**self.logs)

                if not (self.global_step % self.config.save_steps):
                    self.accelerator.wait_for_everyone()
                    self.save_checkpoint()

                if not (self.global_step % self.config.sample_steps):
                    if self.config.validation_prompts is None or self.config.validation_prompts == "":
                        prompt = batch["captions"][random.randint(0, len(batch["captions"]) - 1)]
                        if len(prompt) == 0 and self.config.uncond_validation_prompts:
                            prompt = self.config.uncond_validation_prompts
                    else:
                        prompt = self.config.validation_prompts
                    self.sample(prompt)

                self.accelerator.log(self.logs)

                del batch["captions"]
                del batch["latents"]
                del batch

        self.accelerator.wait_for_everyone()
        self.save_checkpoint()
        self.run.close()


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
