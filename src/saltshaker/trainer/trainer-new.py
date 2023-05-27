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
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EMAModel,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from PIL.PngImagePlugin import PngInfo
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from saltshaker.settings import Settings

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    from torch_xla.amp.syncfree import Adam, AdamW
except ImportError:
    torch_xla = None
    xm = None
    from torch.optim import Adam, AdamW

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
        max_train_steps: int,
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
        self.max_train_steps = max_train_steps
        self.settings = settings
        self.ema_model = ema_model

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

    def encode(self, captions):
        # id rather die than refactor this code
        if self.settings.extended_mode_chunks < 2:
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
            max_chunks = self.settings.extended_mode_chunks
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
            if not self.settings.train_text_encoder
            else self.accelerator.unwrap_model(self.text_encoder)
        )

        if self.settings.extended_mode_chunks < 2:
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

            if self.settings.clip_penultimate:
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
            max_chunks = self.settings.extended_mode_chunks
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
                        if self.settings.clip_penultimate:
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
                        if self.settings.clip_penultimate:
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
                if self.settings.clip_penultimate:
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
            if self.settings.train_text_encoder:
                self.accelerator.clip_grad_norm_(
                    itertools.chain(self.unet.parameters(), self.text_encoder.parameters()), 0.7071
                )
            else:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def train(self):
        pass

    def train_step(self, batch: Tensor, step: int) -> Tensor:
        pass
