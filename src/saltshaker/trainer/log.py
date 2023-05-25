import argparse
import logging
import math
import random
from operator import is_
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

from saltshaker.settings import TrainSettings

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


def log_validation(
    model_name_or_path: Union[str, PathLike],
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet: UNet2DConditionModel,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    epoch: int,
    settings: TrainSettings,
):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=settings.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.set_progress_bar_config(disable=True)

    if settings.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(settings.seed)

    images = []
    for i in range(len(settings.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(
                settings.validation_prompts[i], num_inference_steps=20, generator=generator
            ).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {settings.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
