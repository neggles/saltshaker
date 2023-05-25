import json
import logging
import pickle
import random
import socket
import warnings
from os import PathLike, getenv
from pathlib import Path

import accelerate
import datasets
import diffusers
import torch
import transformers
import typer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EMAModel,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from saltshaker.data.filedisk_loader import AspectBucket, AspectBucketSampler, AspectDataset
from saltshaker.settings import Settings, TrainSettings, get_settings
from saltshaker.trainer import StableDiffusionTrainer, get_cosine_restart_scheduler_scaled

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    from torch_xla.amp.syncfree import Adam, AdamW
except ImportError:
    torch_xla = None
    xm = None
    from torch.optim import Adam, AdamW

# Turn this off to avoid warnings about TypedStorage being deprecated, which we can't do anything about
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

if accelerate.utils.is_rich_available():
    from rich import print
    from rich.traceback import install as traceback_install

    traceback_install(show_locals=True, width=120, word_wrap=True)

# Create typer instance
app = typer.Typer()


@app.command()
def main(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to config file"),
):
    settings: Settings = get_settings(config_path)

    if settings.dataset_name is None and settings.train_data_dir is None:
        raise ValueError("Either dataset_name or train_data_dir must be set in config")

    pass
