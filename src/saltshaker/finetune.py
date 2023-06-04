import json
import logging
import pickle
import random
import socket
import warnings
from os import PathLike, getenv
from pathlib import Path
from sys import version_info
from typing import Any, Dict, List, Optional, Tuple, Union

import accelerate
import datasets
import diffusers
import torch
import transformers
import typer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EMAModel,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import create_repo
from numpy import isin
from regex import subf
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTokenizerFast

from saltshaker.data.dataset import AspectBucketDataset, AspectDatasetSampler
from saltshaker.settings import Settings, TrainSettings, get_settings
from saltshaker.shared import IMAGE_EXTENSIONS
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

# create logger
logger = get_logger(__name__, log_level="INFO")

# Create typer instance
app = typer.Typer()


@app.command()
def main(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to config file"),
):
    settings: Settings = get_settings(config_path)

    if settings.dataset_name is None and settings.train_data_dir is None:
        raise ValueError("Either dataset_name or train_data_dir must be set in config")

    project_config = ProjectConfiguration(
        project_dir=settings.project_dir,
        total_limit=settings.checkpoint_limit,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        mixed_precision=settings.mixed_precision,
        log_with=settings.log_with,
        project_config=project_config,
        downcast_bf16=settings.downcast_bf16,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Print some more environment information, since the original did
    accelerator.print(
        "\n".join(
            [
                "------------------------------------------------",
                f"Hostname: {socket.gethostname()}",
                f"Process ID: {accelerator.process_index} (of {accelerator.num_processes})",
                f"Local ID: {accelerator.local_process_index}",
                f"Is global main: {accelerator.is_main_process}",
                f"Is local main: {accelerator.is_local_main_process}",
                "------------------------------------------------",
                "Version table:",
                f"  Python: {version_info}",
                f"  PyTorch: {torch.__version__}",
                f"  Accelerate: {accelerate.__version__}",
                f"  Transformers: {transformers.__version__}",
                f"  Diffusers: {diffusers.__version__}",
                f"  Datasets: {datasets.__version__}",
                "------------------------------------------------",
                f"Model: {settings.model_name_or_path}",
                f"Output directory: {settings.output_dir}",
                f"Seed: {settings.seed}",
                f"Device: {accelerator.device}",
                f"Distributed type: {accelerator.state.distributed_type}",
                f"Mixed precision: {accelerator.state.mixed_precision}",
                "------------------------------------------------",
                f"Accelerator state dump: \n{accelerator.state}",
                "------------------------------------------------",
            ]
        )
    )
    if not accelerator.is_local_main_process:
        logger.info(accelerator.state, main_process_only=False),

    # Set seed before initializing model.
    if settings.seed is not None:
        logger.info(f"Setting seed: {settings.seed}")
        set_seed(settings.seed)

    # create output directory and optionally HF repo
    if accelerator.is_local_main_process:
        if settings.output_dir is not None:
            settings.output_dir.mkdir(parents=True, exist_ok=True)
        if settings.push_to_hub:
            settings.hub_model_id = create_repo(
                repo_id=settings.hub_model_id or Path(settings.output_dir).name,
                private=True,
                exist_ok=True,
                token=settings.hub_token,
            ).repo_id

    # Load on main process first so we don't download the files multiple times

    with accelerator.main_process_first():
        # Load the noise scheduler
        logger.info("Loading noise scheduler")
        noise_scheduler = DDPMScheduler.from_pretrained(settings.model_name_or_path, subfolder="scheduler")

    with accelerator.main_process_first():
        # load tokenizer
        logger.info("Loading tokenizer")
        tokenizer: CLIPTokenizerFast = CLIPTokenizerFast.from_pretrained(
            settings.model_name_or_path,
            subfolder="tokenizer",
            revision=settings.model_revision,
            cache_dir=settings.cache_dir,
        )

    with accelerator.main_process_first():
        # load TE
        logger.info("Loading text encoder")
        text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            settings.model_name_or_path,
            use_auth_token=True,
            subfolder="text_encoder",
            revision=settings.model_revision,
            cache_dir=settings.cache_dir,
        )

    with accelerator.main_process_first():
        # load VAE
        logger.info("Loading VAE")
        vae_rev = settings.vae_revision or settings.model_revision
        if settings.vae_name_or_path is not None:
            vae: AutoencoderKL = AutoencoderKL.from_pretrained(
                settings.vae_name_or_path, subfolder="vae", revision=vae_rev, cache_dir=settings.cache_dir
            )
        else:
            vae: AutoencoderKL = AutoencoderKL.from_pretrained(
                settings.model_name_or_path, subfolder="vae", revision=vae_rev, cache_dir=settings.cache_dir
            )

    with accelerator.main_process_first():
        # load UNet
        logger.info("Loading UNet")
        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            settings.model_name_or_path,
            subfolder="unet",
            revision=settings.model_revision,
            cache_dir=settings.cache_dir,
        )

    # wait for all processes to load their models
    accelerator.wait_for_everyone()

    # Freeze the VAE
    vae.requires_grad_(False)
    # if we're not training the TE, freeze it
    if not settings.train_text_encoder:
        text_encoder.requires_grad_(False)

    # enable gradient checkpointing if enabled
    if settings.gradient_checkpointing is True:
        logger.info("Enabling gradient checkpointing for unet")
        unet.enable_gradient_checkpointing()
        if settings.train_text_encoder is True:
            logger.info("Enabling gradient checkpointing for text encoder")
            text_encoder.gradient_checkpointing_enable()

    if settings.xformers is True and settings.use_tpu is True:
        logger.warning("XFormers is not supported on TPU, will not enable it")
    elif settings.xformers is True:
        logger.info("Enabling XFormers memory efficient attention for unet")
        unet.set_use_memory_efficient_attention_xformers(True)

    # Create EMA model for unet if enabled
    if settings.use_ema:
        with accelerator.main_process_first():
            logger.info("Enabling EMA for unet")
            ema_unet = UNet2DConditionModel.from_pretrained(
                settings.model_name_or_path,
                subfolder="unet",
                revision=settings.ema_revision or settings.model_revision,
            )
            ema_unet = EMAModel(
                ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config
            )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_state_hook(
        models: List[torch.nn.Module], weights: List[Dict[str, torch.Tensor]], input_dir: str
    ):
        output_dir = Path(input_dir)
        if settings.use_ema:
            ema_unet.save_pretrained(output_dir.joinpath("unet_ema"), safe_serialization=True)

        for model, state_dict in zip(models, weights):
            if isinstance(model, UNet2DConditionModel):
                model.save_pretrained(output_dir.joinpath("unet"), safe_serialization=True)
                weights.remove(state_dict)
            elif isinstance(model, CLIPTextModel) and settings.train_text_encoder:
                model.save_pretrained(output_dir.joinpath("text_encoder"), safe_serialization=True)
                weights.remove(state_dict)

    def load_state_hook(models: List[torch.nn.Module], input_dir: str):
        if settings.use_ema:
            load_model = EMAModel.from_pretrained(input_dir.joinpath("unet_ema"), UNet2DConditionModel)
            ema_unet.load_state_dict(load_model.state_dict())
            del load_model

        for model in models:
            if isinstance(model, UNet2DConditionModel):
                # pop it from the list so that it doesn't get loaded twice
                models.remove(model)
                # then load from diffusers-style checkpoint
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
            elif isinstance(model, CLIPTextModel) and settings.train_text_encoder:
                # pop it from the list so that it doesn't get loaded twice
                models.remove(model)
                # then load from diffusers-style checkpoint
                load_model = CLIPTextModel.from_pretrained(input_dir, subfolder="text_encoder")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

    # Register the hooks
    accelerator.register_save_state_pre_hook(save_state_hook)
    accelerator.register_load_state_pre_hook(load_state_hook)

    # scale LR if enabled
    if settings.scheduler.auto_scale is True:
        logger.info("Enabling automatic LR scaling for unet")
        settings.unet_lr = (
            settings.unet_lr
            * settings.gradient_accumulation_steps
            * settings.batch_size
            * accelerator.num_processes
        )
        if settings.train_text_encoder is True:
            logger.info("Enabling automatic LR scaling for text encoder")
            settings.text_encoder_lr = (
                settings.text_encoder_lr
                * settings.gradient_accumulation_steps
                * settings.batch_size
                * accelerator.num_processes
            )

    # create optimizer with separate LR for text encoder if enabled
    if settings.train_text_encoder:
        logger.info("Initializing optimizer with separate LR for text encoder")
        optimizer = AdamW(
            [
                {"params": unet.parameters()},
                {"params": text_encoder.parameters(), "lr": settings.text_encoder_lr},
            ],
            lr=settings.unet_lr,
            betas=(settings.optimizer.beta1, settings.optimizer.beta2),
            eps=settings.optimizer.epsilon,
            weight_decay=settings.optimizer.weight_decay,
        )
    else:
        logger.info("Initializing optimizer with single LR for unet")
        optimizer = AdamW(
            unet.parameters(),
            lr=settings.unet_lr,
            betas=(settings.optimizer.beta1, settings.optimizer.beta2),
            eps=settings.optimizer.epsilon,
            weight_decay=settings.optimizer.weight_decay,
        )

    # now to acquire dataset. we do this first on the main process so it only downloads once
    with accelerator.main_process_first():
        if settings.dataset_name is not None:
            logger.info(f"Loading HF dataset: {settings.dataset_name}", main_process_only=False)
            # HF dataset
            dataset = datasets.load_dataset(
                path=settings.dataset_name,
                name=settings.dataset_config,
                cache_dir=settings.cache_dir.parent.joinpath("datasets"),
            )["train"]
        elif settings.train_data_dir is not None:
            logger.info(f"Loading image folder dataset: {settings.train_data_dir}", main_process_only=False)
            dataset = datasets.load_dataset(
                "imagefolder",
                data_dir=settings.train_data_dir,
                cache_dir=settings.cache_dir.parent.joinpath("datasets"),
            )["train"]

    # get available column names
    column_names = dataset.column_names
    image_column = settings.image_column
    if image_column not in column_names:
        raise ValueError(f"Image column '{image_column}' not found in dataset columns: {column_names}")
    caption_column = settings.caption_column
    if caption_column not in column_names:
        raise ValueError(f"Caption column '{caption_column}' not found in dataset columns: {column_names}")

    # create dataset
    logger.info("Creating AspectBucketDataset from raw dataset")
    train_dataset = AspectBucketDataset(
        dataset=dataset,
        settings=settings,
        caption_column=caption_column,
        image_column=image_column,
    )
    train_sampler = AspectDatasetSampler(dataset=train_dataset)

    logger.info("Creating dataloader")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=settings.batch_size,
        num_workers=settings.dataloader_num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    # set up the scheduler
    if settings.scheduler.type == "cosine_with_restarts":
        logger.info("Using cosine with restarts scheduler")
        lr_scheduler = get_cosine_restart_scheduler_scaled(
            optimizer=optimizer,
            num_warmup_steps=settings.scheduler.warmup_steps,
            num_training_steps=settings.epochs * len(train_dataloader),
            num_cycles=settings.scheduler.num_cycles,
            max_scale=settings.scheduler.max_scale,
            min_scale=settings.scheduler.min_scale,
        )
    else:
        logger.info(f"Using {settings.scheduler.type} scheduler")
        lr_scheduler = get_scheduler(
            settings.scheduler.type,
            optimizer=optimizer,
            num_warmup_steps=settings.scheduler.warmup_steps,
            num_training_steps=settings.epochs * len(train_dataloader),
        )

    # Now we feed shit into the Accelerator to prepare it...
    logger.info("Preparing models, optimizer, dataloader, scheduler for distributed training...")
    if settings.train_text_encoder:
        unet, optimizer, text_encoder, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, text_encoder, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # Initialize trackers
    logger.info("Ready to create trainer, waiting for all processes so we can print debug info...")
    accelerator.wait_for_everyone()
    if accelerator.distributed_type == accelerate.DistributedType.TPU:
        proc_idx = accelerator.process_index + 1
        n_procs = accelerator.num_processes
        local_proc_idx = accelerator.local_process_index + 1
        xm_ord = xm.get_ordinal() + 1
        xm_world = xm.xrt_world_size()
        xm_local_ord = xm.get_local_ordinal() + 1
        xm_master_ord = xm.is_master_ordinal()
        is_main = accelerator.is_main_process
        is_local_main = accelerator.is_local_main_process

        with accelerator.local_main_process_first():
            logger.info(
                f"[P{proc_idx:03d}]: PID {proc_idx}/{n_procs}, local #{local_proc_idx}, "
                + f"XLA ord={xm_ord}/{xm_world}, local={xm_local_ord}, master={xm_master_ord} "
                + f"Accelerate: main={is_main}, local main={is_local_main} ",
                main_process_only=False,
            )
        accelerator.wait_for_everyone()

    # create trainer
    logger.info("Creating trainer")
    weight_dtype = torch.float16 if settings.mixed_precision == "fp16" else torch.float32
    trainer = StableDiffusionTrainer(
        accelerator=accelerator,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        noise_scheduler=noise_scheduler,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        weight_dtype=weight_dtype,
        settings=settings,
        ema=ema_unet if settings.use_ema else None,
    )

    logger.info("Initializing trackers")
    accelerator.init_trackers(config=settings)

    # do the thing
    logger.info("Ready to start! Waiting for all processes to be ready...")
    accelerator.wait_for_everyone()
    logger.info("Starting training!")
    trainer.train()
    logger.info("Training complete!")
    accelerator.end_training()

    pass
