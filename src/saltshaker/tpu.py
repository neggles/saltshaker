import json
import logging
import pickle
import random
import socket
import warnings
from os import getenv
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
from saltshaker.settings import TrainSettings
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


def reshuffle(data, chunked_tag_shuffle: int = 0, reshuffle_tags: bool = True):
    if reshuffle_tags:
        if chunked_tag_shuffle == 0:
            copy_data = []
            copy_data[:] = data
            random.shuffle(copy_data)
            return copy_data
        else:
            d_head = data[:chunked_tag_shuffle]
            d_tail = data[chunked_tag_shuffle:]
            random.shuffle(d_head)
            random.shuffle(d_tail)
            return d_head + d_tail
    return data


def drop_random(data, partial_dropout: bool = False):
    if random.random() > config.ucg:
        if partial_dropout:
            # the equation https://www.desmos.com/calculator/yrfoynzfcp is used
            # to keep a random percent of the data, where the random number is the x-axis
            x = random.randint(0, 100)
            if x >= 50:
                return ", ".join(reshuffle(data))
            else:
                return ", ".join(reshuffle(data[: len(data) * x * 2 // 100]))
        return ", ".join(reshuffle(data))
    else:
        # drop for unconditional guidance
        return ""


def collate_fn(examples):
    return_dict = {
        "latents": [example[0] for example in examples],
        "captions": [drop_random(example[1]) for example in examples],
        "source_name": [example[2] for example in examples],
        "source_id": [example[3] for example in examples],
    }
    return return_dict


def main(config: TrainSettings) -> None:
    if config.hf_token is None:
        config.hf_token = getenv("HF_TOKEN", None)
    if config.hf_token is None:
        raise ValueError("You need to supply a HuggingFace token via --hf-token or HF_TOKEN env variable")

    config.dataset_name_or_path = Path(config.dataset_name_or_path)
    config.output_path = Path(config.output_path)

    # Set up debug logging as early as possible
    if config.debug is True:
        logging.basicConfig(level=logging.DEBUG)
        transformers.logging.set_verbosity_debug()
        datasets.logging.set_verbosity_debug()
        diffusers.logging.set_verbosity_debug()
    else:
        logging.basicConfig(level=logging.INFO)

    # get device
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        even_batches=False,
        log_with="wandb" if config.use_wandb else None,
        project_dir=config.output_path.joinpath(config.run_name),
    )

    # Set seed
    accelerate.utils.set_seed(config.seed)

    # make output dir
    config.output_path.joinpath(config.run_name, "checkpoints").mkdir(parents=True, exist_ok=True)

    # Inform the user of host, and various versions -- useful for debugging issues.
    accelerator.print("RUN_NAME:", config.run_name)
    accelerator.print("HOST:", socket.gethostname())
    accelerator.print("CUDA:", torch.version.cuda)
    accelerator.print("TORCH:", torch.__version__)
    accelerator.print("TRANSFORMERS:", transformers.__version__)
    accelerator.print("DIFFUSERS:", diffusers.__version__)
    accelerator.print("MODEL:", config.model_name_or_path)
    accelerator.print("MIXED PRECISION:", config.mixed_precision)
    accelerator.print("RANDOM SEED:", config.seed)

    # load tokenizer and text encoder
    with accelerator.main_process_first():
        tokenizer = CLIPTokenizer.from_pretrained(
            config.model_name_or_path, subfolder="tokenizer", use_auth_token=config.hf_token
        )
        text_encoder = CLIPTextModel.from_pretrained(
            config.model_name_or_path, subfolder="text_encoder", use_auth_token=config.hf_token
        )

        # load VAE
        if config.vae is None:
            vae = AutoencoderKL.from_pretrained(
                config.model_name_or_path, subfolder="vae", use_auth_token=config.hf_token
            )
        else:
            vae = AutoencoderKL.from_pretrained(config.vae, use_auth_token=config.hf_token)
            accelerator.print("VAE:", config.vae)

        # load unet
        unet = UNet2DConditionModel.from_pretrained(
            config.model_name_or_path, subfolder="unet", use_auth_token=config.hf_token
        )

        # Freeze vae and (maybe) text_encoder
        vae.requires_grad_(False)
        if not config.train_te:
            text_encoder.requires_grad_(False)

        # load optimizer
        if config.train_te:
            opt_params = [
                {"params": unet.parameters()},
                {"params": text_encoder.parameters(), "lr": config.te_lr},
            ]
        else:
            opt_params = unet.parameters()

        optimizer = AdamW(
            opt_params,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.adam_weight_decay,
        )

        # load scheduler
        noise_scheduler = DDPMScheduler.from_pretrained(
            config.model_name_or_path, subfolder="scheduler", use_auth_token=config.hf_token
        )

    with accelerator.main_process_first():
        with config.dataset_name_or_path.open() as f:
            bucket: AspectBucket = pickle.load(f)

        dataset = AspectDataset(bucket)
        sampler = AspectBucketSampler(bucket=bucket, dataset=dataset)

    accelerator.print(f"Loaded {len(dataset)} images from bucket.")
    accelerator.print(f"Total of {len(sampler)} batches found.")
    config.batch_size = bucket.batch_size
    accelerator.print(f"BATCH SIZE: {config.batch_size}")
    if config.resume_steps > 0:
        accelerator.print(f"RESUME: {config.resume_steps}")

    # prefetch_factor is 2 by default ->
    # 2 * num_workers batches will prefetch
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=10,
        collate_fn=collate_fn,
    )

    if config.lr_scheduler == "cosine_with_restarts":
        print("lr scheduler = cosine with restarts")
        lr_scheduler = get_cosine_restart_scheduler_scaled(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.epochs * len(train_dataloader),
            num_cycles=config.lr_num_cycles,
            max_scale=config.lr_max_scale,
            min_scale=config.lr_min_scale,
        )
    else:
        print(f"lr scheduler = {config.lr_scheduler}")
        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.epochs * len(train_dataloader),
        )

    if config.train_te is True:
        accelerator.print(f"Training text encoder with lr {config.te_lr}")
        accelerator.print(f"Training unet with lr {config.lr}")
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        accelerator.print(f"Training unet with lr {config.lr}")
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float16 if config.fp16 else torch.float32

    # create ema
    if config.use_ema:
        ema_unet = EMAModel(unet.parameters())

    trainer = StableDiffusionTrainer(
        accelerator=accelerator,
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        ema=ema_unet if config.use_ema else None,
        train_dataloader=train_dataloader,
        noise_scheduler=noise_scheduler,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        weight_dtype=weight_dtype,
        config=config,
    )
    trainer.train()

    accelerator.print("Training complete!")
    accelerator.wait_for_everyone()


@app.command()
def cli(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False),
):
    config_dict = json.loads(config_path.read_text())
    config = TrainSettings.parse_obj(config_dict)

    main(config)


if __name__ == "__main__":
    config = TrainSettings()

    main(config)
