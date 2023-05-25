from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

VALID_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp"]

# Latent Scale Factor - https://github.com/huggingface/diffusers/issues/437
L_SCALE_FACTOR = 0.18215


class TrainerOpts(BaseModel):
    model_path: str = Field(...)
    dataset_path: Path = Field(...)
    vae: Optional[str] = None
    run_name: str = "stable_diffusion"
    output_path: Path = Field(...)
    use_wandb: bool = False
    hf_token: Optional[str] = None

    save_steps: int = 1000
    resume_steps: int = -1
    shuffle: bool = True
    reshuffle_tags: bool = False
    chunked_tag_shuffle: int = 0

    sample_steps: int = 200
    sample_count: int = 4
    sample_prompt: Optional[str] = None
    uncond_sample_prompt: Optional[str] = None

    epochs: int = 10
    seed: int = 42
    batch_size: int = 1
    use_ema: bool = False
    ucg: float = 0.1
    partial_dropout: bool = True

    mixed_precision: str = "no"
    enable_tf32: bool = True
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1

    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8

    lr: float = 5e-7
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    lr_num_cycles: int = 1
    lr_min_scale: float = 0.0
    lr_max_scale: float = 1.0

    train_te: bool = False
    te_lr: float = 7e-9
    extended_mode_chunks: int = 0
    clip_penultimate: bool = False
    log_loss_ema: bool = False

    xformers: bool = False
    debug: bool = False
