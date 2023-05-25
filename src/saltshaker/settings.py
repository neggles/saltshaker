import json
import logging
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from pydantic import BaseConfig, BaseModel, BaseSettings, Field, validator
from pydantic.env_settings import (
    EnvSettingsSource,
    InitSettingsSource,
    SecretsSettingsSource,
    SettingsSourceCallable,
)

from saltshaker.shared import CONFIG_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JsonSettingsSource:
    __slots__ = ("json_config_file",)

    def __init__(self, json_config_file: Optional[PathLike] = None) -> None:
        self.json_config_file: Optional[Path] = (
            Path(json_config_file) if json_config_file is not None else None
        )

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:  # noqa C901
        classname = settings.__class__.__name__
        encoding = settings.__config__.env_file_encoding

        if self.json_config_file.exists() and self.json_config_file.is_file():
            logger.info(f"Loading {classname} config from path: {self.json_config_file}")
            return json.loads(self.json_config_file.read_text(encoding=encoding))
        logger.warning(f"No {classname} config found at {self.json_config_file}")
        return {}

    def __repr__(self) -> str:
        return f"JsonSettingsSource(json_config_file={self.json_config_file!r})"


class JsonConfig(BaseConfig):
    json_config_file: Optional[Path] = None
    env_file_encoding = "utf-8"

    @classmethod
    def customise_sources(
        cls,
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> Tuple[SettingsSourceCallable, ...]:
        # pull json_config_file from init_settings if passed, otherwise use the class var
        json_config_file = init_settings.init_kwargs.pop("json_config_file", cls.json_config_file)
        # create a JsonSettingsSource with the json_config_file (which may be None)
        json_settings = JsonSettingsSource(json_config_file=json_config_file)
        # return the new settings sources
        return (
            init_settings,
            env_settings,
            json_settings,
            file_secret_settings,
        )


class TrainSettings(BaseSettings):
    model_name_or_path: Union[str, PathLike] = Field(...)
    revision: Optional[str] = None
    dataset_name_or_path: Union[str, PathLike] = Field(...)
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
    validation_prompts: Optional[List[str]] = None
    uncond_validation_prompts: Optional[List[str]] = None

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


class BucketSettings(BaseModel):
    buckets: List[Tuple[int, int]] = Field(...)
    ratios: List[float] = Field(...)

    def __iter__(self) -> Generator[Tuple[int, int, float], Any, None]:
        for bucket, ratio in zip(self.buckets, self.ratios):
            yield bucket[0], bucket[1], ratio


class AdamSettings(BaseModel):
    use_8bit: bool = Field(False)
    beta1: float = Field(0.9)
    beta2: float = Field(0.999)
    epsilon: float = Field(1e-8)
    weight_decay: float = Field(1e-2)


class SchedulerSettings(BaseModel):
    type: str = Field("constant")
    warmup_steps: int = Field(0)
    num_cycles: int = Field(1)
    min_scale: float = Field(0.0)
    max_scale: float = Field(1.0)
    auto_scale: bool = Field(False, description="Auto-scale the learning rate by the number of workers")


class Settings(BaseSettings):
    # Project config
    project_name: str = "saltshaker"
    project_dir: Path = Field(...)
    project_trackers: List[str] = ["wandb"]
    cache_dir: Optional[Path] = None  # defaults to HF cache dir
    output_dir: Optional[Path] = None  # defaults to project_dir / "output"
    # optionally sync model checkpoints to HF hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None

    # Accelerator config
    mixed_precision: str = "no"
    use_tpu: bool = False
    downcast_bf16: bool = False
    local_rank: int = -1

    # Model config
    model_name_or_path: Union[Path, str] = Field(...)
    model_revision: Optional[str] = None
    vae_name_or_path: Optional[Union[Path, str]] = None
    vae_revision: Optional[str] = None
    use_ema: bool = False
    ema_revision: Optional[str] = None

    # Dataset/processor config
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataloader_num_workers: int = Field(4, description="Number of workers for the dataloader")
    train_data_dir: Optional[Path] = None
    image_column: str = "image"
    caption_column: str = "caption"
    flip: bool = Field(False, description="Randomly flip some images horizontally")
    aspect: BucketSettings = Field(...)

    # Training config
    epochs: int = 10
    seed: int = Field(42, description="Random seed")
    resolution: int = 768
    batch_size: int = 1
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = Field(1.0, description="Max gradient norm for gradient clipping")

    unet_lr: float = Field(1e-5, description="Learning rate for UNet")
    train_te: bool = Field(False, description="Whether to train the Text Encoder")
    te_lr: float = Field(7e-9, description="Learning rate for TE")
    scheduler: SchedulerSettings = Field(..., description="Learning rate scheduler")
    optimizer: AdamSettings = Field(..., description="AdamW optimizer settings")

    max_train_samples: int = Field(-1, description="Cut off training after this many samples (-1 = no limit)")
    validation_prompts: List[str] = Field([], description="Prompts to use for validation")

    # saving and logging
    checkpoint_steps: int = Field(500, description="Save state checkpoint every N steps")
    checkpoint_limit: Optional[int] = Field(None, description="Maximum number of state checkpoints to keep")
    resume_from_checkpoint: Optional[Path] = Field(
        None, description="Resume training from a saved state checkpoint"
    )
    resume_from_steps: int = Field(-1, description="Resume at this step (-1 = no resume)")

    save_steps: int = Field(1000, description="Save the pipeline every N steps")
    keep_steps: int = Field(-1, description="Keep the last N steps of pipeline checkpoints (-1 for all)")
    save_epochs: int = Field(1, description="Save the pipeline every N epochs")
    keep_epochs: int = Field(-1, description="Keep the last N epochs of pipeline checkpoints (-1 for all)")
    save_last: bool = Field(True, description="Save the last pipeline checkpoint")

    class Config(JsonConfig):
        json_config_file = CONFIG_PATH


@lru_cache(maxsize=8)
def get_settings(config_path: Optional[PathLike] = None) -> Settings:
    settings = Settings(json_config_file=config_path)
    return settings
