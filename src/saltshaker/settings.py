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


class AspectBucketInfo(BaseModel):
    buckets: List[Tuple[int, int]] = Field(...)
    ratios: List[float] = Field(...)

    def __iter__(self) -> Generator[Tuple[int, int, float], Any, None]:
        for bucket, ratio in zip(self.buckets, self.ratios):
            yield bucket[0], bucket[1], ratio


class BucketSettings(BaseModel):
    num_buckets: int = 32
    bucket_side_min: int = 256
    bucket_side_max: int = 768 * 2
    bucket_max_area: int = 768 * 768


class AdamSettings(BaseModel):
    use_8bit: bool = Field(False, description="Use 8-bit Adam (CUDA only)")
    beta1: float = Field(0.9, description="Adam beta1")
    beta2: float = Field(0.999, description="Adam beta2")
    weight_decay: float = Field(1e-2, description="Adam weight decay")
    epsilon: float = Field(1e-8, description="Adam epsilon")


class SchedulerSettings(BaseModel):
    type: str = Field("constant")
    warmup_steps: float = Field(0.1, description="Fraction of total steps to warmup for (percentage)")
    num_cycles: int = Field(1, description="Number of restart cycles for cosine scheduler")
    min_scale: float = Field(0.0, description="Minimum learning rate scale")
    max_scale: float = Field(1.0, description="Maximum learning rate scale")
    auto_scale: bool = Field(False, description="Auto-scale the learning rate by the number of workers")


class Settings(BaseSettings):
    # Project config
    project_name: str = "saltshaker"
    project_dir: Path = Field(...)
    log_with: List[str] = ["wandb"]
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
    xformers: bool = Field(False, description="Use XFormers memory-efficient attention. GPU only")

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
    aspect: AspectBucketInfo = Field(...)
    shuffle_tags: bool = Field(True, description="Shuffle tags for each image")
    keep_tags: int = Field(0, description="Keep this many tags per image unshuffled")
    clip_penultimate: bool = Field(False, description="Clip the penultimate layer of the text encoder")
    extended_mode_chunks: int = Field(
        0,
        description="Enables tokenizer extended mode with N max 75-token chunks (<2 disables)",
    )
    ucg: float = Field(0.1, description="Percentage chance of dropping out the text condition per batch.")

    # Training config
    epochs: int = 10
    seed: int = Field(42, description="Random seed")
    resolution: int = 768
    batch_size: int = 1
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = Field(1.0, description="Max gradient norm for gradient clipping")

    unet_lr: float = Field(1e-5, description="Learning rate for UNet")
    train_text_encoder: bool = Field(False, description="Whether to train the Text Encoder")
    text_encoder_lr: float = Field(7e-9, description="Learning rate for TE")
    scheduler: SchedulerSettings = Field(..., description="Learning rate scheduler")
    optimizer: AdamSettings = Field(..., description="AdamW optimizer settings")

    max_train_samples: int = Field(-1, description="Cut off training after this many samples (-1 = no limit)")
    validation_prompts: List[str] = Field([], description="Prompts to use for validation")
    validation_steps: int = Field(1000, description="Run validation every N steps")
    validation_epochs: int = Field(1, description="Run validation every N epochs")

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
