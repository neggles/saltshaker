from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

CONFIG_PATH = Path.cwd().joinpath("data", "config.json")
BUCKET_PATH = Path.cwd().joinpath("data", "buckets.json")

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"]

# Latent Scale Factor - https://github.com/huggingface/diffusers/issues/437
L_SCALE_FACTOR = 0.18215
