from hashlib import sha256
from random import (
    random as rand_float,
    shuffle,
)
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset as HFDataset
from PIL import Image
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor
from tqdm_loggable.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from saltshaker.settings import AspectBucketInfo, Settings

logger = get_logger(__name__)


class AspectBucketDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        accelerator: Accelerator,
        settings: Settings,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.accelerator = accelerator

        self.image_column = settings.image_column
        self.caption_column = settings.caption_column
        self.ucg = settings.ucg
        self.extended_mode_chunks = settings.extended_mode_chunks
        self.clip_penultimate = settings.clip_penultimate

        if isinstance(self.text_encoder, torch.nn.parallel.DistributedDataParallel):
            self.text_encoder: CLIPTextModel = self.text_encoder.module

        self.transforms = Compose(
            [
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

        self.aspect: AspectBucketInfo = settings.aspect
        self.batch_size = settings.batch_size
        self.max_ratio = 2

        # bucket caching
        self._cache_dir = settings.cache_dir
        cache_fname = f"{settings.model_name_or_path.stem}-{settings.project_name}-cache"
        fname_hash = sha256(cache_fname.encode("utf-8")).hexdigest()[:8]
        world_info_str = f"w{self.accelerator.num_processes}p{self.accelerator.local_process_index}"
        self._cachefile_name = f"{cache_fname}-{fname_hash}-b{self.batch_size}-{world_info_str}.pt"
        self._loaded_cache = False

        # initialize the buckets
        self.buckets = self.aspect.buckets
        self._ratios = self.aspect.ratios
        self._interp = interp1d(
            self._ratios, list(range(len(self.buckets))), assume_sorted=True, fill_value=None
        )
        self.buckets = [tuple(x) for x in self.buckets]
        self.bucket_data: Dict[tuple, List[int]] = {b: [] for b in self.buckets}

        # fill the buckets
        self.total_dropped: int = 0
        self._fill_buckets()

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    def __len__(self) -> int:
        return sum(len(v) for v in self.bucket_data.values())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an item from the dataset.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A dictionary containing the image and caption.
        """
        if isinstance(idx, List):
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise ValueError("idx must be an int or a list of length 1, for now at least")
        image: Image.Image = self.dataset[idx][self.image_column]
        image = image.convert("RGB")
        return {
            "pixel_values": self.transforms(image),
            "input_ids": self.dataset[idx][self.caption_column] if rand_float() > self.ucg else "",
        }

    def get_batch_count(self) -> int:
        return sum(len(b) // self.batch_size for b in self.bucket_data.values())

    def get_batch_iterator(self) -> Generator[int, None, None]:
        """Generator providing batches of images with shared aspect ratio buckets.

        ~~Each batch is a tuple of (index, w, h).~~
        pranked actually it just returns indices because that's what torch wants
        and i'm using dataset entries which have metadata (waow)

        ...yeah I don't know what the fuck this is doing either tbh
        """
        max_bucket_length = max(len(v) for v in self.bucket_data.values())
        index_schedule = list(range(max_bucket_length))
        shuffle(index_schedule)

        bucket_len_table = {k: len(self.bucket_data[k]) for k in self.buckets}
        bucket_schedule = []
        for i, b in enumerate(self.buckets):
            bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))
        shuffle(bucket_schedule)

        bucket_pos = {b: 0 for b in self.buckets}
        bucket_total_generated = {b: 0 for b in self.buckets}
        for bucket_index in bucket_schedule:
            b = self.buckets[bucket_index]
            i = bucket_pos[b]
            bucket_len = bucket_len_table[b]

            batch = []
            while len(batch) != self.batch_size:
                # advance in the schedule until we find an index that is contained in the bucket
                k = index_schedule[i]
                if k < bucket_len:
                    entry = self.bucket_data[b][k]
                    batch.append(entry)
                i += 1
            bucket_total_generated[b] += self.batch_size
            bucket_pos[b] = i
            # yield [(idx, *b) for idx in batch]
            yield [idx for idx in batch]

    def _load_cache(self):
        cache_file = self._cache_dir.joinpath(self._cachefile_name)
        if cache_file.exists():
            self.bucket_data = torch.load(cache_file)
            self._loaded_cache = True
        return self._loaded_cache

    def _save_cache(self):
        cache_file = self._cache_dir.joinpath(self._cachefile_name)
        if not self._loaded_cache:
            logger.info("Saving bucket data to cache")
            torch.save(self.bucket_data, cache_file)

    def _fill_buckets(self):
        """Fill the buckets with the indices of the dataset entries."""
        self._load_cache()
        if self._loaded_cache is True:
            logger.info("Using cached bucket data")
        else:
            for entry, index in tqdm(
                self._dimension_iterator(),
                total=len(self.dataset),
                ncols=100,
                desc="Filtering buckets",
            ):
                if not self._process_entry(entry, index):
                    self.total_dropped += 1

        self._save_cache()
        for bucket, values in self.bucket_data.items():
            # shuffle the entries to make sure dropped elements are not always the same
            shuffle(values)

            # drop the last elements to make sure the bucket is divisible by the batch size
            drop_count = len(values) % self.batch_size
            self.bucket_data[bucket] = list(values[: len(values) - drop_count])
            self.total_dropped += drop_count

    def _dimension_iterator(self) -> Generator[Tuple[Tuple[int, int], int], None, None]:
        """Iterate over the dataset and yield the dimensions of the images, along with their index."""
        for idx, sample in enumerate(self.dataset):
            yield (sample[self.image_column].size, idx)

    def _process_entry(self, entry: Tuple[int, int], index: int) -> bool:
        """Process an entry from the dataset and add it to the correct bucket."""
        aspect = entry[0] / entry[1]  # width / height

        if aspect > self.max_ratio or (1 / aspect) > self.max_ratio:
            return False

        best_bucket = self._interp(aspect)
        if best_bucket is None:
            return False

        bucket = self.buckets[round(float(best_bucket))]
        self.bucket_data[bucket].append(index)
        return True

    def collate_fn(self, examples) -> Dict[str, Any]:
        # drop any Nones
        examples = [x for x in examples if x is not None]

        pixel_values = torch.stack([x["pixel_values"] for x in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        if self.extended_mode_chunks < 2:
            max_length = self.tokenizer.model_max_length - 2
            input_ids = [
                self.tokenizer(
                    [example["input_ids"]],
                    truncation=True,
                    return_length=True,
                    return_overflowing_tokens=False,
                    padding=False,
                    add_special_tokens=False,
                    max_length=max_length,
                ).input_ids
                for example in examples
                if example is not None
            ]
        else:
            max_length = self.tokenizer.model_max_length
            max_chunks = self.extended_mode_chunks
            input_ids = [
                self.tokenizer(
                    [example["input_ids"]],
                    truncation=True,
                    return_length=True,
                    return_overflowing_tokens=False,
                    padding=False,
                    add_special_tokens=False,
                    max_length=(max_length * max_chunks) - (max_chunks * 2),
                ).input_ids[0]
                for example in examples
                if example is not None
            ]

        tokens = input_ids

        return {"pixel_values": pixel_values, "tokens": tokens}


class AspectDatasetSampler(Sampler):
    def __init__(self, dataset: AspectBucketDataset, num_replicas: int = 1, rank: int = 0) -> None:
        super().__init__(None)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        # subsample the bucket to only include the elements that are assigned to this rank
        indices = self.dataset.get_batch_iterator()
        indices = list(indices)[self.rank :: self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.dataset.get_batch_count() // self.num_replicas
