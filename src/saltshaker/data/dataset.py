from random import (
    random as rand_float,
    shuffle,
)
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import DistributedType
from accelerate.accelerator import AcceleratorState
from accelerate.logging import get_logger
from datasets import Dataset as HFDataset
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, Sampler, get_worker_info
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor
from tqdm_loggable.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from saltshaker.settings import AspectBucketInfo, Settings

logger = get_logger(__name__)


def _sort_by_ratio(bucket: tuple) -> float:
    return bucket[0] / bucket[1]


def _sort_by_area(bucket: tuple) -> float:
    return bucket[0] * bucket[1]


class AspectBucketDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        device: torch.device,
        settings: Settings,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device

        self.image_column = settings.image_column
        self.caption_column = settings.caption_column
        self.ucg = settings.ucg
        self.extended_mode_chunks = settings.extended_mode_chunks
        self.clip_penultimate = settings.clip_penultimate

        self._accel = AcceleratorState()

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

    def __len__(self) -> int:
        return sum(len(v) for v in self.bucket_data.values())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an item from the dataset.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A dictionary containing the image and caption.
        """
        bucket = self.buckets[idx]
        entry = self.bucket_data[bucket][idx]
        return {
            "pixel_values": self.transforms(self.dataset[entry][self.image_column]),
            "input_ids": self.dataset[entry][self.caption_column] if rand_float() > self.ucg else "",
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

    def _fill_buckets(self):
        """Fill the buckets with the indices of the dataset entries."""

        for entry, index in tqdm(
            self._dimension_iterator(), total=len(self.dataset), ncols=100, desc="Filtering buckets"
        ):
            if not self._process_entry(entry, index):
                self.total_dropped += 1

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
        pixel_values = pixel_values.to(memory_format=torch.channels_last, dtype=torch.float32)

        if self.extended_mode_chunks < 2:
            max_length = self.tokenizer.model_max_length - 2
            input_ids = [
                self.tokenizer(
                    [example["input_ids"]],
                    truncation=True,
                    return_length=True,
                    return_overflowing_tokens=False,
                    padding="longest",
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

        if self.extended_mode_chunks < 2:
            for i, x in enumerate(input_ids):
                for j, y in enumerate(x):
                    input_ids[i][j] = [
                        self.tokenizer.bos_token_id,
                        *y,
                        *np.full((self.tokenizer.model_max_length - len(y) - 1), self.tokenizer.eos_token_id),
                    ]

            if self.clip_penultimate:
                input_ids = [
                    self.text_encoder.text_model.final_layer_norm(
                        self.text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True)[
                            "hidden_states"
                        ][-2]
                    )[0]
                    for input_id in input_ids
                ]
            else:
                input_ids = [
                    self.text_encoder(
                        torch.asarray(input_id).to(self.device), output_hidden_states=True
                    ).last_hidden_state[0]
                    for input_id in input_ids
                ]
        else:
            max_standard_tokens = max_length - 2
            max_chunks = self.extended_mode_chunks
            max_len = (
                np.ceil(max(len(x) for x in input_ids) / max_standard_tokens).astype(int).item()
                * max_standard_tokens
            )
            if max_len > max_standard_tokens:
                z = None
                for i, x in enumerate(input_ids):
                    if len(x) < max_len:
                        input_ids[i] = [*x, *np.full((max_len - len(x)), self.tokenizer.eos_token_id)]
                batch_t = torch.tensor(input_ids)
                chunks = [
                    batch_t[:, i : i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)
                ]
                for chunk in chunks:
                    chunk = torch.cat(
                        (
                            torch.full((chunk.shape[0], 1), self.tokenizer.bos_token_id),
                            chunk,
                            torch.full((chunk.shape[0], 1), self.tokenizer.eos_token_id),
                        ),
                        1,
                    )
                    if z is None:
                        if self.clip_penultimate:
                            z = self.text_encoder.text_model.final_layer_norm(
                                self.text_encoder(chunk.to(self.device), output_hidden_states=True)[
                                    "hidden_states"
                                ][-2]
                            )
                        else:
                            z = self.text_encoder(
                                chunk.to(self.device), output_hidden_states=True
                            ).last_hidden_state
                    else:
                        if self.clip_penultimate:
                            z = torch.cat(
                                (
                                    z,
                                    self.text_encoder.text_model.final_layer_norm(
                                        self.text_encoder(chunk.to(self.device), output_hidden_states=True)[
                                            "hidden_states"
                                        ][-2]
                                    ),
                                ),
                                dim=-2,
                            )
                        else:
                            z = torch.cat(
                                (
                                    z,
                                    self.text_encoder(
                                        chunk.to(self.device), output_hidden_states=True
                                    ).last_hidden_state,
                                ),
                                dim=-2,
                            )
                input_ids = z
            else:
                for i, x in enumerate(input_ids):
                    input_ids[i] = [
                        self.tokenizer.bos_token_id,
                        *x,
                        *np.full((self.tokenizer.model_max_length - len(x) - 1), self.tokenizer.eos_token_id),
                    ]
                if self.clip_penultimate:
                    input_ids = self.text_encoder.text_model.final_layer_norm(
                        self.text_encoder(
                            torch.asarray(input_ids).to(self.device), output_hidden_states=True
                        )["hidden_states"][-2]
                    )
                else:
                    input_ids = self.text_encoder(
                        torch.asarray(input_ids).to(self.device), output_hidden_states=True
                    ).last_hidden_state

        input_ids = torch.stack(tuple(input_ids))
        return {"pixel_values": pixel_values, "input_ids": input_ids, "tokens": tokens}


class AspectDatasetSampler(Sampler):
    def __init__(self, dataset: AspectBucketDataset, num_replicas: int = 1, rank: int = 0) -> None:
        super().__init__(None)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        # subsample the bucket to only include the elements that are assigned to this rank
        indices = self.dataset.get_batch_iterator()
        indices = list(indices)[self.rank :: self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.dataset.get_batch_count() // self.num_replicas
