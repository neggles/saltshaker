from random import shuffle
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from datasets import Dataset as HFDataset
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, Sampler, get_worker_info
from tqdm_loggable.auto import tqdm

from saltshaker.settings import AspectBucketInfo, Settings


class AspectBucketDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        image_column: str,
        caption_column: str,
        settings: Settings,
        flip: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.image_column = image_column
        self.caption_column = caption_column

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
        return self.dataset[entry]

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

        for entry, index in tqdm(self._dimension_iterator(), total=len(self.dataset)):
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

    def _process_entry(self, entry: Tuple[int, int], index: int):
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


class AspectDatasetSampler(Sampler):
    def __init__(self, dataset: AspectBucketDataset):
        super().__init__(None)
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset.get_batch_iterator())

    def __len__(self) -> int:
        return self.dataset.get_batch_count()
