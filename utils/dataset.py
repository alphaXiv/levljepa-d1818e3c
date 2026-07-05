import io
import os
import random
from pathlib import Path

import lance
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision.transforms import v2 as T

from utils.text import tokenize_with_eos_readout


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _pair(value, name):
    values = tuple(float(v) for v in value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    return values


def _clip_normalize():
    return T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)


def _env_int(*names):
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def build_single_image_transform(
    size=224,
    augmentation="resize",
    crop_scale=(0.9, 1.0),
    crop_ratio=(3.0 / 4.0, 4.0 / 3.0),
    color_jitter=None,
    color_jitter_prob=0.0,
    grayscale_prob=0.0,
    gaussian_blur_prob=0.0,
    random_erasing_prob=0.0,
):
    """Build the single-view image transform used by LeVLJEPA training."""
    augmentation = str(augmentation).lower()
    if augmentation in {"none", "resize", "legacy"}:
        return T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize((size, size)),
                _clip_normalize(),
            ]
        )
    if augmentation in {"standard", "clip"}:
        transforms = [
            T.RandomResizedCrop(
                size,
                scale=_pair(crop_scale, "crop_scale"),
                ratio=_pair(crop_ratio, "crop_ratio"),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.RandomHorizontalFlip(p=0.5),
        ]
        if color_jitter_prob > 0:
            if color_jitter is None:
                raise ValueError(
                    "color_jitter must be set when color_jitter_prob > 0."
                )
            jitter = tuple(float(v) for v in color_jitter)
            if len(jitter) != 4:
                raise ValueError(
                    "color_jitter must contain brightness, contrast, saturation, hue."
                )
            transforms.append(
                T.RandomApply([T.ColorJitter(*jitter)], p=color_jitter_prob)
            )
        if grayscale_prob > 0:
            transforms.append(T.RandomGrayscale(p=grayscale_prob))
        if gaussian_blur_prob > 0:
            transforms.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                    p=gaussian_blur_prob,
                )
            )
        transforms.extend(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                _clip_normalize(),
            ]
        )
        if random_erasing_prob > 0:
            transforms.append(T.RandomErasing(p=random_erasing_prob, value=0.0))
        return T.Compose(transforms)
    raise ValueError("augmentation must be one of: resize, legacy, standard, clip.")


class _CC12MLanceBase:
    def __init__(
        self,
        path,
        tokenizer=None,
        max_length=77,
        max_samples=None,
        image_column="image",
        text_column="text",
        text_readout="eot",
    ):
        self.path = str(Path(path))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_readout = text_readout
        self.image_column = image_column
        self.text_column = text_column
        self.columns = [image_column, text_column]
        self._dataset = None
        self._fragments = None
        self._length = self._open().count_rows()
        self._dataset = None
        if max_samples is not None:
            self._length = min(int(max_samples), self._length)

    def _open(self):
        if self._dataset is None:
            self._dataset = lance.dataset(self.path)
        return self._dataset

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dataset"] = None
        state["_fragments"] = None
        return state

    def __len__(self):
        return self._length

    @staticmethod
    def _decode_image(blob):
        if hasattr(blob, "readall"):
            blob = blob.readall()
        elif hasattr(blob, "data") and not isinstance(
            blob, (bytes, bytearray, memoryview)
        ):
            blob = blob.data
        return Image.open(io.BytesIO(blob)).convert("RGB")

    def _fragments_for_scan(self):
        if self._fragments is None:
            self._fragments = self._open().get_fragments()
        return self._fragments

    def _tokenize(self, text):
        if self.tokenizer is None:
            return {"text": text}
        if self.text_readout == "eot":
            tokens = tokenize_with_eos_readout(
                self.tokenizer, text, max_length=self.max_length, padding="max_length"
            )
        elif self.text_readout == "pad77":
            tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            raise ValueError("text_readout must be either 'eot' or 'pad77'.")
        return {k: v.squeeze(0) for k, v in tokens.items()}

    def _rows(self, indices):
        table = self._open().take(indices, columns=self.columns)
        return zip(
            table[self.image_column].to_pylist(),
            table[self.text_column].to_pylist(),
        )

    def _make_item(self, blob, text):
        raise NotImplementedError


class _CC12MLanceMapBase(_CC12MLanceBase, Dataset):
    """Map-style Lance dataset using batched random row reads."""

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        normalized = [int(i) % self._length for i in indices]
        items = []
        for row_idx, (blob, text) in zip(normalized, self._rows(normalized)):
            for attempt in range(5):
                try:
                    return_idx = (row_idx + attempt) % self._length
                    if attempt:
                        blob, text = next(self._rows([return_idx]))
                    items.append(self._make_item(blob, text))
                    break
                except Exception:
                    if attempt == 4:
                        raise RuntimeError(
                            "CC12M Lance: failed to decode 5 consecutive samples "
                            f"starting at idx={row_idx}"
                        )
        return items


class _CC12MLanceScanBase(_CC12MLanceBase, IterableDataset):
    """Iterable Lance dataset using fragment-local scans and bounded shuffling."""

    def __init__(
        self,
        *args,
        scan_batch_size=2048,
        shuffle_buffer_size=8192,
        batch_readahead=4,
        fragment_readahead=2,
        scan_in_order=True,
        seed=0,
        cycle=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scan_batch_size = max(1, int(scan_batch_size))
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self.batch_readahead = max(0, int(batch_readahead))
        self.fragment_readahead = max(0, int(fragment_readahead))
        self.scan_in_order = bool(scan_in_order)
        self.seed = int(seed)
        self.cycle = bool(cycle)

    @staticmethod
    def _rank_and_worker():
        rank = None
        world_size = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = _env_int(
                "RANK",
                "SLURM_PROCID",
                "PMI_RANK",
                "OMPI_COMM_WORLD_RANK",
            )
            world_size = _env_int(
                "WORLD_SIZE",
                "SLURM_NTASKS",
                "PMI_SIZE",
                "OMPI_COMM_WORLD_SIZE",
            )

        rank = 0 if rank is None else rank
        world_size = 1 if world_size is None else max(1, world_size)

        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers
        return shard_id, num_shards

    def _local_fragments(self, rng):
        fragments = list(self._fragments_for_scan())
        if not fragments:
            raise RuntimeError(f"Lance dataset has no fragments: {self.path}")

        shard_id, num_shards = self._rank_and_worker()
        local = fragments[shard_id::num_shards]
        if not local:
            # More workers than fragments. Duplicate a fragment instead of letting
            # the worker terminate; prefer lowering num_workers in this case.
            local = [fragments[shard_id % len(fragments)]]
        rng.shuffle(local)
        return local

    def _scan_rows(self, fragments):
        scanner = self._open().scanner(
            columns=self.columns,
            batch_size=self.scan_batch_size,
            batch_readahead=self.batch_readahead,
            fragment_readahead=self.fragment_readahead,
            scan_in_order=self.scan_in_order,
            fragments=fragments,
        )
        for batch in scanner.to_batches():
            image_idx = batch.schema.get_field_index(self.image_column)
            text_idx = batch.schema.get_field_index(self.text_column)
            images = batch.column(image_idx).to_pylist()
            texts = batch.column(text_idx).to_pylist()
            yield from zip(images, texts)

    def _items_from_rows(self, rows):
        failures = 0
        for blob, text in rows:
            try:
                yield self._make_item(blob, text)
                failures = 0
            except Exception as exc:
                failures += 1
                if failures >= 5:
                    raise RuntimeError(
                        "CC12M Lance scan: failed to decode 5 consecutive samples"
                    ) from exc

    def _shuffle(self, items, rng):
        if self.shuffle_buffer_size <= 1:
            yield from items
            return

        buffer = []
        for item in items:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(item)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = item

        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self):
        epoch = 0
        while True:
            shard_id, _ = self._rank_and_worker()
            rng = random.Random(self.seed + epoch * 1_000_003 + shard_id)
            rows = self._shuffle(self._scan_rows(self._local_fragments(rng)), rng)
            yield from self._items_from_rows(rows)
            if not self.cycle:
                return
            epoch += 1


class CC12MLanceDataset(_CC12MLanceMapBase):
    def __init__(
        self,
        path,
        size=224,
        tokenizer=None,
        max_length=77,
        max_samples=None,
        image_column="image",
        text_column="text",
        text_readout="eot",
        image_augmentation="resize",
        crop_scale=(0.9, 1.0),
        crop_ratio=(3.0 / 4.0, 4.0 / 3.0),
        color_jitter=None,
        color_jitter_prob=0.0,
        grayscale_prob=0.0,
        gaussian_blur_prob=0.0,
        random_erasing_prob=0.0,
    ):
        super().__init__(
            path=path,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples,
            image_column=image_column,
            text_column=text_column,
            text_readout=text_readout,
        )
        self.transform = build_single_image_transform(
            size=size,
            augmentation=image_augmentation,
            crop_scale=crop_scale,
            crop_ratio=crop_ratio,
            color_jitter=color_jitter,
            color_jitter_prob=color_jitter_prob,
            grayscale_prob=grayscale_prob,
            gaussian_blur_prob=gaussian_blur_prob,
            random_erasing_prob=random_erasing_prob,
        )

    def _make_item(self, blob, text):
        return {"image": self.transform(self._decode_image(blob)), **self._tokenize(text)}


class CC12MLanceScanDataset(_CC12MLanceScanBase):
    def __init__(
        self,
        path,
        size=224,
        tokenizer=None,
        max_length=77,
        max_samples=None,
        image_column="image",
        text_column="text",
        text_readout="eot",
        image_augmentation="resize",
        crop_scale=(0.9, 1.0),
        crop_ratio=(3.0 / 4.0, 4.0 / 3.0),
        color_jitter=None,
        color_jitter_prob=0.0,
        grayscale_prob=0.0,
        gaussian_blur_prob=0.0,
        random_erasing_prob=0.0,
        **scan_kwargs,
    ):
        super().__init__(
            path=path,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples,
            image_column=image_column,
            text_column=text_column,
            text_readout=text_readout,
            **scan_kwargs,
        )
        self.transform = build_single_image_transform(
            size=size,
            augmentation=image_augmentation,
            crop_scale=crop_scale,
            crop_ratio=crop_ratio,
            color_jitter=color_jitter,
            color_jitter_prob=color_jitter_prob,
            grayscale_prob=grayscale_prob,
            gaussian_blur_prob=gaussian_blur_prob,
            random_erasing_prob=random_erasing_prob,
        )

    def _make_item(self, blob, text):
        return {"image": self.transform(self._decode_image(blob)), **self._tokenize(text)}
