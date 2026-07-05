import argparse
import os
import tarfile
from pathlib import Path


def default_snapshot_dir():
    hf_home = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    snapshots = (
        hf_home
        / "hub"
        / "datasets--pixparse--cc12m-wds"
        / "snapshots"
    )
    candidates = sorted(p for p in snapshots.glob("*") if p.is_dir())
    candidates = [p for p in candidates if list(p.glob("cc12m-train-*.tar"))]
    if not candidates:
        raise FileNotFoundError(f"No CC12M snapshot with tar shards found in {snapshots}")
    return candidates[-1]


def iter_shard(shard_path):
    pending = {}
    with tarfile.open(shard_path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".txt"}:
                continue
            key = str(Path(member.name).with_suffix(""))
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            record = pending.setdefault(key, {})
            if suffix in {".jpg", ".jpeg"}:
                record["image"] = extracted.read()
            else:
                record["text"] = extracted.read().decode("utf-8", errors="replace")
            if "image" in record and "text" in record:
                yield key, record["image"], record["text"]
                pending.pop(key, None)


def batched_records(shards, batch_rows, max_samples=None):
    import pyarrow as pa
    from tqdm import tqdm

    schema = pa.schema(
        [
            pa.field("sample_idx", pa.int64()),
            pa.field("key", pa.string()),
            pa.field("image", pa.binary()),
            pa.field("text", pa.string()),
        ]
    )

    sample_idx = 0
    keys, images, texts, sample_indices = [], [], [], []
    with tqdm(shards, desc="CC12M shards", unit="shard") as pbar:
        for shard in pbar:
            for key, image, text in iter_shard(shard):
                sample_indices.append(sample_idx)
                keys.append(key)
                images.append(image)
                texts.append(text)
                sample_idx += 1
                if sample_idx % 1000 == 0:
                    pbar.set_postfix(samples=sample_idx)
                if len(images) >= batch_rows:
                    yield pa.RecordBatch.from_arrays(
                        [
                            pa.array(sample_indices, pa.int64()),
                            pa.array(keys, pa.string()),
                            pa.array(images, pa.binary()),
                            pa.array(texts, pa.string()),
                        ],
                        schema=schema,
                    )
                    keys, images, texts, sample_indices = [], [], [], []
                if max_samples is not None and sample_idx >= max_samples:
                    break
            if max_samples is not None and sample_idx >= max_samples:
                break

    if images:
        yield pa.RecordBatch.from_arrays(
            [
                pa.array(sample_indices, pa.int64()),
                pa.array(keys, pa.string()),
                pa.array(images, pa.binary()),
                pa.array(texts, pa.string()),
            ],
            schema=schema,
        )


def main():
    parser = argparse.ArgumentParser(description="Build a CC12M Lance dataset.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing cc12m-train-*.tar. Defaults to HF_HOME snapshot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/cc12m/train.lance"),
    )
    parser.add_argument("--batch-rows", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    import lance
    import pyarrow as pa

    source_dir = args.source_dir or default_snapshot_dir()
    shards = sorted(source_dir.glob("cc12m-train-*.tar"))
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    if not shards:
        raise FileNotFoundError(f"No cc12m-train-*.tar shards found in {source_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("sample_idx", pa.int64()),
            pa.field("key", pa.string()),
            pa.field("image", pa.binary()),
            pa.field("text", pa.string()),
        ]
    )

    print(f"source_dir={source_dir}")
    print(f"shards={len(shards)}")
    print(f"output={args.output}")
    if args.max_samples is not None:
        print(f"max_samples={args.max_samples}")

    lance.write_dataset(
        batched_records(shards, args.batch_rows, args.max_samples),
        str(args.output),
        schema=schema,
        mode="overwrite",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
