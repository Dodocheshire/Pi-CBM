"""Extract 200 train and 10 adapter-validation images per ImageNet-1K class.

The DataComp 0% randomized quarter-size subset stores ImageNet training images
as Parquet. One shard is downloaded at a time, and its revision and selected
rows are recorded for reproducibility.
"""

import argparse
import json
import random
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

SOURCE = "https://huggingface.co/datasets/datacomp/imagenet-1k-random-0.0-frac-1over4"
FALLBACK_SOURCE = "https://hf-mirror.net/datasets/datacomp/imagenet-1k-random-0.0-frac-1over4"
REVISION = "356588cce9ca8d2be61f0d4541db6274b448954f"
CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)
SHARD_COUNT = 52
PREFETCH = 4
TRAIN_PER_CLASS = 200
ADAPTER_VAL_PER_CLASS = 10
PER_CLASS = TRAIN_PER_CLASS + ADAPTER_VAL_PER_CLASS
# The first 24 shards were already selected with the original 110-image cap.
# Keep those images, then fill every class to 210 from subsequent shards.
INITIAL_PER_CLASS = 110
EXTENSION_START_SHARD = 24
SEED = 42


def download_shard(index: int, scratch: Path, proxy: str | None) -> Path:
    name = f"train-{index:05d}-of-{SHARD_COUNT:05d}.parquet"
    finished = scratch / name
    if finished.exists():
        return finished

    partial = scratch / f"{name}.part"

    def fetch(source: str, proxy: str | None):
        route = ["--socks5-hostname", proxy] if proxy else ["--noproxy", "*"]
        subprocess.run(
            [
                "curl",
                *route,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "8",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "--speed-time",
                "90",
                "--speed-limit",
                "1024",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                f"{source}/resolve/{REVISION}/data/{name}",
            ],
            check=True,
        )

    if proxy:
        try:
            fetch(SOURCE, proxy)
        except subprocess.CalledProcessError:
            # The direct mirror keeps a long download alive after the SSH relay ends.
            fetch(FALLBACK_SOURCE, None)
    else:
        fetch(FALLBACK_SOURCE, None)
    partial.rename(finished)
    return finished


def selected_counts(selected: Path, wnids: list[str]) -> dict[str, int]:
    return {wnid: len(list((selected / wnid).glob("*.JPEG"))) for wnid in wnids}


def extract_shard(
    shard: Path, index: int, selected: Path, wnids: list[str], counts: dict[str, int]
) -> None:
    # Row numbers are local to a source shard and become part of each filename.
    row = 0
    cap = INITIAL_PER_CLASS if index < EXTENSION_START_SHARD else PER_CLASS
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=128, columns=["image", "label"]):
        images = batch.column("image").to_pylist()
        labels = batch.column("label").to_pylist()
        for image, label in zip(images, labels):
            wnid = wnids[label]
            if counts[wnid] < cap:
                directory = selected / wnid
                directory.mkdir(exist_ok=True)
                (directory / f"s{index:05d}_r{row:06d}.JPEG").write_bytes(image["bytes"])
                counts[wnid] += 1
            row += 1


def split_selected(root: Path, selected: Path, wnids: list[str]) -> None:
    source_info = json.loads((root / "source.json").read_text())
    supplement_file = root / "supplement_manifest.jsonl"
    supplements = {}
    if supplement_file.exists():
        supplements = {
            record["target_filename"]: record
            for record in map(json.loads, supplement_file.read_text().splitlines())
        }

    manifest = []
    for wnid in wnids:
        names = sorted(
            {
                path.name
                for directory in ("selected", "train", "adapter_val")
                for path in (root / directory / wnid).glob("*.JPEG")
            }
        )
        validation = set(random.Random(f"{SEED}:{wnid}").sample(names, ADAPTER_VAL_PER_CLASS))
        for name in names:
            split = "adapter_val" if name in validation else "train"
            target = root / split / wnid / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                (selected / wnid / name).rename(target)

            if name in supplements:
                record = dict(supplements[name])
                record.update({"split": split, "wnid": wnid, "filename": name})
                record.pop("target_filename")
            else:
                record = {
                    "split": split,
                    "wnid": wnid,
                    "filename": name,
                    "source": source_info["source"],
                    "revision": source_info["revision"],
                    "source_split": "train",
                    "source_shard": int(name[1:6]),
                    "source_row": int(name[8:14]),
                }
            manifest.append(record)

    with (root / "manifest.jsonl").open("w") as output:
        for record in manifest:
            output.write(json.dumps(record) + "\n")
    shutil.rmtree(selected)
    (root / "COMPLETE").write_text(
        "200 train and 10 adapter-validation images for each of 1000 classes\n"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--reference-val", type=Path, required=True)
    parser.add_argument("--proxy", help="SOCKS5 proxy on the download host")
    args = parser.parse_args()

    root = args.output
    scratch = args.scratch
    selected = root / "selected"
    root.mkdir(parents=True, exist_ok=True)
    selected.mkdir(exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    state_file = root / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"next_shard": 0}
    if (root / "COMPLETE").exists():
        print("Subset already complete", flush=True)
        return


    class_index_file = root / "imagenet_class_index.json"
    if class_index_file.exists():
        mapping = json.loads(class_index_file.read_text())
    else:
        mapping = json.load(urllib.request.urlopen(CLASS_INDEX_URL))
    wnids = [mapping[str(index)][0] for index in range(1000)]
    assert set(wnids) == {path.name for path in args.reference_val.iterdir() if path.is_dir()}
    class_index_file.write_text(json.dumps(mapping, indent=2))
    (root / "source.json").write_text(
        json.dumps(
            {
                "source": SOURCE,
                "fallback_source": FALLBACK_SOURCE,
                "revision": REVISION,
                "split": "train",
                "selection": (
                    "up to 110 per class from shards 0-23, then fill to 210 "
                    "per class from shard 24 onward, in source row order"
                ),
                "train_per_class": TRAIN_PER_CLASS,
                "adapter_val_per_class": ADAPTER_VAL_PER_CLASS,
                "adapter_val_seed": SEED,
                "class_index_source": CLASS_INDEX_URL,
            },
            indent=2,
        )
    )

    supplement_file = root / "supplement_manifest.jsonl"
    if supplement_file.exists() and state.get("phase") == "split":
        supplement_records = list(map(json.loads, supplement_file.read_text().splitlines()))
        supplement_info = {
            "source": supplement_records[0]["source"],
            "revision": supplement_records[0]["revision"],
            "split": "train",
            "selection": "first non-duplicate images from the label-verified supplement, scanning source shards and rows in order to fill classes below 210",
            "counts_by_wnid": {
                wnid: sum(record["wnid"] == wnid for record in supplement_records)
                for wnid in sorted({record["wnid"] for record in supplement_records})
            },
            "manifest": "supplement_manifest.jsonl",
        }
        source_info = json.loads((root / "source.json").read_text())
        source_info["supplementation"] = supplement_info
        (root / "source.json").write_text(json.dumps(source_info, indent=2) + chr(10))

    if state.get("phase") == "supplement_required":
        raise RuntimeError("ImageNet subset is incomplete; use a label-verified training source for the recorded class deficits")
    if state.get("phase") == "split":
        split_selected(root, selected, wnids)
        print("Subset complete: 200,000 train + 10,000 adapter_val", flush=True)
        return
    next_shard = state["next_shard"]
    # An interrupted shard is regenerated from scratch; completed shards remain.
    for path in selected.glob(f"*/s{next_shard:05d}_r*.JPEG"):
        path.unlink()
    counts = selected_counts(selected, wnids)

    with ThreadPoolExecutor(max_workers=PREFETCH) as downloads:
        pending = {
            index: downloads.submit(download_shard, index, scratch, args.proxy)
            for index in range(next_shard, min(next_shard + PREFETCH, SHARD_COUNT))
        }
        for index in range(next_shard, SHARD_COUNT):
            started = time.monotonic()
            shard = pending.pop(index).result()
            following = index + PREFETCH
            if following < SHARD_COUNT:
                pending[following] = downloads.submit(
                    download_shard, following, scratch, args.proxy
                )
            extract_shard(shard, index, selected, wnids, counts)
            state_file.write_text(json.dumps({"next_shard": index + 1}))
            shard.unlink()
            complete = sum(count == PER_CLASS for count in counts.values())
            print(
                f"shard={index:03d} selected={sum(counts.values()):,}/{PER_CLASS * 1000:,} "
                f"complete_classes={complete}/1000 "
                f"free_gb={shutil.disk_usage(root).free / 1e9:.1f} "
                f"seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
            if complete == 1000:
                state_file.write_text(json.dumps({"phase": "split"}))
                split_selected(root, selected, wnids)
                print("Subset complete: 200,000 train + 10,000 adapter_val", flush=True)
                return

    raise RuntimeError("Not every class reached 210 images after all train shards")


if __name__ == "__main__":
    main()
