"""Collect only missing classes from the clean-label DataComp half-size subset.

Images remain in a staging directory until the separate label-QC step passes.
The downloader resumes at the next shard and records every selected row.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import pyarrow.parquet as pq

SOURCE = "https://huggingface.co/datasets/datacomp/imagenet-1k-random-0.0-frac-1over2"
MIRROR = "https://hf-mirror.net/datasets/datacomp/imagenet-1k-random-0.0-frac-1over2"
REVISION = "2604c097f0275b11659ca2b50e46eba8b4284995"
SHARD_COUNT = 104
TARGET_PER_CLASS = 210


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def download_shard(index: int, scratch: Path) -> Path:
    name = f"train-{index:05d}-of-{SHARD_COUNT:05d}.parquet"
    complete = scratch / name
    if complete.exists():
        return complete

    partial = scratch / f"{name}.part"
    command = [
        "curl",
        "--noproxy",
        "*",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--speed-time",
        "300",
        "--speed-limit",
        "1024",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        f"{MIRROR}/resolve/{REVISION}/data/{name}",
    ]
    # Run one curl attempt at a time. A new process re-reads the current
    # partial-file length, so a slow-transfer retry resumes at the true offset.
    for attempt in range(1, 9):
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            break
        retained = partial.stat().st_size if partial.exists() else 0
        print(
            f"shard={index:03d} retry={attempt}/8 "
            f"curl_exit={result.returncode} retained_bytes={retained}",
            flush=True,
        )
        if attempt == 8:
            raise subprocess.CalledProcessError(result.returncode, command)
        time.sleep(5)
    partial.rename(complete)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()

    selected = args.target / "selected"
    class_index = json.loads((args.target / "imagenet_class_index.json").read_text())
    wnids = [class_index[str(index)][0] for index in range(1000)]
    current_counts = {
        wnid: len(list((selected / wnid).glob("*.JPEG"))) for wnid in wnids
    }
    deficits = {
        wnid: TARGET_PER_CLASS - count
        for wnid, count in current_counts.items()
        if count < TARGET_PER_CLASS
    }
    current_hashes = {
        wnid: {
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (selected / wnid).glob("*.JPEG")
        }
        for wnid in deficits
    }

    args.scratch.mkdir(parents=True, exist_ok=True)
    images = args.work / "images"
    images.mkdir(parents=True, exist_ok=True)
    state_file = args.work / "state.json"
    state = (
        json.loads(state_file.read_text())
        if state_file.exists()
        else {"next_shard": 0, "candidates": [], "deficits": deficits}
    )
    if state["deficits"] != deficits:
        raise RuntimeError("The target dataset changed since this supplement scan began")

    candidates = state["candidates"]
    hashes = {wnid: set(values) for wnid, values in current_hashes.items()}
    for record in candidates:
        hashes[record["wnid"]].add(record["sha256"])
    found = {wnid: 0 for wnid in deficits}
    for record in candidates:
        found[record["wnid"]] += 1

    next_shard = state["next_shard"]
    for path in images.glob(f"*/s{next_shard:05d}_r*.JPEG"):
        path.unlink()
    started = time.monotonic()
    for shard_index in range(next_shard, SHARD_COUNT):
        shard = download_shard(shard_index, args.scratch)
        row = 0
        for batch in pq.ParquetFile(shard).iter_batches(
            batch_size=128, columns=["image", "label"]
        ):
            image_values = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for image, label in zip(image_values, labels):
                wnid = wnids[label]
                if wnid in deficits and found[wnid] < deficits[wnid]:
                    data = image["bytes"]
                    image_hash = sha256_bytes(data)
                    if image_hash not in hashes[wnid]:
                        filename = f"s{shard_index:05d}_r{row:06d}.JPEG"
                        path = images / wnid / filename
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                        record = {
                            "target_filename": filename,
                            "wnid": wnid,
                            "source": SOURCE,
                            "revision": REVISION,
                            "source_split": "train",
                            "source_shard": shard_index,
                            "source_row": row,
                            "sha256": image_hash,
                            "work_path": str(path),
                        }
                        candidates.append(record)
                        hashes[wnid].add(image_hash)
                        found[wnid] += 1
                row += 1
        shard.unlink()
        state = {
            "next_shard": shard_index + 1,
            "candidates": candidates,
            "deficits": deficits,
        }
        temporary = state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_file)
        print(
            f"shard={shard_index:03d} "
            f"found={sum(found.values())}/{sum(deficits.values())} "
            f"by_class={found} seconds={time.monotonic() - started:.1f}",
            flush=True,
        )
        if found == deficits:
            state["complete"] = True
            state_file.write_text(json.dumps(state, indent=2))
            print("All 99 unique, clean-label supplement candidates are staged", flush=True)
            return

    raise RuntimeError(f"Only found {found}; required {deficits}")


if __name__ == "__main__":
    main()
