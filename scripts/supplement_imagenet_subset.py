"""Fill ImageNet class shortages from a second training-only subset.

The original subset stays in place. Added files and their exact source rows are
recorded so the final manifest can identify which source supplied each image.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

TARGET_PER_CLASS = 210


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as image:
        for block in iter(lambda: image.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--supplement-source", type=Path, required=True)
    args = parser.parse_args()

    target = args.target
    supplement = args.supplement_source
    selected = target / "selected"
    state = json.loads((target / "state.json").read_text())
    if state != {"next_shard": 52}:
        raise RuntimeError(f"Unexpected source-download state: {state}")

    target_classes = json.loads((target / "imagenet_class_index.json").read_text())
    source_classes = json.loads((supplement / "imagenet_class_index.json").read_text())
    if target_classes != source_classes:
        raise RuntimeError("ImageNet class-index mappings differ")
    source_info = json.loads((supplement / "source.json").read_text())
    if source_info["split"] != "train":
        raise RuntimeError("Supplement images must come from the training split")

    wnids = [target_classes[str(index)][0] for index in range(1000)]
    counts = {wnid: len(list((selected / wnid).glob("*.JPEG"))) for wnid in wnids}
    deficits = {wnid: TARGET_PER_CLASS - counts[wnid] for wnid in wnids if counts[wnid] < TARGET_PER_CLASS}
    if sum(deficits.values()) != 99:
        raise RuntimeError(f"Expected the audited 99-image deficit, found {sum(deficits.values())}")

    source_records = {}
    for line in (supplement / "manifest.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["split"] == "train" and record["wnid"] in deficits:
            source_records.setdefault(record["wnid"], []).append(record)

    existing_hashes = {
        wnid: {digest(path) for path in (selected / wnid).glob("*.JPEG")}
        for wnid in deficits
    }
    supplement_manifest = []
    ordinal = 0
    for wnid, missing in deficits.items():
        added = 0
        for source_record in source_records[wnid]:
            source_image = supplement / "train" / wnid / source_record["filename"]
            image_hash = digest(source_image)
            if image_hash in existing_hashes[wnid]:
                continue

            filename = f"s99999_r{ordinal:06d}.JPEG"
            target_image = selected / wnid / filename
            if target_image.exists():
                raise RuntimeError(f"Supplement filename already exists: {target_image}")
            shutil.copy2(source_image, target_image)
            existing_hashes[wnid].add(image_hash)
            supplement_manifest.append(
                {
                    "target_filename": filename,
                    "wnid": wnid,
                    "source": source_info["source"],
                    "revision": source_info["revision"],
                    "source_split": "train",
                    "source_filename": source_record["filename"],
                    "source_shard": source_record["source_shard"],
                    "source_row": source_record["source_row"],
                    "sha256": image_hash,
                }
            )
            ordinal += 1
            added += 1
            if added == missing:
                break
        if added != missing:
            raise RuntimeError(f"Source subset lacks {missing - added} images for {wnid}")
        print(f"{wnid}: {counts[wnid]} + {added} = {TARGET_PER_CLASS}", flush=True)

    with (target / "supplement_manifest.jsonl").open("w") as output:
        for record in supplement_manifest:
            output.write(json.dumps(record) + chr(10))

    target_info = json.loads((target / "source.json").read_text())
    target_info["supplementation"] = {
        "source": source_info["source"],
        "revision": source_info["revision"],
        "split": "train",
        "selection": "first non-duplicate training images in the old subset manifest order, only for classes below 210",
        "counts_by_wnid": deficits,
        "manifest": "supplement_manifest.jsonl",
    }
    (target / "source.json").write_text(json.dumps(target_info, indent=2) + chr(10))
    (target / "state.json").write_text(json.dumps({"phase": "split"}))
    print(f"Prepared {len(supplement_manifest)} supplementary training images", flush=True)


if __name__ == "__main__":
    main()
