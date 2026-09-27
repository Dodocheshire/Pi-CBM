"""Small datasets and reusable, content-addressed frozen feature caches."""

import hashlib
import json
import math
from pathlib import Path

import clip
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

from .models.backbones import load_backbone, target_preprocess


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def split_seed(config: dict) -> int:
    value = config["data"].get("split_seed")
    return config["experiment"]["seed"] if value is None else value


def read_cache(path: Path | str) -> dict:
    """Map spatial features and preprocessed images without loading all bytes."""
    result = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
    for split in result["splits"].values():
        for name in ("internal", "images"):
            if f"{name}_file" in split:
                shape = split[f"{name}_shape"]
                split[name] = torch.from_file(
                    split[f"{name}_file"],
                    shared=False,
                    size=math.prod(shape),
                    dtype=getattr(torch, split.get(f"{name}_dtype", "float32")),
                ).reshape(shape)
    return result


class PairedImages(Dataset):
    """Apply target and teacher preprocessing to the exact same source image."""

    def __init__(self, dataset, indices, target_transform, teacher_transform):
        self.dataset = dataset
        self.indices = indices
        self.target_transform = target_transform
        self.teacher_transform = teacher_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[int(self.indices[index])]
        return self.target_transform(image), self.teacher_transform(image), label


class ImageNetTrain(ConcatDataset):
    """Keep the fixed 100/10 train split while exposing one training index space."""

    def __init__(self, root: Path):
        training = datasets.ImageFolder(str(root / "train"))
        adapter_val = datasets.ImageFolder(str(root / "adapter_val"))
        if training.class_to_idx != adapter_val.class_to_idx:
            raise ValueError("ImageNet train and adapter_val must use the same WNIDs")
        super().__init__([training, adapter_val])
        self.train_count = len(training)
        self.classes = training.classes
        self.class_to_idx = training.class_to_idx
        self.targets = training.targets + adapter_val.targets
        self.samples = training.samples + adapter_val.samples


def stratified_indices(labels, count: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """count is per class; zero means all available examples."""
    chosen, remaining = [], []
    labels = np.asarray(labels)
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        n = count or len(indices)
        if n > len(indices):
            raise ValueError(
                f"Requested {n} examples for class {label}, only {len(indices)} available"
            )
        chosen.extend(indices[:n])
        remaining.extend(indices[n:])
    return rng.permutation(chosen), np.asarray(remaining, dtype=np.int64)


def make_splits(config: dict):
    cfg = config["data"]
    root = cfg["root"]
    if cfg["dataset"] in {"cifar10", "cifar100"}:
        dataset_class = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[cfg["dataset"]]
        train = dataset_class(root, train=True, download=True)
        test = dataset_class(root, train=False, download=True)
    elif cfg["dataset"] == "cub":
        # Large datasets are user-provisioned. This path never downloads them.
        train = datasets.ImageFolder(str(Path(root) / "train"))
        test = datasets.ImageFolder(str(Path(root) / "test"))
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("train/test class folders must have the same names")
    elif cfg["dataset"] == "places365":
        # Use the official file lists: first-level folders are letters, not classes.
        # Hold out validation examples from train-standard; official val is test here.
        train = datasets.Places365(root, split="train-standard", small=True, download=False)
        test = datasets.Places365(root, split="val", small=True, download=False)
    elif cfg["dataset"] == "imagenet":
        # The mirror subset already has a fixed adapter-validation split.
        # Never resample it when an experiment seed changes.
        train = ImageNetTrain(Path(root))
        test = datasets.ImageFolder(cfg["official_val_root"])
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("ImageNet train and official validation WNIDs differ")
        indices = {
            "train": np.arange(train.train_count),
            "val": np.arange(train.train_count, len(train)),
            "test": np.arange(len(test)),
        }
        for name, count in (("train", cfg["train_per_class"]), ("val", cfg["val_per_class"])):
            labels = np.asarray(train.targets)[indices[name]]
            if not np.all(np.bincount(labels, minlength=len(train.classes)) == count):
                raise ValueError(f"ImageNet {name} does not contain {count} images per class")
        return train, test, indices
    else:
        raise ValueError("Supported datasets: cifar10, cifar100, cub, places365, imagenet")
    rng = np.random.default_rng(split_seed(config))
    if cfg["val_per_class"] < 1:
        raise ValueError("A positive val_per_class is required for model selection")
    val_indices, available = stratified_indices(train.targets, cfg["val_per_class"], rng)
    selected, _ = stratified_indices(
        np.asarray(train.targets)[available], cfg["train_per_class"], rng
    )
    train_indices = available[selected]
    test_indices, _ = stratified_indices(test.targets, cfg["test_per_class"], rng)
    return train, test, {"train": train_indices, "val": val_indices, "test": test_indices}


def prepare_data(config: dict, device: str) -> tuple[dict, Path]:
    concepts = Path(config["data"]["concepts"]).read_text().splitlines()
    # Include data content identity, not just the concept file's basename.
    identity = {
        "format": 2,
        "data": config["data"],
        "seed": split_seed(config),
        "backbone": config["backbone"],
        "teacher": config["teacher"],
        "concepts": concepts,
        "torchvision": __import__("torchvision").__version__,
    }
    if config["data"]["dataset"] == "imagenet":
        subset = Path(config["data"]["root"])
        if not (subset / "COMPLETE").exists():
            raise ValueError("ImageNet subset download is not complete")
        identity["subset_manifest_sha256"] = hashlib.sha256(
            (subset / "manifest.jsonl").read_bytes()
        ).hexdigest()
    cache_path = Path(config["paths"]["cache"]) / f"features-{digest(identity)[:16]}.pt"
    if cache_path.exists():
        print(f"Reusing {cache_path}", flush=True)
        return read_cache(cache_path), cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    train, test, indices = make_splits(config)
    native = config["teacher"].get("precision", "float32") == "native"
    teacher, teacher_transform = clip.load(
        config["teacher"]["name"], device=device if native else "cpu", jit=False
    )
    teacher = (teacher if native else teacher.float()).to(device).eval().requires_grad_(False)
    backbone = load_backbone(config["backbone"], device)
    target_transform = target_preprocess(config["backbone"]["name"])

    result = {"identity": identity, "concepts": concepts, "classes": train.classes, "splits": {}}
    with torch.no_grad():
        # Large concept dictionaries should not be encoded in one GPU batch.
        text = torch.cat(
            [
                teacher.encode_text(tokens.to(device)).float()
                for tokens in clip.tokenize(concepts).split(
                    config["data"]["batch_size"] if native else 128
                )
            ]
        )
        # Upstream saves native CLIP embeddings, then normalizes and multiplies
        # them in FP32 on CPU. Keep that order for official training runs.
        text = F.normalize(text.cpu() if native else text, dim=-1)
        result["text"] = text.cpu()
        for split, selection in indices.items():
            source = test if split == "test" else train
            paired = PairedImages(source, selection, target_transform, teacher_transform)
            loader = DataLoader(
                paired,
                batch_size=config["data"]["batch_size"],
                shuffle=False,
                num_workers=config["data"].get("num_workers", 0),
                pin_memory=device.startswith("cuda"),
            )
            features, targets, labels = [], [], []
            offset = 0
            for target_images, teacher_images, y in tqdm(loader, desc=f"Extract {split}"):
                if config["data"]["cache_images"]:
                    if offset == 0:
                        image_shape = (len(paired), *target_images.shape[1:])
                        image_path = cache_path.with_name(f"{cache_path.stem}-{split}-images.bin")
                        image_store = torch.from_file(
                            str(image_path), shared=True, size=math.prod(image_shape)
                        ).reshape(image_shape)
                    image_store[offset : offset + len(y)] = target_images
                internal = backbone.prefix(target_images.to(device))
                features.append(backbone.suffix(internal).float().cpu())
                visual = teacher.encode_image(teacher_images.to(device)).float()
                visual = F.normalize(visual.cpu() if native else visual, dim=-1)
                targets.append((visual @ text.T).cpu())
                labels.append(y)
                if config["data"]["cache_internal"]:
                    if offset == 0:
                        internal_shape = (len(paired), *internal.shape[1:])
                        internal_path = cache_path.with_name(f"{cache_path.stem}-{split}.bin")
                        internal_dtype = (
                            internal.dtype
                            if config["data"].get("internal_dtype") == "native"
                            else torch.float32
                        )
                        # Stream full-precision prefixes to a memory-mapped file;
                        # clean and noisy paths use exactly the same features.
                        internal_store = torch.from_file(
                            str(internal_path),
                            shared=True,
                            size=math.prod(internal_shape),
                            dtype=internal_dtype,
                        ).reshape(internal_shape)
                    internal_store[offset : offset + len(y)] = internal.cpu()
                offset += len(y)
            split_data = {
                "features": torch.cat(features),
                "teacher": torch.cat(targets),
                "labels": torch.cat(labels),
                "indices": torch.tensor(selection),
            }
            if config["data"].get("input_mode", "cache") == "stream":
                # Places365 exposes its file list as imgs; ImageFolder uses samples.
                paths = source.imgs if config["data"]["dataset"] == "places365" else source.samples
                split_data["image_paths"] = [paths[int(index)][0] for index in selection]
            if config["data"]["cache_internal"]:
                split_data["internal_file"] = str(internal_path)
                split_data["internal_shape"] = internal_shape
                split_data["internal_dtype"] = str(internal_dtype).removeprefix("torch.")
                del internal_store
            if config["data"]["cache_images"]:
                split_data["images_file"] = str(image_path)
                split_data["images_shape"] = image_shape
                del image_store
            result["splits"][split] = split_data
    result["split_hash"] = digest({k: v.tolist() for k, v in indices.items()})
    result["concept_hash"] = digest(concepts)
    torch.save(result, cache_path)
    return read_cache(cache_path), cache_path
