"""Serve cached features or decode source images for one model input batch."""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader

from .data import digest
from .models.backbones import load_backbone, target_preprocess
from .precision import adapter_autocast


class ImagePaths(Dataset):
    """Decode only the images requested by a training or evaluation batch."""

    def __init__(self, paths: list[str], backbone_name: str):
        self.paths = paths
        self.transform = target_preprocess(backbone_name)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(default_loader(self.paths[index]))


class ModelInputs(Dataset):
    """Keep cached labels and concept targets aligned with their source image."""

    def __init__(self, split: dict, model, config: dict, *fields: torch.Tensor):
        key = model.injection.cache_key
        self.stream = config["data"].get("input_mode", "cache") == "stream" and key not in split
        self.inputs = (
            ImagePaths(split["image_paths"], config["backbone"]["name"])
            if self.stream
            else split[key]
        )
        self.fields = fields

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (self.inputs[index], *(field[index] for field in self.fields))


def input_loader(
    split: dict,
    model,
    config: dict,
    *fields: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = ModelInputs(split, model, config, *fields)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=config["data"].get("num_workers", 0) if dataset.stream else 0,
        pin_memory=config["experiment"]["device"].startswith("cuda"),
    )


def cache_internal_inputs(data: dict, config: dict, device: str) -> None:
    """Store the frozen pre-layer4 output separately from the clean CBM cache.

    The clean feature cache and its fitted sparse head remain reusable when this
    option changes. BF16 halves disk traffic for a BF16 adapter run.
    """
    cache = Path(config["paths"]["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    identity = digest(
        [data["identity"], data["split_hash"], config["backbone"], config["training"]["precision"]]
    )[:16]
    dtype = torch.bfloat16 if config["training"]["precision"] == "bfloat16" else torch.float32
    backbone = load_backbone(config["backbone"], device)
    for name, split in data["splits"].items():
        stem = cache / f"internal-{identity}-{name}"
        binary = stem.with_suffix(".bin")
        metadata = stem.with_suffix(".json")
        if not metadata.exists():
            images = ImagePaths(split["image_paths"], config["backbone"]["name"])
            loader = DataLoader(
                images,
                batch_size=config["data"]["batch_size"],
                num_workers=config["data"]["num_workers"],
                pin_memory=device.startswith("cuda"),
            )
            offset = 0
            with torch.no_grad(), adapter_autocast(config):
                for batch in loader:
                    values = backbone.prefix(batch.to(device, non_blocking=True))
                    if offset == 0:
                        shape = (len(images), *values.shape[1:])
                        temporary = stem.with_suffix(".bin.part")
                        storage = torch.from_file(
                            str(temporary), shared=True, size=math.prod(shape), dtype=dtype
                        ).reshape(shape)
                    storage[offset : offset + len(batch)] = values.to(dtype).cpu()
                    offset += len(batch)
            del storage
            temporary.replace(binary)
            metadata.write_text(
                json.dumps({"shape": shape, "dtype": str(dtype).removeprefix("torch.")})
            )
        saved = json.loads(metadata.read_text())
        split["internal"] = torch.from_file(
            str(binary),
            shared=True,
            size=math.prod(saved["shape"]),
            dtype=getattr(torch, saved["dtype"]),
        ).reshape(saved["shape"])
