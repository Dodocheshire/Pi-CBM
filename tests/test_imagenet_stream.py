"""Exercise both spatial injection paths without a tensor image cache."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from pi_cbm.config import load_config
from pi_cbm.data import prepare_data
from pi_cbm.evaluation import evaluate
from pi_cbm.evaluation.checkpoint import restore_model
from pi_cbm.inputs import cache_internal_inputs, input_loader
from pi_cbm.models import ConceptBottleneck
from pi_cbm.pipeline import build_model, cpu_state
from pi_cbm.training.adapt import adapt


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = nn.Sequential(nn.Conv2d(3, 5, 1), nn.BatchNorm2d(5), nn.ReLU())
        self.suffix = nn.Sequential(
            nn.Conv2d(5, 6, 1), nn.BatchNorm2d(6), nn.AdaptiveAvgPool2d(1), nn.Flatten(1)
        )
        self.eval().requires_grad_(False)

    def forward(self, images):
        return self.suffix(self.prefix(images))


@pytest.mark.parametrize("site", ["target_image", "target_internal"])
def test_streamed_images_train_noise_and_keep_prefix_frozen(tmp_path, monkeypatch, site):
    def preprocess(image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255

    monkeypatch.setattr("pi_cbm.inputs.target_preprocess", lambda _: preprocess)
    monkeypatch.setattr("pi_cbm.pipeline.load_backbone", lambda *_: TinyBackbone())
    config = load_config(
        "configs/server/imagenet.yaml",
        [
            "experiment.device=cpu",
            "data.batch_size=2",
            "data.num_workers=0",
            "training.batch_size=2",
            "training.epochs=1",
            "training.stage=generator_only",
            "noise.hidden_dim=8",
            "noise.image_grid=2",
            f"injection.site={site}",
            f"injection.point={'input' if site == 'target_image' else 'pre_layer4'}",
        ],
    )
    config["paths"]["cache"] = str(tmp_path)
    data = {"identity": {"test": "fixed-images"}, "text": torch.randn(4, 7), "splits": {}}
    cbm = ConceptBottleneck(6, 4, 3)
    clean = {}
    for split, count in (("train", 4), ("val", 2)):
        paths = []
        for index in range(count):
            path = tmp_path / f"{split}-{index}.JPEG"
            Image.new("RGB", (4, 4), (index * 40 + 30, 60, 100)).save(path)
            paths.append(str(path))
        features = torch.randn(count, 6)
        data["splits"][split] = {
            "image_paths": paths,
            "features": features,
            "teacher": torch.randn(count, 4),
            "labels": torch.tensor([index % 3 for index in range(count)]),
        }
        clean[split] = cbm.concepts(features).detach()

    model = build_model(cbm, data, torch.arange(4), config, "cpu")
    assert len(list(tmp_path.glob("channel-scale-*.pt"))) == 1
    if model.prefix is not None:
        running_mean = model.prefix[1].running_mean.clone()
    history, refit = adapt(model, data, clean, torch.arange(4), config, "cpu")
    assert len(history) == 1 and refit is None
    assert torch.isfinite(torch.tensor(history[0]["train_loss"]))
    if model.prefix is not None:
        torch.testing.assert_close(model.prefix[1].running_mean, running_mean)
        monkeypatch.setattr("pi_cbm.evaluation.checkpoint.load_backbone", lambda *_: TinyBackbone())
        restored = restore_model(
            {
                "config": config,
                "cbm": cpu_state(model.cbm),
                "generator": cpu_state(model.generator),
            },
            "cpu",
        )
        assert restored.prefix is not None
        metrics = evaluate(
            restored, data["splits"]["val"], clean["val"], torch.arange(4), config, "cpu"
        )
        assert metrics["examples"] == 2


def test_imagenet_feature_cache_keeps_paths_and_tracks_subset_manifest(tmp_path, monkeypatch):
    subset = tmp_path / "subset"
    official = tmp_path / "official"
    for root, split, count in ((subset, "train", 2), (subset, "adapter_val", 1), (official, "", 1)):
        for wnid in ("n00000001", "n00000002"):
            directory = root / split / wnid
            directory.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (4, 4), (30 + index, 60, 100)).save(directory / f"{index}.JPEG")
    (subset / "COMPLETE").write_text("complete")
    manifest = subset / "manifest.jsonl"
    manifest.write_text("first selection")

    def preprocess(image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255

    class TinyTeacher(nn.Module):
        def encode_text(self, tokens):
            return torch.cat([tokens.float() + offset for offset in range(3)], dim=1)

        def encode_image(self, images):
            return images.mean(dim=(2, 3))

    class TinyFeatures(nn.Module):
        def __init__(self):
            super().__init__()
            self.prefix = nn.Identity()
            self.suffix = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))

    monkeypatch.setattr("pi_cbm.data.target_preprocess", lambda _: preprocess)
    monkeypatch.setattr("pi_cbm.data.load_backbone", lambda *_: TinyFeatures())
    monkeypatch.setattr("pi_cbm.data.clip.load", lambda *_, **__: (TinyTeacher(), preprocess))
    monkeypatch.setattr(
        "pi_cbm.data.clip.tokenize", lambda words: torch.arange(1, len(words) + 1)[:, None]
    )
    concepts = tmp_path / "concepts.txt"
    concepts.write_text("blue\ngreen")
    config = load_config("configs/server/imagenet.yaml", ["experiment.device=cpu"])
    config["data"].update(
        root=str(subset),
        official_val_root=str(official),
        concepts=str(concepts),
        train_per_class=2,
        val_per_class=1,
        batch_size=2,
        num_workers=0,
    )
    config["paths"]["cache"] = str(tmp_path / "cache")

    data, first_cache = prepare_data(config, "cpu")
    assert {name: len(split["labels"]) for name, split in data["splits"].items()} == {
        "train": 4,
        "val": 2,
        "test": 2,
    }
    assert "train" in data["splits"]["train"]["image_paths"][0]
    assert "adapter_val" in data["splits"]["val"]["image_paths"][0]
    assert "images" not in data["splits"]["train"]
    assert "internal" not in data["splits"]["train"]
    manifest.write_text("changed selection")
    _, second_cache = prepare_data(config, "cpu")
    assert second_cache != first_cache


def test_internal_sidecar_reuses_frozen_features_without_source_images(tmp_path, monkeypatch):
    def preprocess(image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255

    monkeypatch.setattr("pi_cbm.inputs.target_preprocess", lambda _: preprocess)
    monkeypatch.setattr("pi_cbm.inputs.load_backbone", lambda *_: TinyBackbone())
    config = load_config(
        "configs/server/imagenet.yaml",
        ["experiment.device=cpu", "data.batch_size=2", "data.num_workers=0"],
    )
    config["paths"]["cache"] = str(tmp_path / "cache")
    paths = []
    for index in range(3):
        path = tmp_path / f"image-{index}.JPEG"
        Image.new("RGB", (4, 4), (index * 30, 60, 100)).save(path)
        paths.append(str(path))
    data = {
        "identity": {"subset": "tiny"},
        "split_hash": "tiny-split",
        "splits": {"train": {"image_paths": paths}},
    }
    cache_internal_inputs(data, config, "cpu")
    first = data["splits"]["train"]["internal"].clone()
    assert first.shape == (3, 5, 4, 4)
    assert first.dtype == torch.bfloat16
    for path in paths:
        Path(path).unlink()
    cache_internal_inputs(data, config, "cpu")
    torch.testing.assert_close(data["splits"]["train"]["internal"], first, rtol=0, atol=0)
    model = SimpleNamespace(injection=SimpleNamespace(cache_key="internal"))
    loader = input_loader(data["splits"]["train"], model, config, batch_size=2)
    (batch,) = next(iter(loader))
    assert batch.shape == (2, 5, 4, 4)


def test_imagenet_quota_extension_preserves_selected_source_rows(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from scripts import download_imagenet_subset

    monkeypatch.setattr(download_imagenet_subset, "INITIAL_PER_CLASS", 2)
    monkeypatch.setattr(download_imagenet_subset, "EXTENSION_START_SHARD", 1)
    monkeypatch.setattr(download_imagenet_subset, "PER_CLASS", 3)
    selected = tmp_path / "selected"
    selected.mkdir()
    wnids = ["n00000001", "n00000002"]
    counts = dict.fromkeys(wnids, 0)
    for shard_index in (0, 1):
        shard = tmp_path / f"shard-{shard_index}.parquet"
        pq.write_table(
            pa.table(
                {
                    "image": [{"bytes": bytes([shard_index, row])} for row in range(8)],
                    "label": [row % 2 for row in range(8)],
                }
            ),
            shard,
        )
        download_imagenet_subset.extract_shard(shard, shard_index, selected, wnids, counts)
        assert set(counts.values()) == ({2} if shard_index == 0 else {3})
    for wnid in wnids:
        files = sorted((selected / wnid).glob("*.JPEG"))
        assert len(files) == 3
        assert sum(path.name.startswith("s00000") for path in files) == 2
