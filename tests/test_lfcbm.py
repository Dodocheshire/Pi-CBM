import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from clip.model import ModifiedResNet, convert_weights
from torch.nn import functional as F

from pi_cbm.data import read_cache
from pi_cbm.models.backbones import CLIPResNetSplit
from pi_cbm.randomness import seeded
from pi_cbm.training.lfcbm import cbm_training_data, fit_lfcbm
from pi_cbm.vendor.lfcbm import elasticnet, similarity


def test_upstream_files_are_unmodified():
    manifest = json.loads(Path("pi_cbm/vendor/lfcbm/provenance.json").read_text())
    for record in manifest["files"].values():
        assert (
            hashlib.sha256(Path(record["local_path"]).read_bytes()).hexdigest() == record["sha256"]
        )


def test_full_cbm_training_set_is_reassembled_in_original_order():
    def split(indices):
        values = torch.tensor(indices)
        return {
            "indices": values,
            "labels": values,
            "features": values[:, None],
            "teacher": values[:, None],
        }

    data = {"splits": {"train": split([4, 0, 2]), "val": split([3, 1])}}
    result = cbm_training_data(data, True)
    assert result["indices"].tolist() == list(range(5))
    assert result["features"].flatten().tolist() == list(range(5))
    assert cbm_training_data(data, False) is data["splits"]["train"]


def test_native_clip_split_and_half_cache_preserve_features(tmp_path):
    torch.set_num_threads(2)
    visual = ModifiedResNet((1, 1, 1, 1), 8, 2, input_resolution=32, width=8).eval()
    convert_weights(visual)
    backbone = CLIPResNetSplit(visual).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = visual(images.half()).float()
        actual = backbone(images)
        internal = backbone.prefix(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert internal.dtype == torch.float16 and actual.dtype == torch.float32
    path = tmp_path / "internal.bin"
    mapped = torch.from_file(str(path), shared=True, size=internal.numel(), dtype=torch.float16)
    mapped.copy_(internal.flatten())
    cache = tmp_path / "cache.pt"
    torch.save(
        {
            "splits": {
                "test": {
                    "internal_file": str(path),
                    "internal_shape": internal.shape,
                    "internal_dtype": "float16",
                }
            }
        },
        cache,
    )
    restored = read_cache(cache)["splits"]["test"]["internal"]
    torch.testing.assert_close(restored, internal, rtol=0, atol=0)
    values = restored.float().requires_grad_(True)
    backbone.requires_grad_(False).suffix(values).square().mean().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0


def test_fresh_training_matches_the_actual_upstream_script(tmp_path, monkeypatch):
    """Run the published train_cbm.py on tiny cached features, not a second reimplementation."""
    torch.set_num_threads(2)
    with seeded(123):
        text = torch.randn(4, 5)
        source_projection = torch.randn(6, 5)
        data = {"classes": ["a", "b", "c"], "concepts": ["c0", "c1", "c2", "c3"], "splits": {}}
        for name, count in (("train", 24), ("test", 12)):
            features = torch.randn(count, 6)
            visual = features @ source_projection
            labels = torch.arange(count) % 3
            torch.save(features, tmp_path / f"{name}-target.pt")
            torch.save(visual, tmp_path / f"{name}-clip.pt")
            data["splits"][name] = {
                "features": features,
                "labels": labels,
                "indices": torch.arange(count),
                "teacher": F.normalize(visual, dim=1) @ F.normalize(text, dim=1).T,
            }
    torch.save(text, tmp_path / "text.pt")
    concepts = tmp_path / "concepts.txt"
    concepts.write_text("c0\nc1\nc2\nc3")
    classes = tmp_path / "classes.txt"
    classes.write_text("a\nb\nc")

    def names(*args):
        split = "train" if args[3].endswith("train") else "test"
        return (
            str(tmp_path / f"{split}-target.pt"),
            str(tmp_path / f"{split}-clip.pt"),
            str(tmp_path / "text.pt"),
        )

    monkeypatch.setitem(
        sys.modules,
        "utils",
        SimpleNamespace(save_activations=lambda **kw: None, get_save_names=names),
    )
    monkeypatch.setitem(
        sys.modules,
        "data_utils",
        SimpleNamespace(
            LABEL_FILES={"tiny": str(classes)},
            get_targets_only=lambda name: data["splits"][
                "train" if name.endswith("train") else "test"
            ]["labels"],
        ),
    )
    monkeypatch.setitem(sys.modules, "similarity", similarity)
    monkeypatch.setitem(sys.modules, "glm_saga", SimpleNamespace(elasticnet=elasticnet))
    monkeypatch.setitem(sys.modules, "glm_saga.elasticnet", elasticnet)
    spec = importlib.util.spec_from_file_location(
        "upstream_train", "tests/fixtures/train_cbm_upstream.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    args = upstream.parser.parse_args(
        [
            "--dataset",
            "tiny",
            "--device",
            "cpu",
            "--concept_set",
            str(concepts),
            "--save_dir",
            str(tmp_path / "original"),
            "--activation_dir",
            str(tmp_path),
            "--clip_cutoff",
            "-100",
            "--interpretability_cutoff",
            "-1",
            "--proj_steps",
            "53",
            "--proj_batch_size",
            "16",
            "--n_iters",
            "3",
            "--saga_batch_size",
            "8",
        ]
    )
    settings = {
        "activation_cutoff": -100,
        "interpretability_cutoff": -1,
        "projection_steps": 53,
        "projection_lr": 0.001,
        "projection_batch_size": 16,
        "include_adapter_val": False,
        "validation_split": "test",
        "head": {"steps": 3, "batch_size": 8, "lr": 0.1, "lambda": 0.0007, "alpha": 0.99},
    }
    with seeded(7):
        upstream.train_cbm_and_save(args)
    with seeded(7):
        model, selected, metrics = fit_lfcbm(data, {"baseline": settings}, "cpu")
    output = next((tmp_path / "original").iterdir())
    mapping = {
        "projection.weight": "W_c",
        "head.weight": "W_g",
        "head.bias": "b_g",
        "concept_mean": "proj_mean",
        "concept_std": "proj_std",
    }
    for name, filename in mapping.items():
        expected = torch.load(output / f"{filename}.pt", weights_only=True)
        torch.testing.assert_close(
            model.state_dict()[name], expected.reshape_as(model.state_dict()[name]), rtol=0, atol=0
        )
    assert [data["concepts"][i] for i in selected] == (
        output / "concepts.txt"
    ).read_text().splitlines()
    assert metrics["head"]["effective_lambda"] == pytest.approx(0.0007 / 0.99)
