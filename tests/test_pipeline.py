import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from pi_cbm.config import load_config
from pi_cbm.data import prepare_data, read_cache, stratified_indices
from pi_cbm.evaluation import evaluate, predict
from pi_cbm.evaluation.checkpoint import ImagePredictor, restore_model
from pi_cbm.models.import_lfcbm import import_lfcbm
from pi_cbm.pipeline import build_model, cpu_state
from pi_cbm.randomness import seeded
from pi_cbm.training.adapt import adapt


def small_data():
    torch.manual_seed(42)
    projection = torch.randn(6, 4)
    classifier = torch.randn(4, 3)
    data = {
        "classes": ["a", "b", "c"],
        "concepts": ["c0", "c1", "c2", "c3"],
        "text": torch.randn(4, 7),
        "splits": {},
    }
    for name, count in (("train", 60), ("val", 30), ("test", 30)):
        features = torch.randn(count, 6)
        teacher = features @ projection
        data["splits"][name] = {
            "features": features,
            "teacher": teacher,
            "labels": (teacher @ classifier).argmax(1),
        }
    return data


def test_removed_options_are_rejected_before_training():
    for override in (
        "noise.parameterization=mean",
        "noise.parameterization=scale",
        "injection.site=concept",
        "data.dataset=imagefolder",
    ):
        with pytest.raises(ValueError):
            load_config("configs/base.yaml", [override])
    with pytest.raises(KeyError):
        load_config("configs/base.yaml", ["noise.conditioning=local_global"])
    with pytest.raises(KeyError):
        load_config("configs/base.yaml", ["inference.mode=mean_probs"])
    with pytest.raises(ValueError, match="cache_images"):
        load_config("configs/base.yaml", ["injection.site=target_image", "injection.point=input"])


def test_stratified_split_is_disjoint_and_reproducible():
    labels = np.repeat(np.arange(3), 20)
    selected, remaining = stratified_indices(labels, 5, np.random.default_rng(7))
    assert not set(selected) & set(remaining)
    assert np.bincount(labels[selected]).tolist() == [5, 5, 5]
    second, _ = stratified_indices(labels, 5, np.random.default_rng(7))
    np.testing.assert_array_equal(selected, second)


def test_generator_initialization_is_independent_of_cache_rng(make_model):
    config = load_config("configs/experiment/cifar10_global.yaml", ["experiment.device=cpu"])
    data = small_data()
    cbm = make_model().cbm
    first = build_model(cbm, data, torch.arange(4), config, "cpu")
    torch.randn(1000)
    second = build_model(cbm, data, torch.arange(4), config, "cpu")
    for name, value in first.generator.state_dict().items():
        torch.testing.assert_close(value, second.generator.state_dict()[name], rtol=0, atol=0)


def test_nec_counts_effective_head_weights_per_class(make_model):
    model = make_model(site="baseline")
    with torch.no_grad():
        model.cbm.head.weight.copy_(
            torch.tensor([[0.0, 0.5, 0.0, 0.0], [1.0, 1e-6, 0.0, -0.2], [0.0, 0.0, 0.0, -1.0]])
        )
        model.cbm.head.bias.fill_(1.0)
    split = small_data()["splits"]["test"]
    clean_concepts = model.cbm.concepts(split["features"])
    config = load_config("configs/base.yaml", ["experiment.device=cpu"])
    metrics = evaluate(model, split, clean_concepts, torch.arange(4), config, "cpu")
    # The tiny weight counts as strictly nonzero, but not as an effective link.
    assert metrics["head_nonzero"] == 5
    assert metrics["nec"] == pytest.approx(4 / 3)


def test_gradient_clip_zero_prevents_generator_update(make_model):
    data = small_data()
    reference = make_model(site="baseline").cbm
    clean = {
        name: reference.concepts(split["features"])
        for name, split in data["splits"].items()
    }
    frozen, trainable = make_model(), make_model()
    initial = deepcopy(frozen.generator.state_dict())
    config = load_config(
        "configs/base.yaml",
        ["experiment.device=cpu", "training.epochs=1", "training.batch_size=30"],
    )
    config["training"]["grad_clip_norm"] = 0.0
    with seeded(11):
        adapt(frozen, data, clean, torch.arange(4), config, "cpu")
    config["training"]["grad_clip_norm"] = 1.0
    with seeded(11):
        adapt(trainable, data, clean, torch.arange(4), config, "cpu")
    for name, value in initial.items():
        torch.testing.assert_close(frozen.generator.state_dict()[name], value, rtol=0, atol=0)
    assert any(
        not torch.equal(trainable.generator.state_dict()[name], value)
        for name, value in initial.items()
    )


@pytest.mark.parametrize("stage", ["generator_only", "refit_head", "finetune_projection"])
def test_training_stages_freeze_expected_modules_and_restore_checkpoint(make_model, stage):
    torch.set_num_threads(2)
    model = make_model()
    data = small_data()
    config = load_config(
        "configs/base.yaml",
        [
            f"training.stage={stage}",
            "training.epochs=2",
            "training.batch_size=30",
            "baseline.head.steps=3",
            "baseline.head.batch_size=16",
            "experiment.device=cpu",
            "noise.generator=mlp",
            "noise.hidden_dim=8",
            "noise.initial_scale=0.1",
        ],
    )
    with torch.no_grad():
        clean = {
            name: model.cbm.concepts(split["features"]) for name, split in data["splits"].items()
        }
    original = deepcopy(model.cbm.state_dict())
    history, refit = adapt(model, data, clean, torch.arange(4), config, "cpu")
    assert len(history) == 2
    assert (refit is not None) == (stage != "generator_only")
    assert torch.equal(model.cbm.projection.weight, original["projection.weight"]) == (
        stage != "finetune_projection"
    )
    assert torch.equal(model.cbm.head.weight, original["head.weight"]) == (
        stage == "generator_only"
    )
    saved = {"config": config, "cbm": cpu_state(model.cbm), "generator": cpu_state(model.generator)}
    restored = restore_model(saved, "cpu", nn.Identity())
    with seeded(80):
        first = predict(model, data["splits"]["test"]["features"])[0]
    with seeded(80):
        second = predict(restored, data["splits"]["test"]["features"])[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_import_preserves_logits_and_concept_order(make_model, tmp_path):
    model = make_model().cbm
    for name, tensor in {
        "W_c": model.projection.weight,
        "W_g": model.head.weight,
        "b_g": model.head.bias,
        "proj_mean": model.concept_mean[None],
        "proj_std": model.concept_std[None],
    }.items():
        torch.save(tensor, tmp_path / f"{name}.pt")
    (tmp_path / "args.txt").write_text(json.dumps({"backbone": "resnet18"}))
    (tmp_path / "concepts.txt").write_text("c2\nc0\nc3\nc1\n")
    data = small_data()
    imported, selected, _ = import_lfcbm(tmp_path, data, "resnet18", "cpu")
    assert selected.tolist() == [2, 0, 3, 1]
    features = data["splits"]["test"]["features"]
    torch.testing.assert_close(imported(features)[0], model(features)[0], rtol=0, atol=0)


def test_spatial_and_image_cache_round_trip(tmp_path):
    split = {}
    for name in ("internal", "images"):
        values = torch.randn(4, 3, 2, 2)
        path = tmp_path / f"{name}.bin"
        mapped = torch.from_file(str(path), shared=True, size=values.numel()).reshape(values.shape)
        mapped.copy_(values)
        split[f"{name}_file"] = str(path)
        split[f"{name}_shape"] = values.shape
        split[f"{name}_expected"] = values.clone()
    cache = tmp_path / "cache.pt"
    torch.save({"splits": {"train": split}}, cache)
    restored = read_cache(cache)["splits"]["train"]
    for name in ("internal", "images"):
        torch.testing.assert_close(restored[name], split[f"{name}_expected"], rtol=0, atol=0)


def test_image_preparation_caches_the_exact_backbone_input(tmp_path, monkeypatch):
    class TinyImages:
        classes = ["a", "b"]
        targets = [0, 0, 1, 1]

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return Image.new("RGB", (4, 4), (index * 40, 20, 10)), self.targets[index]

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.prefix = nn.Identity()
            self.suffix = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))

    class TinyTeacher(nn.Module):
        def encode_text(self, tokens):
            return torch.cat([tokens.float(), tokens.float() + 1], dim=1)

        def encode_image(self, images):
            return images.mean(dim=(2, 3))[:, :2]

    def preprocess(image):
        return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255

    dataset = TinyImages()
    monkeypatch.setattr(
        "pi_cbm.data.make_splits",
        lambda _: (
            dataset,
            dataset,
            {"train": np.array([0, 2]), "val": np.array([1, 3]), "test": np.array([0, 2])},
        ),
    )
    monkeypatch.setattr("pi_cbm.data.load_backbone", lambda *_: TinyBackbone())
    monkeypatch.setattr("pi_cbm.data.target_preprocess", lambda _: preprocess)
    monkeypatch.setattr("pi_cbm.data.clip.load", lambda *_, **__: (TinyTeacher(), preprocess))
    monkeypatch.setattr(
        "pi_cbm.data.clip.tokenize", lambda concepts: torch.arange(1, len(concepts) + 1)[:, None]
    )
    config = load_config(
        "configs/experiment/cifar10_image.yaml", ["experiment.device=cpu", "data.batch_size=2"]
    )
    config["paths"]["cache"] = str(tmp_path)
    prepared, cache = prepare_data(config, "cpu")
    expected = torch.stack([preprocess(dataset[index][0]) for index in (0, 2)])
    torch.testing.assert_close(prepared["splits"]["train"]["images"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        prepared["splits"]["train"]["features"], expected.mean((2, 3)), rtol=0, atol=0
    )
    assert "internal" not in prepared["splits"]["train"]
    reused, same_cache = prepare_data(config, "cpu")
    assert same_cache == cache
    torch.testing.assert_close(reused["splits"]["train"]["images"], expected, rtol=0, atol=0)


def test_image_predictor_matches_cached_image_path(make_model, tmp_path, monkeypatch):
    model = make_model("target_image").eval()
    config = load_config("configs/experiment/cifar10_image.yaml", ["experiment.device=cpu"])
    config["noise"].update(generator="mlp", hidden_dim=8, initial_scale=0.1, image_grid=2)
    saved = {
        "config": config,
        "cbm": cpu_state(model.cbm),
        "generator": cpu_state(model.generator),
        "concepts": ["a", "b", "c", "d"],
    }

    class ToyBackbone(nn.Module):
        def __init__(self, suffix):
            super().__init__()
            self.prefix = nn.Identity()
            self.suffix = suffix

        def forward(self, images):
            return self.suffix(self.prefix(images))

    monkeypatch.setattr(
        "pi_cbm.evaluation.checkpoint.load_backbone", lambda *_: ToyBackbone(model.suffix)
    )
    torch.save(saved, tmp_path / "checkpoint.pt")
    predictor = ImagePredictor(tmp_path, "cpu")
    images = torch.randn(3, 3, 4, 4)
    with seeded(7):
        expected = predict(model, images)
    with seeded(7):
        actual = predictor(images)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
