"""Verify dataset labels and train/validation isolation without real downloads."""

from collections import Counter
from types import SimpleNamespace

from PIL import Image

from pi_cbm.config import load_config
from pi_cbm.data import make_splits


def test_cub_folders_keep_official_test_separate(tmp_path):
    for split, count in [("train", 6), ("test", 3)]:
        for category in ["001.bird_a", "002.bird_b"]:
            directory = tmp_path / split / category
            directory.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (2, 2)).save(directory / f"{index}.jpg")
    config = load_config("configs/server/cub.yaml")
    config["data"].update(root=str(tmp_path), val_per_class=2)
    train, test, indices = make_splits(config)
    assert len(train) == 12 and len(test) == 6
    assert train.class_to_idx == test.class_to_idx
    assert not set(indices["train"]) & set(indices["val"])
    assert len(indices["train"]) == 8 and len(indices["val"]) == 4
    assert set(indices["test"]) == set(range(6))


def test_places_uses_official_labels_and_separate_evaluation_split(monkeypatch):
    calls = []

    def fake_places(root, split, small, download):
        calls.append((split, small, download))
        count = 7 if split == "train-standard" else 3
        return SimpleNamespace(
            classes=["/a/airport", "/b/beach"],
            targets=[label for label in range(2) for _ in range(count)],
        )

    monkeypatch.setattr("pi_cbm.data.datasets.Places365", fake_places)
    config = load_config("configs/server/places365.yaml")
    config["data"].update(train_per_class=3, val_per_class=2)
    train, test, indices = make_splits(config)
    assert calls == [("train-standard", True, False), ("val", True, False)]
    assert not set(indices["train"]) & set(indices["val"])
    assert Counter(train.targets[i] for i in indices["train"]) == {0: 3, 1: 3}
    assert Counter(train.targets[i] for i in indices["val"]) == {0: 2, 1: 2}
    assert len(test.targets) == 6 and set(indices["test"]) == set(range(6))
    # Adapter seeds must not change this dataset selection.
    config["experiment"]["seed"] = 2
    _, _, repeated = make_splits(config)
    for key in indices:
        assert indices[key].tolist() == repeated[key].tolist()


def test_imagenet_uses_preselected_train_and_adapter_validation(tmp_path):
    subset = tmp_path / "subset"
    official = tmp_path / "official_val"
    for root, split, count in (
        (subset, "train", 3),
        (subset, "adapter_val", 2),
        (official, "", 2),
    ):
        for wnid in ("n00000001", "n00000002"):
            directory = root / split / wnid
            directory.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (4, 4)).save(directory / f"{index}.JPEG")
    config = load_config("configs/server/imagenet.yaml")
    config["data"].update(
        root=str(subset),
        official_val_root=str(official),
        train_per_class=3,
        val_per_class=2,
    )
    train, test, indices = make_splits(config)
    assert train.class_to_idx == test.class_to_idx
    assert len(indices["train"]) == 6
    assert len(indices["val"]) == 4
    assert len(indices["test"]) == 4
    assert not set(indices["train"]) & set(indices["val"])
    assert Counter(train.targets[i] for i in indices["train"]) == {0: 3, 1: 3}
    assert Counter(train.targets[i] for i in indices["val"]) == {0: 2, 1: 2}
    assert all("adapter_val" in train.samples[i][0] for i in indices["val"])
    config["experiment"]["seed"] = 7
    _, _, repeated = make_splits(config)
    for key in indices:
        assert (indices[key] == repeated[key]).all()
