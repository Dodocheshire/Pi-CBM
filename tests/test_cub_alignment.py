import torch
from PIL import Image
from pytorchcv.model_provider import get_model

from pi_cbm.models.backbones import CUBResNetSplit, load_backbone, target_preprocess


def test_cub_split_matches_original_features_and_frozen_suffix_keeps_gradients():
    original = get_model("resnet18_cub", pretrained=False).eval().requires_grad_(False)
    split = CUBResNetSplit(original).eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        expected = original.features(images).flatten(1)
        actual = split(images)
        internal = split.prefix(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, split.feature_dim) == (2, 512)
    assert internal.shape == (2, split.internal_channels, 14, 14) == (2, 256, 14, 14)
    assert split.point == "pre_stage4"
    internal.requires_grad_(True)
    split.suffix(internal).square().mean().backward()
    assert torch.isfinite(internal.grad).all() and internal.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in original.parameters())


def test_cub_loader_requests_task_trained_weights_and_freezes(monkeypatch):
    def offline_model(name, pretrained):
        assert name == "resnet18_cub" and pretrained is True
        return get_model(name, pretrained=False)

    monkeypatch.setattr("pytorchcv.model_provider.get_model", offline_model)
    backbone = load_backbone({"name": "resnet18_cub"}, "cpu")
    assert isinstance(backbone, CUBResNetSplit)
    assert not any(module.training for module in backbone.modules())
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_cub_preprocessing_uses_original_size_and_normalization():
    output = target_preprocess("resnet18_cub")(Image.new("RGB", (320, 280), (255, 0, 0)))
    expected = (torch.tensor([1.0, 0.0, 0.0]) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
        [0.229, 0.224, 0.225]
    )
    assert output.shape == (3, 224, 224)
    torch.testing.assert_close(output, expected[:, None, None].expand_as(output))
