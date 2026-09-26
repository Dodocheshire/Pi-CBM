import torch
from clip.model import ModifiedResNet
from torch import nn
from torchvision import models

from pi_cbm.models.backbones import CLIPResNetSplit, ResNetSplit


def test_clip_split_matches_original_forward_without_downloading_weights():
    original = ModifiedResNet(
        layers=(1, 1, 1, 1), output_dim=6, heads=4, input_resolution=32, width=8
    ).eval()
    split = CLIPResNetSplit(original).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(split(images), original(images), rtol=0, atol=0)
    assert split.point == "pre_attnpool"


def test_resnet_split_matches_original_feature_extractor():
    torch.manual_seed(8)
    original = models.resnet18(weights=None).eval()
    original.fc = nn.Identity()
    torch.manual_seed(8)
    split = ResNetSplit("resnet18", pretrained=False).eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(split(images), original(images), rtol=0, atol=0)
    assert split.point == "pre_layer4"
