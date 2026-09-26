import pytest
import torch
from torch import nn

from pi_cbm.generators import NoiseGenerator
from pi_cbm.injections import build_injection
from pi_cbm.models import ConceptBottleneck, NoisyCBM


@pytest.fixture
def make_model():
    def factory(site="target_global", parameterization="mean_scale", architecture="mlp"):
        torch.manual_seed(7)
        cbm = ConceptBottleneck(6, 4, 3).requires_grad_(False)
        suffix = (
            nn.Sequential(
                nn.Conv2d(3 if site == "target_image" else 5, 6, 1),
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
            )
            .eval()
            .requires_grad_(False)
        )
        channels = {"target_global": 6, "target_internal": 5, "target_image": 3, "baseline": 6}[
            site
        ]
        config = {
            "generator": architecture,
            "parameterization": parameterization,
            "hidden_dim": 8,
            "initial_scale": 0.1,
            "image_grid": 2,
        }
        generator = (
            None
            if site == "baseline"
            else NoiseGenerator(channels, torch.randn(4, 7), torch.ones(channels), config)
        )
        return NoisyCBM(cbm, build_injection(site), generator, suffix)

    return factory
