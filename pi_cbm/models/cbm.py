"""Every prediction passes through the same named concept bottleneck."""

import torch
from torch import nn


class ConceptBottleneck(nn.Module):
    def __init__(self, feature_dim: int, concepts: int, classes: int):
        super().__init__()
        self.projection = nn.Linear(feature_dim, concepts, bias=False)
        self.head = nn.Linear(concepts, classes)
        self.register_buffer("concept_mean", torch.zeros(concepts))
        self.register_buffer("concept_std", torch.ones(concepts))

    def concepts(self, features: torch.Tensor) -> torch.Tensor:
        return (self.projection(features) - self.concept_mean) / self.concept_std

    def forward(self, features: torch.Tensor):
        concepts = self.concepts(features)
        return self.head(concepts), concepts


class NoisyCBM(nn.Module):
    def __init__(self, cbm, injection, generator, suffix):
        super().__init__()
        self.cbm = cbm
        self.injection = injection
        self.generator = generator
        self.suffix = suffix

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen BatchNorm statistics must not change during generator training.
        self.suffix.eval()
        return self

    def encode_concepts(self, inputs: torch.Tensor):
        return self.injection(self, inputs)

    def forward(self, inputs: torch.Tensor):
        concepts, energy = self.encode_concepts(inputs)
        return self.cbm.head(concepts), concepts, energy
