"""Select a target tensor, sample its noise, then finish the frozen CBM path."""

import torch
from torch import nn
from torch.nn import functional as F


def to_tokens(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(1)
    return value.flatten(2).transpose(1, 2)


def from_tokens(tokens: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if len(shape) == 2:
        return tokens.squeeze(1)
    return tokens.transpose(1, 2).reshape(shape)


def channel_scale(values: torch.Tensor) -> torch.Tensor:
    """Channel RMS from training data; clamp only avoids division by zero."""
    # Accumulate in batches instead of materializing a large spatial cache.
    total = torch.zeros(values.shape[1])
    count = 0
    for batch in values.split(64):
        tokens = to_tokens(batch.float())
        total += tokens.square().sum(dim=(0, 1))
        count += tokens.shape[0] * tokens.shape[1]
    return (total / count).sqrt().clamp_min(1e-6)


class Injection(nn.Module):
    cache_key = "features"

    def finish(self, model, perturbed):
        return model.cbm.concepts(perturbed)

    def forward(self, model, inputs):
        tokens, energy = model.generator(to_tokens(inputs))
        return self.finish(model, from_tokens(tokens, inputs.shape)), energy


class InternalInjection(Injection):
    cache_key = "internal"

    def finish(self, model, perturbed):
        # No no_grad here: gradients must cross the frozen suffix to reach noise.
        return model.cbm.concepts(model.suffix(perturbed))


class ImageInjection(Injection):
    cache_key = "images"

    def forward(self, model, images):
        # The generator predicts a smooth spatial distribution on a small grid.
        # Sampling at image resolution keeps the perturbation stochastic per pixel.
        grid = model.generator.image_grid
        pooled = F.adaptive_avg_pool2d(images, (grid, grid))
        mean, scale = model.generator.distribution(to_tokens(pooled))
        mean = F.interpolate(
            from_tokens(mean, pooled.shape), size=images.shape[-2:], mode="bilinear"
        )
        scale = F.interpolate(
            from_tokens(scale, pooled.shape), size=images.shape[-2:], mode="bilinear"
        )
        noisy = images + mean + scale * torch.randn_like(images)
        feature_scale = model.generator.feature_scale[None, :, None, None]
        energy = ((mean / feature_scale).square() + (scale / feature_scale).square()).mean()
        return model.cbm.concepts(model.suffix(noisy)), energy


class BaselineInjection(Injection):
    def forward(self, model, inputs):
        return model.cbm.concepts(inputs), inputs.new_zeros(())


def build_injection(site: str) -> Injection:
    return {
        "baseline": BaselineInjection,
        "target_global": Injection,
        "target_internal": InternalInjection,
        "target_image": ImageInjection,
    }[site]()
