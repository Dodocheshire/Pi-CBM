import pytest
import torch

from pi_cbm.evaluation import predict
from pi_cbm.randomness import seeded


@pytest.mark.parametrize("architecture", ["mlp", "concept_cross_attention"])
def test_learned_distribution_starts_at_zero_mean_and_configured_scale(make_model, architecture):
    model = make_model(architecture=architecture)
    tokens = torch.randn(3, 1, 6)
    mean, scale = model.generator.distribution(tokens)
    torch.testing.assert_close(mean, torch.zeros_like(tokens))
    torch.testing.assert_close(scale, torch.full_like(tokens, 0.1))
    with seeded(4):
        first = model.generator(tokens)[0]
    with seeded(4):
        second = model.generator(tokens)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_fixed_gaussian_has_no_trainable_generator(make_model):
    model = make_model(parameterization="fixed_gaussian")
    assert not list(model.generator.parameters())
    mean, scale = model.generator.distribution(torch.ones(2, 1, 6))
    assert torch.count_nonzero(mean) == 0
    torch.testing.assert_close(scale, torch.full_like(scale, 0.1))


def test_image_noise_is_sampled_per_pixel_and_gradients_cross_frozen_backbone(make_model):
    model = make_model("target_image", architecture="concept_cross_attention").train()
    images = torch.randn(2, 3, 4, 4)
    before = {name: value.clone() for name, value in model.suffix.state_dict().items()}
    with seeded(6):
        logits, concepts, energy = model(images)
    assert logits.shape == (2, 3) and concepts.shape == (2, 4)
    assert energy > 0
    (logits.square().mean() + energy).backward()
    assert sum(p.grad.abs().sum() for p in model.generator.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.suffix.parameters())
    for name, value in model.suffix.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("site", ["target_global", "target_internal", "target_image"])
def test_negligible_noise_recovers_clean_path(make_model, site):
    model = make_model(site).eval()
    torch.nn.init.constant_(model.generator.scale_head.bias, -100)
    inputs = (
        torch.randn(3, 6)
        if site == "target_global"
        else torch.randn(3, 3 if site == "target_image" else 5, 4, 4)
    )
    features = inputs if site == "target_global" else model.suffix(inputs)
    expected_logits, expected_concepts = model.cbm(features)
    logits, concepts, _ = model(inputs)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(concepts, expected_concepts, rtol=0, atol=0)


def test_concept_attention_uses_text_and_intervention_runs_after_noise(make_model):
    model = make_model(architecture="concept_cross_attention").eval()
    torch.nn.init.normal_(model.generator.mean_head.weight)
    inputs = torch.randn(3, 6)
    with seeded(3):
        first = model(inputs)[0]
    model.generator.text.add_(2)
    with seeded(3):
        second = model(inputs)[0]
    assert not torch.allclose(first, second)
    probabilities, concepts = predict(model, inputs, (torch.arange(4), 0.0))
    assert torch.count_nonzero(concepts) == 0
    torch.testing.assert_close(probabilities, model.cbm.head.bias.softmax(-1).expand(3, -1))
