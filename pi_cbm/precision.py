"""Use BF16 for adapter forward passes while keeping the LF-CBM fit unchanged."""

from contextlib import nullcontext

import torch


def adapter_autocast(config: dict):
    if config["training"]["precision"] == "bfloat16" and config["experiment"]["device"].startswith(
        "cuda"
    ):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()
