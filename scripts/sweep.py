"""Sequential Cartesian sweep; --dry-run only prints the resolved changes."""

import argparse
import itertools
from pathlib import Path

import yaml

from pi_cbm.config import load_config
from pi_cbm.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sweep/local.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sweep = yaml.safe_load(args.config.read_text())
    keys = list(sweep["grid"])
    seen = set()
    for combination in itertools.product(*sweep["grid"].values()):
        changes = dict(zip(keys, combination))
        overrides = [f"{key}={value}" for key, value in changes.items()]
        config = load_config(args.config.parent / sweep["base"], overrides)
        # Fixed Gaussian has no learned structure, so do not run duplicate
        # experiments merely because the generator architecture grid differs.
        identity = changes.copy()
        if config["noise"]["parameterization"] == "fixed_gaussian":
            identity.pop("noise.generator", None)
        signature = str(identity)
        if signature in seen:
            continue
        seen.add(signature)
        print(overrides, flush=True)
        if not args.dry_run:
            run_experiment(config)


if __name__ == "__main__":
    main()
