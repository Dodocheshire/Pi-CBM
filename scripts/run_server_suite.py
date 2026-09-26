"""One GPU worker for a complete injection ablation on one dataset.

Each dataset uses a fixed clean CBM and split. Seeds vary adapter initialization,
training shuffle and noise draws. Completed runs can be resumed after SSH loss.
"""

import argparse
import json
import os
from pathlib import Path

from pi_cbm.config import load_config


def experiments(config_path: Path, seeds: list[int]):
    yield (
        "baseline",
        load_config(config_path, ["injection.site=baseline", "experiment.name=baseline"]),
    )
    for site in ("target_global", "target_internal", "target_image"):
        for seed in seeds:
            name = f"{site}-mean_scale-seed{seed}"
            storage = (
                ["injection.point=input", "data.cache_images=true", "data.cache_internal=false"]
                if site == "target_image"
                else []
            )
            config = load_config(
                config_path,
                [
                    f"injection.site={site}",
                    f"experiment.seed={seed}",
                    f"experiment.name={name}",
                    *storage,
                ],
            )
            yield name, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    run_suite(experiments(args.config, args.seeds), args.gpu, args.config.stem)


def run_suite(configurations, gpu: str, name: str):
    """Execute an ordered config list and resume only identical code/config runs."""
    # CLIP checks CUDA availability during import. Select the GPU before any
    # module imports CLIP/PyTorch, otherwise both workers may initialize GPU 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    from pi_cbm.data import digest
    from pi_cbm.pipeline import run_experiment, source_hash

    progress_path = Path("artifacts") / f"suite-{name}.json"
    progress_path.parent.mkdir(exist_ok=True)
    progress = (
        json.loads(progress_path.read_text()) if progress_path.exists() else {"completed": {}}
    )
    implementation = source_hash()

    def save():
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(progress_path)

    for name, config in configurations:
        signature = digest({"config": config, "source": implementation})
        if signature in progress["completed"]:
            continue
        progress.update(status="running", current=name, gpu=gpu)
        save()
        print(f"Starting {name}", flush=True)
        try:
            run_dir = run_experiment(config)
        except Exception as error:
            progress.update(status="failed", error=str(error))
            save()
            raise
        progress["completed"][signature] = {"name": name, "run": str(run_dir)}
        save()
    progress.update(status="complete", current=None)
    save()


if __name__ == "__main__":
    main()
