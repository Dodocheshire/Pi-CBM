"""Summarize completed GPU suites without selecting models on test accuracy."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import yaml


def summarize(progress_files: list[Path], output: Path):
    rows, groups = [], defaultdict(list)
    for path in progress_files:
        progress = json.loads(path.read_text())
        if progress["status"] != "complete":
            raise ValueError(f"Suite is not complete: {path}")
        for entry in progress["completed"].values():
            run = Path(entry["run"])
            metrics = json.loads((run / "metrics.json").read_text())
            config = yaml.safe_load((run / "config.yaml").read_text())
            row = {
                "dataset": config["data"]["dataset"],
                "site": config["injection"]["site"],
                "seed": config["experiment"]["seed"],
                "run": str(run),
                "test": metrics["test"],
                "manifest": metrics["manifest"],
            }
            rows.append(row)
            groups[(row["dataset"], row["site"])].append(row)
    aggregates = []
    lines = [
        "| 数据集 | 注入位置 | 运行数 | 准确率均值 ± 标准差 | NLL | 概念漂移 MSE | 分类头非零数 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (dataset, site), members in sorted(groups.items()):
        accuracies = [row["test"]["accuracy"] * 100 for row in members]
        entry = {
            "dataset": dataset,
            "site": site,
            "runs": len(members),
            "accuracy_mean_percent": mean(accuracies),
            "accuracy_std_percent": stdev(accuracies) if len(accuracies) > 1 else None,
            "nll": mean(row["test"]["nll"] for row in members),
            "concept_drift_mse": mean(row["test"]["concept_drift_mse"] for row in members),
            "head_nonzero": members[0]["test"]["head_nonzero"],
        }
        aggregates.append(entry)
        deviation = f" ± {entry['accuracy_std_percent']:.2f}" if len(members) > 1 else ""
        lines.append(
            f"| {dataset} | {site} | {len(members)} | {mean(accuracies):.2f}{deviation}% | "
            f"{entry['nll']:.4f} | {entry['concept_drift_mse']:.5f} | {entry['head_nonzero']} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps({"aggregate": aggregates, "runs": rows}, indent=2)
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("progress", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("reports/server_injection_positions"))
    args = parser.parse_args()
    summarize(args.progress, args.output)


if __name__ == "__main__":
    main()
