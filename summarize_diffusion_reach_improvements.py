from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    return parser.parse_args()


def read_summary(path: Path, variant: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return {
        "variant": variant,
        "seed": int(value["seed"]),
        "episodes": int(value["episodes"]),
        "success_rate": float(value["success_mean"]),
        "episode_reward": float(value["episode_reward_mean"]),
        "final_distance": float(value["final_distance_mean"]),
        "action_smoothness": float(value["action_smoothness_mean"]),
        "collision_rate": float(value["collision_rate_mean"]),
        "inference_ms": float(value["mean_inference_ms_mean"]),
        "source": str(path.resolve()),
    }


def main() -> int:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(3):
        rows.append(
            read_summary(
                args.baseline_root
                / f"diffusion_reach_target_seed{seed}"
                / "evaluation"
                / "evaluation_summary.json",
                "baseline",
            )
        )
    for variant in ("old_checkpoint_inference_fixed", "retrained_cosine"):
        for seed in range(3):
            rows.append(
                read_summary(
                    args.root
                    / f"{variant}_seed{seed}"
                    / "evaluation"
                    / "evaluation_summary.json",
                    variant,
                )
            )

    columns = list(rows[0])
    with (args.root / "per_seed_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    metrics = (
        "success_rate",
        "episode_reward",
        "final_distance",
        "action_smoothness",
        "collision_rate",
        "inference_ms",
    )
    for variant in ("baseline", "old_checkpoint_inference_fixed", "retrained_cosine"):
        selected = [row for row in rows if row["variant"] == variant]
        summary = {"variant": variant, "seeds": [row["seed"] for row in selected]}
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
        aggregate.append(summary)
    with (args.root / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")

    labels = [item["variant"] for item in aggregate]
    means = [100.0 * item["success_rate_mean"] for item in aggregate]
    errors = [100.0 * item["success_rate_std"] for item in aggregate]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, means, yerr=errors, capsize=5)
    axis.set_ylabel("Success rate (%)")
    axis.set_title("Diffusion Policy: Reach Target ablations")
    axis.set_ylim(0, 100)
    axis.tick_params(axis="x", rotation=12)
    figure.tight_layout()
    figure.savefig(args.root / "success_rate_comparison.png", dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
