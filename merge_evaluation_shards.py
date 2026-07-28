from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from arl_robot.io_utils import runtime_metadata, write_json


NUMERIC_METRICS = (
    "success",
    "episode_reward",
    "episode_length",
    "final_distance",
    "collision_rate",
    "action_smoothness",
    "mean_inference_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_dirs = sorted(
        path.parent
        for path in args.input_root.glob("shard_*/evaluation_summary.json")
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No completed shards under {args.input_root}")

    configs = [
        json.loads((path / "evaluation_config.json").read_text(encoding="utf-8"))
        for path in shard_dirs
    ]
    identity = ("algorithm", "task", "model_seed", "checkpoint")
    for key in identity:
        values = {str(config[key]) for config in configs}
        if len(values) != 1:
            raise ValueError(f"Shard mismatch for {key}: {sorted(values)}")

    frames = [pd.read_csv(path / "episode_metrics.csv") for path in shard_dirs]
    episodes = pd.concat(frames, ignore_index=True).sort_values("episode")
    if len(episodes) != args.expected_episodes:
        raise ValueError(
            f"Expected {args.expected_episodes} rows, found {len(episodes)}"
        )
    expected_ids = np.arange(args.expected_episodes, dtype=np.int64)
    actual_ids = episodes["episode"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError("Episode IDs are missing or duplicated across shards")

    args.output.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.output / "episode_metrics.csv", index=False)

    trajectory_parts: dict[str, list[np.ndarray]] = {}
    for shard_dir in shard_dirs:
        with np.load(shard_dir / "trajectories.npz", allow_pickle=False) as archive:
            for key in archive.files:
                trajectory_parts.setdefault(key, []).append(archive[key])
    np.savez_compressed(
        args.output / "trajectories.npz",
        **{
            key: np.concatenate(parts, axis=0)
            for key, parts in trajectory_parts.items()
        },
    )

    first = configs[0]
    summary = {
        "algorithm": first["algorithm"],
        "task": first["task"],
        "seed": int(first["model_seed"]),
        "evaluation_seed": int(first["seed"]),
        "episodes": args.expected_episodes,
        "checkpoint": str(Path(first["checkpoint"]).resolve()),
        "elapsed_seconds": float(
            sum(
                json.loads(
                    (path / "evaluation_summary.json").read_text(encoding="utf-8")
                )["elapsed_seconds"]
                for path in shard_dirs
            )
        ),
        "runtime": runtime_metadata(),
        "merged_shards": [str(path.resolve()) for path in shard_dirs],
    }
    for key in NUMERIC_METRICS:
        values = pd.to_numeric(episodes[key], errors="raise").to_numpy(float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        )
    write_json(args.output / "evaluation_summary.json", summary)
    write_json(
        args.output / "evaluation_config.json",
        {
            **first,
            "episodes": args.expected_episodes,
            "episode_offset": 0,
            "output": str(args.output.resolve()),
            "merged_shards": [str(path.resolve()) for path in shard_dirs],
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "episodes": len(episodes),
                "shards": len(shard_dirs),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
