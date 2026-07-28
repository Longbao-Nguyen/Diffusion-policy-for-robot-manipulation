from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arl_robot.demo_data import load_demo_archive
from arl_robot.io_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate(path: Path, expected_episodes: int) -> dict:
    data = load_demo_archive(path)
    observations = data["observations"]
    actions = data["actions"]
    episode_ids = data["episode_ids"]
    timesteps = data["timesteps"]
    dones = data["dones"]
    rewards = data.get("rewards")
    success_bonus = data.get("success_bonus")
    errors = []

    if observations.ndim != 2 or observations.shape[1] != 29:
        errors.append(f"observation shape is {observations.shape}, expected N x 29")
    if actions.ndim != 2 or actions.shape[1] != 8:
        errors.append(f"action shape is {actions.shape}, expected N x 8")
    if not np.isfinite(observations).all():
        errors.append("observations contain NaN or infinity")
    if not np.isfinite(actions).all():
        errors.append("actions contain NaN or infinity")
    if np.any(actions < -1.00001) or np.any(actions > 1.00001):
        errors.append("normalized actions exceed [-1, 1]")

    unique_episodes = np.unique(episode_ids)
    if len(unique_episodes) != expected_episodes:
        errors.append(
            f"found {len(unique_episodes)} episodes, expected {expected_episodes}"
        )
    for episode_id in unique_episodes:
        mask = episode_ids == episode_id
        episode_timesteps = timesteps[mask]
        expected_timesteps = np.arange(len(episode_timesteps))
        if not np.array_equal(episode_timesteps, expected_timesteps):
            errors.append(f"episode {episode_id} timesteps are not contiguous")
        episode_dones = dones[mask]
        if int(episode_dones.sum()) != 1 or not bool(episode_dones[-1]):
            errors.append(f"episode {episode_id} has invalid done markers")
        if success_bonus is not None:
            episode_bonus = success_bonus[mask]
            if int(np.sum(episode_bonus > 0.0)) != 1 or not bool(
                episode_bonus[-1] > 0.0
            ):
                errors.append(
                    f"episode {episode_id} has invalid success bonus markers"
                )

    return {
        "path": str(path.resolve()),
        "status": "PASS" if not errors else "FAIL",
        "episodes": len(unique_episodes),
        "transitions": len(actions),
        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "reward_mean": (
            float(np.mean(rewards)) if rewards is not None else None
        ),
        "success_bonus_count": (
            int(np.sum(success_bonus > 0.0))
            if success_bonus is not None
            else None
        ),
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    reports = [
        validate(path, args.expected_episodes) for path in args.datasets
    ]
    if args.output is not None:
        write_json(args.output, {"datasets": reports})
    print(json.dumps(reports, indent=2))
    return 0 if all(report["status"] == "PASS" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
