from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arl_robot.demo_data import REQUIRED_KEYS, load_demo_archive
from arl_robot.io_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.input.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No episode shards found in {args.input}")
    arrays: dict[str, list[np.ndarray]] | None = None
    shard_metadata = []
    episode_ids_seen = set()
    for path in paths:
        archive = load_demo_archive(path)
        if arrays is None:
            arrays = {key: [] for key in archive}
        if set(archive) != set(arrays):
            raise ValueError(f"{path} has inconsistent archive keys")
        unique_ids = np.unique(archive["episode_ids"])
        if len(unique_ids) != 1:
            raise ValueError(f"{path} contains {len(unique_ids)} episode IDs")
        episode_id = int(unique_ids[0])
        if episode_id in episode_ids_seen:
            raise ValueError(f"Duplicate episode ID {episode_id}")
        episode_ids_seen.add(episode_id)
        for key in arrays:
            arrays[key].append(archive[key])
        metadata_path = path.with_suffix(".json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        shard_metadata.append(
            {
                "episode_id": episode_id,
                "path": str(path.resolve()),
                "transitions": len(archive["actions"]),
                "metadata": metadata,
            }
        )
    if (
        len(episode_ids_seen) != args.expected_episodes
        and not args.allow_incomplete
    ):
        raise RuntimeError(
            f"Expected {args.expected_episodes} episode shards, "
            f"found {len(episode_ids_seen)}"
        )
    if arrays is None:
        raise RuntimeError("No arrays were loaded")
    merged = {
        key: np.concatenate(parts, axis=0) for key, parts in arrays.items()
    }
    order = np.lexsort((merged["timesteps"], merged["episode_ids"]))
    merged = {key: value[order] for key, value in merged.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)
    write_json(
        args.output.with_suffix(".json"),
        {
            "episodes": len(episode_ids_seen),
            "episode_ids": sorted(episode_ids_seen),
            "transitions": len(merged["actions"]),
            "state_dim": int(merged["observations"].shape[1]),
            "action_dim": int(merged["actions"].shape[1]),
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "shards": shard_metadata,
        },
    )
    print(
        f"Merged episodes={len(episode_ids_seen)} "
        f"transitions={len(merged['actions'])} into {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
