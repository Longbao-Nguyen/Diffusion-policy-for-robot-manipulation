from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from arl_robot.constants import SUPPORTED_TASKS
from arl_robot.io_utils import runtime_metadata, set_global_seed, write_json
from arl_robot.rlbench_env import RLBenchStateEnv, RewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-render a successful Diffusion Policy evaluation trajectory and "
            "verify native RLBench success before saving it."
        )
    )
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    return parser.parse_args()


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def successful_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if is_true(row["success"])]


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display_frames = [frames[0]] * fps + frames + [frames[-1]] * (2 * fps)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    ) as writer:
        for frame in display_frames:
            writer.append_data(frame)


def main() -> int:
    args = parse_args()
    metrics_path = args.evaluation_dir / "episode_metrics.csv"
    trajectories_path = args.evaluation_dir / "trajectories.npz"
    candidates = successful_rows(metrics_path)[: args.max_candidates]
    if not candidates:
        raise RuntimeError(f"No successful episodes in {metrics_path}")

    trajectory = np.load(trajectories_path)
    episode_ids = trajectory["episode_ids"]
    timesteps = trajectory["timesteps"]
    actions = trajectory["actions"]
    set_global_seed(int(candidates[0]["evaluation_seed"]))
    env = RLBenchStateEnv(
        task_name=args.task,
        robot="panda",
        max_episode_steps=args.max_episode_steps,
        reward_config=RewardConfig(),
        record_camera=args.camera,
        camera_image_size=(args.image_size, args.image_size),
    )
    started = time.time()
    selected: dict[str, str] | None = None
    selected_frames: list[np.ndarray] | None = None
    replay_metrics: dict | None = None
    try:
        for candidate_index, row in enumerate(candidates):
            episode = int(row["episode"])
            evaluation_seed = int(row["evaluation_seed"])
            indices = np.flatnonzero(episode_ids == episode)
            indices = indices[np.argsort(timesteps[indices])]
            if len(indices) == 0:
                print(f"candidate={candidate_index} episode={episode} missing", flush=True)
                continue
            _, _ = env.reset(seed=evaluation_seed + episode)
            frames = [env.render_rgb()]
            final_info = {}
            terminated = False
            truncated = False
            for index in indices:
                _, _, terminated, truncated, final_info = env.step(actions[index])
                frames.append(env.render_rgb())
                if terminated or truncated:
                    break
            success = bool(final_info.get("episode_metrics", {}).get("success", False))
            print(
                f"candidate={candidate_index:03d} episode={episode:03d} "
                f"recorded_steps={len(indices)} replay_success={int(success)}",
                flush=True,
            )
            if success:
                selected = row
                selected_frames = frames
                replay_metrics = final_info["episode_metrics"]
                break
    finally:
        env.close()

    if selected_frames is None or selected is None:
        raise RuntimeError(
            f"None of {len(candidates)} successful stored trajectories "
            "reproduced native RLBench success with RGB recording enabled"
        )

    write_video(args.output, selected_frames, args.fps)
    metadata_path = args.output.with_suffix(".json")
    write_json(
        metadata_path,
        {
            "algorithm": "diffusion",
            "task": args.task,
            "recording_mode": "replay_of_successful_evaluation_trajectory",
            "evaluation_dir": str(args.evaluation_dir.resolve()),
            "model_seed": args.model_seed,
            "episode": int(selected["episode"]),
            "evaluation_seed": int(selected["evaluation_seed"]),
            "original_episode_metrics": selected,
            "replay_episode_metrics": replay_metrics,
            "native_rlbench_success_reverified": True,
            "frames": len(selected_frames),
            "fps": args.fps,
            "camera": args.camera,
            "image_size": args.image_size,
            "elapsed_seconds": time.time() - started,
            "runtime": runtime_metadata(),
        },
    )
    print(f"video={args.output}", flush=True)
    print(f"metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
