from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from arl_robot.constants import SUPPORTED_TASKS
from arl_robot.diffusion import DiffusionPolicy, DiffusionPolicyRunner
from arl_robot.io_utils import runtime_metadata, set_global_seed, write_json
from arl_robot.rlbench_env import RLBenchStateEnv, RewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record the first successful Diffusion Policy rollout."
    )
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--execute-horizon", type=int, default=4)
    parser.add_argument("--observation-clip", type=float, default=None)
    parser.add_argument("--force-gripper-open", action="store_true")
    parser.add_argument("--camera", default="front")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_runner(args: argparse.Namespace) -> DiffusionPolicyRunner:
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("task") != args.task:
        raise ValueError(f"Checkpoint task {checkpoint.get('task')} != {args.task}")
    model = DiffusionPolicy(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return DiffusionPolicyRunner(
        model,
        checkpoint["observation_mean"],
        checkpoint["observation_std"],
        device=device,
        execute_horizon=args.execute_horizon,
        seed=args.seed + args.model_seed,
        action_mean=checkpoint.get("action_mean"),
        action_std=checkpoint.get("action_std"),
        observation_clip=(
            args.observation_clip
            if args.observation_clip is not None
            else checkpoint.get("observation_clip")
        ),
        force_gripper_open=args.force_gripper_open,
    )


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A short pause makes the initial and successful terminal states visible.
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
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    set_global_seed(args.seed)
    runner = load_runner(args)
    env = RLBenchStateEnv(
        task_name=args.task,
        robot="panda",
        max_episode_steps=args.max_episode_steps,
        reward_config=RewardConfig(),
        record_camera=args.camera,
        camera_image_size=(args.image_size, args.image_size),
    )
    started = time.time()
    successful_attempt: int | None = None
    successful_episode_seed: int | None = None
    successful_metrics: dict | None = None
    successful_frames: list[np.ndarray] | None = None
    try:
        for attempt in range(args.max_attempts):
            episode_seed = args.seed + attempt
            observation, _ = env.reset(seed=episode_seed)
            runner.reset()
            frames = [env.render_rgb()]
            done = False
            final_info = {}
            while not done:
                action = runner.predict(observation)
                observation, _, terminated, truncated, final_info = env.step(action)
                frames.append(env.render_rgb())
                done = terminated or truncated
            metrics = final_info["episode_metrics"]
            print(
                f"attempt={attempt:03d} seed={episode_seed} "
                f"success={int(metrics['success'])} "
                f"reward={metrics['episode_reward']:.4f} "
                f"steps={metrics['episode_length']}",
                flush=True,
            )
            if bool(metrics["success"]):
                successful_attempt = attempt
                successful_episode_seed = episode_seed
                successful_metrics = metrics
                successful_frames = frames
                break
    finally:
        env.close()

    if successful_frames is None:
        raise RuntimeError(
            f"No successful {args.task} rollout in {args.max_attempts} attempts"
        )

    write_video(args.output, successful_frames, args.fps)
    metadata_path = args.output.with_suffix(".json")
    write_json(
        metadata_path,
        {
            "algorithm": "diffusion",
            "task": args.task,
            "checkpoint": str(args.checkpoint.resolve()),
            "model_seed": args.model_seed,
            "base_evaluation_seed": args.seed,
            "successful_attempt": successful_attempt,
            "successful_episode_seed": successful_episode_seed,
            "frames": len(successful_frames),
            "fps": args.fps,
            "camera": args.camera,
            "image_size": args.image_size,
            "execute_horizon": args.execute_horizon,
            "observation_clip": args.observation_clip,
            "force_gripper_open": args.force_gripper_open,
            "episode_metrics": successful_metrics,
            "elapsed_seconds": time.time() - started,
            "runtime": runtime_metadata(),
        },
    )
    print(f"video={args.output}", flush=True)
    print(f"metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
