from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from arl_robot.constants import SUPPORTED_TASKS
from arl_robot.diffusion import DiffusionPolicy, DiffusionPolicyRunner
from arl_robot.io_utils import (
    append_csv,
    runtime_metadata,
    set_global_seed,
    write_json,
)
from arl_robot.reinforce import GaussianActor
from arl_robot.rlbench_env import RLBenchStateEnv, RewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=("diffusion", "ppo", "sac", "reinforce"),
    )
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--execute-horizon", type=int, default=4)
    parser.add_argument("--observation-clip", type=float, default=None)
    parser.add_argument("--force-gripper-open", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


class SB3Runner:
    def __init__(self, model) -> None:
        self.model = model

    def reset(self) -> None:
        return None

    def predict(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)


class ReinforceRunner:
    def __init__(self, actor: GaussianActor) -> None:
        self.actor = actor.eval().cpu()

    def reset(self) -> None:
        return None

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self.actor.act(observation, deterministic=True)


def load_runner(args: argparse.Namespace):
    if args.algorithm == "diffusion":
        device = torch.device(
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else ("cpu" if args.device == "auto" else args.device)
        )
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if checkpoint.get("task") != args.task:
            raise ValueError(
                f"Checkpoint task {checkpoint.get('task')} != {args.task}"
            )
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
    if args.algorithm == "reinforce":
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        actor = GaussianActor(**checkpoint["actor_config"])
        actor.load_state_dict(checkpoint["actor_state_dict"])
        return ReinforceRunner(actor)
    if args.algorithm == "ppo":
        from stable_baselines3 import PPO

        return SB3Runner(PPO.load(args.checkpoint, device=args.device))
    from stable_baselines3 import SAC

    return SB3Runner(SAC.load(args.checkpoint, device=args.device))


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    runner = load_runner(args)
    env = RLBenchStateEnv(
        task_name=args.task,
        robot="panda",
        max_episode_steps=args.max_episode_steps,
        reward_config=RewardConfig(),
    )
    trajectory = {
        "episode_ids": [],
        "timesteps": [],
        "observations": [],
        "actions": [],
        "rewards": [],
        "final_distances": [],
        "collisions": [],
        "smoothness": [],
    }
    episode_rows = []
    started = time.time()
    try:
        for local_episode in range(args.episodes):
            episode = args.episode_offset + local_episode
            observation, _ = env.reset(seed=args.seed + episode)
            runner.reset()
            done = False
            timestep = 0
            total_inference_seconds = 0.0
            final_info = {}
            while not done:
                inference_started = time.perf_counter()
                action = runner.predict(observation)
                total_inference_seconds += time.perf_counter() - inference_started
                next_observation, reward, terminated, truncated, info = env.step(
                    action
                )
                trajectory["episode_ids"].append(episode)
                trajectory["timesteps"].append(timestep)
                trajectory["observations"].append(observation.copy())
                trajectory["actions"].append(action.copy())
                trajectory["rewards"].append(reward)
                trajectory["final_distances"].append(info["final_distance"])
                trajectory["collisions"].append(info["collision"])
                trajectory["smoothness"].append(info["smoothness"])
                observation = next_observation
                timestep += 1
                done = terminated or truncated
                final_info = info
            metrics = final_info["episode_metrics"]
            row = {
                "algorithm": args.algorithm,
                "task": args.task,
                "seed": args.model_seed,
                "evaluation_seed": args.seed,
                "episode": episode,
                **metrics,
                "inference_seconds": total_inference_seconds,
                "mean_inference_ms": 1000.0
                * total_inference_seconds
                / max(timestep, 1),
            }
            append_csv(args.output / "episode_metrics.csv", row)
            episode_rows.append(row)
            print(
                f"episode={episode:03d} success={int(metrics['success'])} "
                f"reward={metrics['episode_reward']:.4f} "
                f"steps={metrics['episode_length']}",
                flush=True,
            )
    finally:
        env.close()

    np.savez_compressed(
        args.output / "trajectories.npz",
        episode_ids=np.asarray(trajectory["episode_ids"], dtype=np.int32),
        timesteps=np.asarray(trajectory["timesteps"], dtype=np.int32),
        observations=np.asarray(trajectory["observations"], dtype=np.float32),
        actions=np.asarray(trajectory["actions"], dtype=np.float32),
        rewards=np.asarray(trajectory["rewards"], dtype=np.float32),
        final_distances=np.asarray(
            trajectory["final_distances"], dtype=np.float32
        ),
        collisions=np.asarray(trajectory["collisions"], dtype=np.bool_),
        smoothness=np.asarray(trajectory["smoothness"], dtype=np.float32),
    )
    numeric_metrics = (
        "success",
        "episode_reward",
        "episode_length",
        "final_distance",
        "collision_rate",
        "action_smoothness",
        "mean_inference_ms",
    )
    summary = {
        "algorithm": args.algorithm,
        "task": args.task,
        "seed": args.model_seed,
        "evaluation_seed": args.seed,
        "episodes": args.episodes,
        "checkpoint": str(args.checkpoint.resolve()),
        "elapsed_seconds": time.time() - started,
        "runtime": runtime_metadata(),
    }
    for key in numeric_metrics:
        values = np.asarray([float(row[key]) for row in episode_rows])
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    write_json(args.output / "evaluation_summary.json", summary)
    write_json(
        args.output / "evaluation_config.json",
        {
            **vars(args),
            "checkpoint": str(args.checkpoint.resolve()),
            "output": str(args.output.resolve()),
            "reward_shaping": "disabled",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
