from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from arl_robot.constants import SUPPORTED_TASKS
from arl_robot.io_utils import (
    append_csv,
    runtime_metadata,
    set_global_seed,
    write_json,
)
from arl_robot.reinforce import (
    GaussianActor,
    ValueNetwork,
    discounted_returns,
)
from arl_robot.rlbench_env import RLBenchStateEnv, RewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", required=True, choices=("ppo", "sac", "reinforce")
    )
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--progress-scale", type=float, default=5.0)
    parser.add_argument("--task-progress-scale", type=float, default=10.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--step-penalty", type=float, default=0.001)
    parser.add_argument("--collision-penalty", type=float, default=0.0)
    parser.add_argument("--smoothness-penalty", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reinforce-batch-episodes", type=int, default=8)
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> RLBenchStateEnv:
    return RLBenchStateEnv(
        task_name=args.task,
        robot="panda",
        max_episode_steps=args.max_episode_steps,
        reward_config=RewardConfig(
            progress_scale=args.progress_scale,
            task_progress_scale=args.task_progress_scale,
            success_bonus=args.success_bonus,
            step_penalty=args.step_penalty,
            collision_penalty=args.collision_penalty,
            smoothness_penalty=args.smoothness_penalty,
        ),
    )


def save_episode(
    output: Path,
    metrics: dict,
    algorithm: str,
    task: str,
    seed: int,
    episode: int,
    environment_steps: int,
) -> None:
    append_csv(
        output / "episode_metrics.csv",
        {
            "algorithm": algorithm,
            "task": task,
            "seed": seed,
            "episode": episode,
            "environment_steps": environment_steps,
            **metrics,
        },
    )


def train_sb3(args: argparse.Namespace) -> dict:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback

    class SuccessReplayBuffer(ReplayBuffer):
        """Mix successful transitions into every SAC batch when available."""

        def __init__(self, *buffer_args, success_threshold: float, **buffer_kwargs):
            super().__init__(*buffer_args, **buffer_kwargs)
            self.success_threshold = float(success_threshold)

        def sample(self, batch_size: int, env=None):
            upper = self.buffer_size if self.full else self.pos
            if upper <= 0:
                return super().sample(batch_size, env)
            rewards = self.rewards[:upper].max(axis=1)
            successful = np.flatnonzero(rewards >= self.success_threshold)
            preferred = min(len(successful), max(1, batch_size // 4))
            regular = batch_size - preferred
            indices = np.random.randint(0, upper, size=regular)
            if preferred:
                indices = np.concatenate(
                    [
                        indices,
                        np.random.choice(successful, preferred, replace=True),
                    ]
                )
                np.random.shuffle(indices)
            return self._get_samples(indices, env=env)

    env = make_env(args)
    started = time.time()

    class MetricsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.episode = 0
            self.recent_rewards: deque[float] = deque(maxlen=20)
            self.last_log_step = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                metrics = info.get("episode_metrics")
                if metrics is None:
                    continue
                save_episode(
                    args.output,
                    metrics,
                    args.algorithm,
                    args.task,
                    args.seed,
                    self.episode,
                    self.num_timesteps,
                )
                self.recent_rewards.append(float(metrics["episode_reward"]))
                self.episode += 1
            if self.num_timesteps - self.last_log_step >= 1000:
                append_csv(
                    args.output / "training_metrics.csv",
                    {
                        "algorithm": args.algorithm,
                        "task": args.task,
                        "seed": args.seed,
                        "environment_steps": self.num_timesteps,
                        "episodes": self.episode,
                        "rolling_mean_reward": (
                            float(np.mean(self.recent_rewards))
                            if self.recent_rewards
                            else float("nan")
                        ),
                        "elapsed_seconds": time.time() - started,
                    },
                )
                self.last_log_step = self.num_timesteps
            if self.num_timesteps > 0 and self.num_timesteps % 10_000 == 0:
                self.model.save(
                    args.output / f"checkpoint_step_{self.num_timesteps}"
                )
            return True

    policy_kwargs = {"net_arch": [args.hidden_dim, args.hidden_dim]}
    if args.algorithm == "ppo":
        rollout_steps = min(1024, max(8, args.total_steps))
        batch_size = min(256, rollout_steps)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            n_steps=rollout_steps,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            buffer_size=min(args.total_steps, 200_000),
            learning_starts=min(1000, args.total_steps // 10),
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            replay_buffer_class=SuccessReplayBuffer,
            replay_buffer_kwargs={
                "success_threshold": args.success_bonus * 0.5
            },
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    callback = MetricsCallback()
    try:
        model.learn(total_timesteps=args.total_steps, callback=callback)
        model.save(args.output / "model")
    finally:
        env.close()
    return {
        "episodes": callback.episode,
        "environment_steps": int(model.num_timesteps),
        "checkpoint": str((args.output / "model.zip").resolve()),
        "elapsed_seconds": time.time() - started,
    }


def train_reinforce(args: argparse.Namespace) -> dict:
    env = make_env(args)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    actor = GaussianActor(
        env.state_dim, env.action_dim, hidden_dim=args.hidden_dim
    ).to(device)
    value = ValueNetwork(env.state_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(value.parameters()),
        lr=args.learning_rate,
    )
    started = time.time()
    environment_steps = 0
    episode = 0
    recent_rewards: deque[float] = deque(maxlen=20)
    episodes_in_batch = 0
    optimizer.zero_grad(set_to_none=True)
    try:
        while environment_steps < args.total_steps:
            observation, _ = env.reset(seed=args.seed + episode)
            observations = []
            log_probabilities = []
            entropies = []
            rewards = []
            done = False
            final_info = {}
            while not done and environment_steps < args.total_steps:
                tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action, log_probability, entropy = actor.sample(tensor)
                next_observation, reward, terminated, truncated, info = env.step(
                    action[0].detach().cpu().numpy()
                )
                observations.append(tensor[0])
                log_probabilities.append(log_probability[0])
                entropies.append(entropy[0])
                rewards.append(reward)
                observation = next_observation
                environment_steps += 1
                done = terminated or truncated
                final_info = info

            returns = discounted_returns(rewards, args.gamma, device)
            observation_tensor = torch.stack(observations)
            values = value(observation_tensor)
            advantages = returns - values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            policy_loss = -(
                torch.stack(log_probabilities) * advantages
            ).mean()
            value_loss = 0.5 * (values - returns).square().mean()
            entropy = torch.stack(entropies).mean()
            loss = policy_loss + value_loss - 1e-3 * entropy
            (loss / args.reinforce_batch_episodes).backward()
            episodes_in_batch += 1
            if (
                episodes_in_batch >= args.reinforce_batch_episodes
                or environment_steps >= args.total_steps
            ):
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(value.parameters()), 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                episodes_in_batch = 0

            episode_metrics = final_info.get(
                "episode_metrics",
                {
                    "success": False,
                    "episode_reward": float(sum(rewards)),
                    "episode_length": len(rewards),
                    "final_distance": final_info.get(
                        "final_distance", float("nan")
                    ),
                    "control_distance": final_info.get(
                        "control_distance", float("nan")
                    ),
                    "collision_count": float("nan"),
                    "collision_rate": float("nan"),
                    "action_smoothness": float("nan"),
                    "terminated": False,
                    "truncated": True,
                },
            )
            save_episode(
                args.output,
                episode_metrics,
                args.algorithm,
                args.task,
                args.seed,
                episode,
                environment_steps,
            )
            recent_rewards.append(float(sum(rewards)))
            append_csv(
                args.output / "training_metrics.csv",
                {
                    "algorithm": args.algorithm,
                    "task": args.task,
                    "seed": args.seed,
                    "environment_steps": environment_steps,
                    "episodes": episode + 1,
                    "rolling_mean_reward": float(np.mean(recent_rewards)),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "elapsed_seconds": time.time() - started,
                },
            )
            if episode % 10 == 0:
                print(
                    f"episode={episode} steps={environment_steps} "
                    f"reward={sum(rewards):.4f} success="
                    f"{int(bool(episode_metrics['success']))}",
                    flush=True,
                )
                torch.save(
                    {
                        "actor_state_dict": actor.state_dict(),
                        "actor_config": actor.config(),
                        "value_state_dict": value.state_dict(),
                        "task": args.task,
                        "seed": args.seed,
                        "episode": episode,
                        "environment_steps": environment_steps,
                    },
                    args.output / "checkpoint_last.pt",
                )
            episode += 1
        checkpoint = {
            "actor_state_dict": actor.state_dict(),
            "actor_config": actor.config(),
            "value_state_dict": value.state_dict(),
            "task": args.task,
            "seed": args.seed,
        }
        torch.save(checkpoint, args.output / "model.pt")
    finally:
        env.close()
    return {
        "episodes": episode,
        "environment_steps": environment_steps,
        "checkpoint": str((args.output / "model.pt").resolve()),
        "elapsed_seconds": time.time() - started,
    }


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "run_config.json",
        {
            **vars(args),
            "output": str(args.output.resolve()),
            "robot": "panda",
            "action": "normalized 7 absolute joint targets + gripper command",
            "runtime": runtime_metadata(),
        },
    )
    summary = (
        train_reinforce(args)
        if args.algorithm == "reinforce"
        else train_sb3(args)
    )
    write_json(
        args.output / "training_summary.json",
        {
            "algorithm": args.algorithm,
            "task": args.task,
            "seed": args.seed,
            **summary,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
