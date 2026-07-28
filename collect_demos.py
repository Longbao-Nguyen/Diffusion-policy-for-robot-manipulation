from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

from arl_robot.constants import SUPPORTED_TASKS
from arl_robot.io_utils import runtime_metadata, set_global_seed, write_json
from arl_robot.rlbench_env import RLBenchStateEnv
from arl_robot.task_state import extract_task_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-positions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    dones: list[bool] = []
    episode_ids: list[int] = []
    timesteps: list[int] = []
    rewards: list[float] = []
    approach_progresses: list[float] = []
    task_progresses: list[float] = []
    success_bonuses: list[float] = []
    episode_summaries = []
    failures = []
    started = time.time()

    env = RLBenchStateEnv(
        task_name=args.task,
        robot="panda",
        static_positions=args.static_positions,
        max_episode_steps=500,
    )
    try:
        for episode in range(args.episodes):
            global_episode = args.episode_offset + episode
            success = False
            for attempt in range(1, args.max_attempts + 1):
                episode_start = len(observations)
                try:
                    variation = int(
                        np.random.randint(env._task.variation_count())
                    )
                    env._task.set_variation(variation)
                    _, initial_observation = env._task.reset()
                    previous_state = env.state_from_observation(
                        initial_observation
                    )
                    previous_metrics = extract_task_metrics(args.task, env._task)
                    episode_transition_count = 0

                    def record_step(observation) -> None:
                        nonlocal previous_state, previous_metrics
                        nonlocal episode_transition_count
                        raw_action = observation.misc.get("joint_position_action")
                        if raw_action is None:
                            return
                        next_state = env.state_from_observation(observation)
                        current_metrics = extract_task_metrics(args.task, env._task)
                        approach_progress = (
                            previous_metrics.approach_distance
                            - current_metrics.approach_distance
                        )
                        task_progress = (
                            previous_metrics.task_distance
                            - current_metrics.task_distance
                        )
                        reward = (
                            env.reward_config.progress_scale * approach_progress
                            + env.reward_config.task_progress_scale * task_progress
                            - env.reward_config.step_penalty
                        )
                        observations.append(previous_state.copy())
                        actions.append(env.normalize_expert_action(raw_action))
                        next_observations.append(next_state.copy())
                        dones.append(False)
                        episode_ids.append(global_episode)
                        timesteps.append(episode_transition_count)
                        rewards.append(float(reward))
                        approach_progresses.append(float(approach_progress))
                        task_progresses.append(float(task_progress))
                        success_bonuses.append(0.0)
                        previous_state = next_state
                        previous_metrics = current_metrics
                        episode_transition_count += 1

                    env._task._scene.get_demo(
                        # RLBench initializes its expert gripper/action record
                        # only when record=True. The returned Demo is discarded;
                        # the callback below stores our normalized transitions.
                        record=True,
                        callable_each_step=record_step,
                        randomly_place=not args.static_positions,
                    )
                    task_success, _ = env._task._task.success()
                    if not task_success or episode_transition_count == 0:
                        raise RuntimeError(
                            "Expert ended without success or transitions"
                        )
                    dones[-1] = True
                    rewards[-1] += env.reward_config.success_bonus
                    success_bonuses[-1] = env.reward_config.success_bonus
                    metrics = env.describe()
                    episode_summaries.append(
                        {
                            "episode": global_episode,
                            "attempt": attempt,
                            "transitions": episode_transition_count,
                            "variation": variation,
                            "success": True,
                            "action_dim": metrics["action_dim"],
                            "state_dim": metrics["state_dim"],
                        }
                    )
                    success = True
                    print(
                        f"episode={global_episode:03d} attempt={attempt} "
                        f"transitions={episode_transition_count} success=1",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    # Remove a partial failed attempt before retrying.
                    del observations[episode_start:]
                    del actions[episode_start:]
                    del next_observations[episode_start:]
                    del dones[episode_start:]
                    del episode_ids[episode_start:]
                    del timesteps[episode_start:]
                    del rewards[episode_start:]
                    del approach_progresses[episode_start:]
                    del task_progresses[episode_start:]
                    del success_bonuses[episode_start:]
                    failures.append(
                        {
                            "episode": global_episode,
                            "attempt": attempt,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(
                        f"episode={global_episode:03d} attempt={attempt} "
                        f"success=0 error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
            if not success:
                raise RuntimeError(
                    f"Could not collect episode {global_episode} after "
                    f"{args.max_attempts} attempts"
                )
    finally:
        env.close()

    arrays = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "episode_ids": np.asarray(episode_ids, dtype=np.int32),
        "timesteps": np.asarray(timesteps, dtype=np.int32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "approach_progress": np.asarray(approach_progresses, dtype=np.float32),
        "task_progress": np.asarray(task_progresses, dtype=np.float32),
        "success_bonus": np.asarray(success_bonuses, dtype=np.float32),
    }
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "task": args.task,
        "robot": "panda",
        "episodes_requested": args.episodes,
        "episodes_collected": len(episode_summaries),
        "transitions": len(observations),
        "state_dim": int(arrays["observations"].shape[1]),
        "action_dim": int(arrays["actions"].shape[1]),
        "action": "normalized 7 absolute Panda joint targets + binary gripper",
        "reward": env.reward_config.__dict__,
        "seed": args.seed,
        "episode_offset": args.episode_offset,
        "static_positions": args.static_positions,
        "elapsed_seconds": time.time() - started,
        "episodes": episode_summaries,
        "failed_attempts": failures,
        "runtime": runtime_metadata(),
    }
    write_json(args.output.with_suffix(".json"), metadata)
    print(
        f"Wrote {args.output}: episodes={len(episode_summaries)}, "
        f"transitions={len(observations)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
