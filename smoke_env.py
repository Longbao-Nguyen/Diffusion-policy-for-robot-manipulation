from __future__ import annotations

import argparse
import json
import traceback

import numpy as np

from arl_robot.constants import SUPPORTED_TASKS
from arl_robot.io_utils import json_safe
from arl_robot.rlbench_env import RLBenchStateEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--expert-demo", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = []
    for task_name in args.tasks:
        env = None
        try:
            print(f"stage=launch task={task_name}", flush=True)
            env = RLBenchStateEnv(
                task_name=task_name,
                robot="panda",
                max_episode_steps=max(args.steps, 2),
            )
            print(f"stage=reset task={task_name}", flush=True)
            state, reset_info = env.reset(seed=args.seed)
            steps = []
            for _ in range(args.steps):
                # Holding the current pose is safer than a random full-range
                # absolute joint target during an environment smoke test.
                current = env._task.get_observation()
                absolute = np.concatenate(
                    [
                        np.asarray(current.joint_positions, dtype=np.float32),
                        [float(current.gripper_open)],
                    ]
                )
                action = env.normalize_expert_action(absolute)
                state, reward, terminated, truncated, info = env.step(action)
                steps.append(
                    {
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "final_distance": info["final_distance"],
                    }
                )
                if terminated or truncated:
                    break
            demo_length = None
            if args.expert_demo:
                print(f"stage=expert_demo task={task_name}", flush=True)
                demos = env._task.get_demos(
                    amount=1, live_demos=True, max_attempts=5
                )
                demo_length = len(demos[0])
            print(f"stage=done task={task_name}", flush=True)
            results.append(
                {
                    "task": task_name,
                    "status": "PASS",
                    "environment": env.describe(),
                    "reset": reset_info,
                    "state_shape": list(state.shape),
                    "finite_state": bool(np.isfinite(state).all()),
                    "steps": steps,
                    "expert_demo_length": demo_length,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "task": task_name,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            if env is not None:
                print(f"stage=shutdown task={task_name}", flush=True)
                env.close()
    print(json.dumps(json_safe(results), indent=2))
    return 0 if all(item["status"] == "PASS" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
