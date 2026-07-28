from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from .constants import SUPPORTED_TASKS
from .task_state import collision_detected, extract_task_metrics


@dataclass
class RewardConfig:
    progress_scale: float = 5.0
    task_progress_scale: float = 10.0
    success_bonus: float = 10.0
    step_penalty: float = 0.001
    collision_penalty: float = 0.0
    smoothness_penalty: float = 0.0


STATE_LAYOUT = {
    "gripper_open": [0, 1],
    "joint_positions": [1, 8],
    "joint_velocities": [8, 15],
    "gripper_pose": [15, 22],
    "task_features": [22, 29],
}


def make_observation_config(
    camera_name: str | None = None,
    camera_image_size: tuple[int, int] = (128, 128),
) -> ObservationConfig:
    config = ObservationConfig()
    config.set_all(False)
    config.gripper_open = True
    config.joint_positions = True
    config.joint_velocities = True
    config.gripper_pose = True
    if camera_name is not None:
        camera_attribute = f"{camera_name}_camera"
        if not hasattr(config, camera_attribute):
            raise ValueError(f"Unknown RLBench camera: {camera_name}")
        camera_config = getattr(config, camera_attribute)
        camera_config.rgb = True
        camera_config.image_size = tuple(int(value) for value in camera_image_size)
    return config


class RLBenchStateEnv(gym.Env):
    """Gymnasium wrapper for state-only RLBench manipulation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_name: str,
        robot: str = "panda",
        max_episode_steps: int = 200,
        static_positions: bool = False,
        variation: int | str = "random",
        reward_config: RewardConfig | dict[str, float] | None = None,
        record_camera: str | None = None,
        camera_image_size: tuple[int, int] = (128, 128),
        launch: bool = True,
    ) -> None:
        super().__init__()
        if task_name not in SUPPORTED_TASKS:
            raise ValueError(f"task_name must be one of {SUPPORTED_TASKS}")
        self.task_name = task_name
        self.robot = robot
        self.max_episode_steps = int(max_episode_steps)
        self.static_positions = bool(static_positions)
        self.variation = variation
        self.record_camera = record_camera
        self.camera_image_size = tuple(int(value) for value in camera_image_size)
        if reward_config is None:
            self.reward_config = RewardConfig()
        elif isinstance(reward_config, RewardConfig):
            self.reward_config = reward_config
        else:
            self.reward_config = RewardConfig(**reward_config)

        self._environment: Environment | None = None
        self._task = None
        self._joint_low: np.ndarray | None = None
        self._joint_high: np.ndarray | None = None
        self._step_count = 0
        self._episode_return = 0.0
        self._collision_count = 0
        self._smoothness_sum = 0.0
        self._previous_action: np.ndarray | None = None
        self._previous_control_distance: float | None = None
        self._previous_approach_distance: float | None = None
        self._previous_task_distance: float | None = None
        self._latest_rgb_frame: np.ndarray | None = None

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )
        # Policies operate in a normalized action space. The wrapper converts
        # the first 7 values to Panda absolute joint positions and thresholds
        # the last value into RLBench's binary gripper command.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        if launch:
            self.launch()

    def launch(self) -> None:
        if self._environment is not None:
            return
        action_mode = MoveArmThenGripper(
            JointPosition(absolute_mode=True),
            Discrete(),
        )
        self._environment = Environment(
            action_mode=action_mode,
            obs_config=make_observation_config(
                self.record_camera,
                self.camera_image_size,
            ),
            headless=True,
            robot_setup=self.robot,
            static_positions=self.static_positions,
            shaped_rewards=False,
        )
        self._environment.launch()
        self._task = self._environment.get_task(
            task_file_to_task_class(self.task_name)
        )
        cyclics, intervals = self._task._robot.arm.get_joint_intervals()
        lows, highs = [], []
        for cyclic, interval in zip(cyclics, intervals):
            if cyclic:
                lows.append(-np.pi)
                highs.append(np.pi)
            else:
                lower, joint_range = interval
                lows.append(lower)
                highs.append(lower + joint_range)
        self._joint_low = np.asarray(lows, dtype=np.float32)
        self._joint_high = np.asarray(highs, dtype=np.float32)
        if self._joint_low.shape != (7,):
            raise RuntimeError(
                f"Expected Panda 7-DOF arm, got {self._joint_low.shape[0]} joints"
            )

    @property
    def action_dim(self) -> int:
        return int(self.action_space.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.observation_space.shape[0])

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        if self._joint_low is None or self._joint_high is None:
            raise RuntimeError("Environment has not been launched")
        return self._joint_low.copy(), self._joint_high.copy()

    def state_from_observation(self, observation) -> np.ndarray:
        metrics = extract_task_metrics(self.task_name, self._task)
        state = np.concatenate(
            [
                [float(observation.gripper_open)],
                np.asarray(observation.joint_positions, dtype=np.float32),
                np.asarray(observation.joint_velocities, dtype=np.float32),
                np.asarray(observation.gripper_pose, dtype=np.float32),
                metrics.task_features,
            ]
        ).astype(np.float32)
        if state.shape != (29,) or not np.isfinite(state).all():
            raise RuntimeError(
                f"Invalid state for {self.task_name}: shape={state.shape}"
            )
        return state

    def _capture_rgb_frame(self, observation) -> None:
        if self.record_camera is None:
            self._latest_rgb_frame = None
            return
        frame = getattr(observation, f"{self.record_camera}_rgb", None)
        if frame is None:
            raise RuntimeError(
                f"RLBench did not return RGB for camera {self.record_camera}"
            )
        frame_array = np.asarray(frame)
        if np.issubdtype(frame_array.dtype, np.floating):
            frame_array = np.clip(frame_array * 255.0, 0.0, 255.0)
        self._latest_rgb_frame = frame_array.astype(np.uint8)

    def render_rgb(self) -> np.ndarray:
        if self._latest_rgb_frame is None:
            raise RuntimeError("RGB recording is not enabled or no frame is available")
        return self._latest_rgb_frame.copy()

    def normalize_expert_action(self, absolute_action: np.ndarray) -> np.ndarray:
        action = np.asarray(absolute_action, dtype=np.float32)
        if action.shape != (8,):
            raise ValueError(f"Expected expert action shape (8,), got {action.shape}")
        span = np.maximum(self._joint_high - self._joint_low, 1e-6)
        arm = 2.0 * (action[:7] - self._joint_low) / span - 1.0
        gripper = 1.0 if action[7] > 0.5 else -1.0
        return np.clip(np.concatenate([arm, [gripper]]), -1.0, 1.0).astype(
            np.float32
        )

    def denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        action = np.clip(
            np.asarray(normalized_action, dtype=np.float32), -1.0, 1.0
        )
        if action.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action.shape}")
        arm = self._joint_low + 0.5 * (action[:7] + 1.0) * (
            self._joint_high - self._joint_low
        )
        gripper = 1.0 if action[7] >= 0.0 else 0.0
        return np.concatenate([arm, [gripper]]).astype(np.float32)

    def _select_variation(self) -> int:
        count = int(self._task.variation_count())
        if self.variation == "random":
            selected = int(self.np_random.integers(count))
        else:
            selected = int(self.variation)
        self._task.set_variation(selected)
        return selected

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._environment is None:
            self.launch()
        if seed is not None:
            np.random.seed(seed)
        variation = self._select_variation()
        descriptions, observation = self._task.reset()
        self._capture_rgb_frame(observation)
        metrics = extract_task_metrics(self.task_name, self._task)
        self._step_count = 0
        self._episode_return = 0.0
        self._collision_count = 0
        self._smoothness_sum = 0.0
        self._previous_action = None
        self._previous_control_distance = metrics.control_distance
        self._previous_approach_distance = metrics.approach_distance
        self._previous_task_distance = metrics.task_distance
        info = {
            "task": self.task_name,
            "robot": self.robot,
            "variation": variation,
            "descriptions": descriptions,
            "final_distance": metrics.final_distance,
            "control_distance": metrics.control_distance,
        }
        return self.state_from_observation(observation), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        normalized_action = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        )
        rlbench_action = self.denormalize_action(normalized_action)
        observation, sparse_reward, terminated = self._task.step(rlbench_action)
        self._capture_rgb_frame(observation)
        self._step_count += 1

        metrics = extract_task_metrics(self.task_name, self._task)
        collision = collision_detected(self._task)
        smoothness = (
            0.0
            if self._previous_action is None
            else float(np.mean(np.square(normalized_action - self._previous_action)))
        )
        progress = (
            0.0
            if self._previous_control_distance is None
            else self._previous_control_distance - metrics.control_distance
        )
        approach_progress = (
            0.0
            if self._previous_approach_distance is None
            else self._previous_approach_distance - metrics.approach_distance
        )
        task_progress = (
            0.0
            if self._previous_task_distance is None
            else self._previous_task_distance - metrics.task_distance
        )
        success = bool(sparse_reward > 0.0)
        reward = (
            self.reward_config.progress_scale * approach_progress
            + self.reward_config.task_progress_scale * task_progress
            + self.reward_config.success_bonus * float(success)
            - self.reward_config.step_penalty
            - self.reward_config.collision_penalty * float(collision)
            - self.reward_config.smoothness_penalty * smoothness
        )
        self._episode_return += reward
        self._collision_count += int(collision)
        self._smoothness_sum += smoothness
        self._previous_action = normalized_action.copy()
        self._previous_control_distance = metrics.control_distance
        self._previous_approach_distance = metrics.approach_distance
        self._previous_task_distance = metrics.task_distance
        truncated = self._step_count >= self.max_episode_steps and not terminated

        info: dict[str, Any] = {
            "task": self.task_name,
            "robot": self.robot,
            "success": success,
            "sparse_reward": float(sparse_reward),
            "progress": float(progress),
            "approach_progress": float(approach_progress),
            "task_progress": float(task_progress),
            "success_bonus": self.reward_config.success_bonus * float(success),
            "step_penalty": self.reward_config.step_penalty,
            "collision": bool(collision),
            "smoothness": smoothness,
            "final_distance": metrics.final_distance,
            "control_distance": metrics.control_distance,
            "step": self._step_count,
        }
        if terminated or truncated:
            info["episode_metrics"] = {
                "task": self.task_name,
                "robot": self.robot,
                "success": success,
                "episode_reward": self._episode_return,
                "episode_length": self._step_count,
                "final_distance": metrics.final_distance,
                "control_distance": metrics.control_distance,
                "collision_count": self._collision_count,
                "collision_rate": self._collision_count / self._step_count,
                "action_smoothness": self._smoothness_sum
                / max(self._step_count - 1, 1),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        return (
            self.state_from_observation(observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def describe(self) -> dict[str, Any]:
        low, high = self.joint_limits
        return {
            "task": self.task_name,
            "robot": self.robot,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_layout": STATE_LAYOUT,
            "normalized_action_bounds": [-1.0, 1.0],
            "joint_lower_limits": low,
            "joint_upper_limits": high,
            "reward": asdict(self.reward_config),
            "max_episode_steps": self.max_episode_steps,
        }

    def close(self) -> None:
        if self._environment is not None:
            self._environment.shutdown()
            self._environment = None
            self._task = None
