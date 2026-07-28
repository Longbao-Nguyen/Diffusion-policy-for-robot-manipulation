from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape


@dataclass
class TaskMetrics:
    task_features: np.ndarray
    final_distance: float
    control_distance: float
    approach_distance: float
    task_distance: float


def _position(obj) -> np.ndarray:
    return np.asarray(obj.get_position(), dtype=np.float32)


def extract_task_metrics(task_name: str, task_environment) -> TaskMetrics:
    """Return a fixed 7D privileged task state and task-aware distances.

    The state is intentionally fixed-width so all tasks share the same robot
    observation schema. Each task is still trained with a separate policy.
    """
    tip = _position(task_environment._robot.arm.get_tip())

    if task_name == "reach_target":
        target = _position(Shape("target"))
        distance = float(np.linalg.norm(tip - target))
        features = np.concatenate([target, np.zeros(3), [distance]])
        return TaskMetrics(
            features.astype(np.float32), distance, distance, distance, 0.0
        )

    if task_name == "push_button":
        button = _position(Shape("push_button_target"))
        joint_position = float(Joint("target_button_joint").get_joint_position())
        distance = float(np.linalg.norm(tip - button))
        features = np.concatenate(
            [button, np.zeros(3), [joint_position]]
        )
        button_remaining = max(0.0, 0.003 - abs(joint_position))
        return TaskMetrics(
            features.astype(np.float32),
            distance,
            distance + button_remaining,
            distance,
            button_remaining,
        )

    if task_name == "slide_block_to_target":
        block = _position(Shape("block"))
        target = _position(ProximitySensor("success"))
        block_to_target = float(np.linalg.norm(block - target))
        tip_to_block = float(np.linalg.norm(tip - block))
        features = np.concatenate([block, target, [block_to_target]])
        control_distance = tip_to_block + block_to_target
        return TaskMetrics(
            features.astype(np.float32),
            block_to_target,
            control_distance,
            tip_to_block,
            block_to_target,
        )

    raise ValueError(f"Unsupported task: {task_name}")


def collision_detected(task_environment) -> bool:
    """Log arm collision status without changing task physics."""
    try:
        return bool(task_environment._robot.arm.check_arm_collision())
    except Exception:
        return False
