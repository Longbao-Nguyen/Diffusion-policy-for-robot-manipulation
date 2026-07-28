from __future__ import annotations

SUPPORTED_TASKS = (
    "reach_target",
    "push_button",
    "slide_block_to_target",
)

TASK_DIFFICULTY = {
    "reach_target": "easy",
    "push_button": "medium",
    "slide_block_to_target": "hard",
}

DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_ROBOT = "panda"
PANDA_ARM_DOF = 7
PANDA_ACTION_DIM = 8

