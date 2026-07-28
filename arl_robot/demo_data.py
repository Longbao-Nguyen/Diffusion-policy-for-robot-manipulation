from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "dones",
    "episode_ids",
    "timesteps",
)


def load_demo_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive]
        if missing:
            raise ValueError(f"Demo archive is missing keys: {missing}")
        data = {key: np.asarray(archive[key]) for key in archive.files}
    count = len(data["observations"])
    if any(len(data[key]) != count for key in REQUIRED_KEYS):
        raise ValueError("Demo arrays do not have equal transition counts")
    return data


def split_episode_ids(
    episode_ids: np.ndarray,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    unique = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    train_end = max(1, int(round(len(shuffled) * train_fraction)))
    val_count = int(round(len(shuffled) * validation_fraction))
    val_end = min(len(shuffled), train_end + max(1, val_count))
    if len(shuffled) < 3:
        return {"train": shuffled, "validation": shuffled, "test": shuffled}
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


@dataclass
class NormalizationStats:
    observation_mean: np.ndarray
    observation_std: np.ndarray

    @classmethod
    def from_observations(
        cls, observations: np.ndarray, std_floor: float = 1e-4
    ) -> "NormalizationStats":
        mean = observations.mean(axis=0).astype(np.float32)
        std = observations.std(axis=0).astype(np.float32)
        return cls(mean, np.maximum(std, float(std_floor)))

    def normalize(self, observations: np.ndarray) -> np.ndarray:
        return (observations - self.observation_mean) / self.observation_std

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "observation_mean": self.observation_mean,
            "observation_std": self.observation_std,
        }


@dataclass
class ActionNormalizationStats:
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def from_actions(
        cls, actions: np.ndarray, std_floor: float = 1e-4
    ) -> "ActionNormalizationStats":
        mean = actions.mean(axis=0).astype(np.float32)
        std = actions.std(axis=0).astype(np.float32)
        return cls(mean, np.maximum(std, float(std_floor)))

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        return (actions - self.action_mean) / self.action_std

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "action_mean": self.action_mean,
            "action_std": self.action_std,
        }


class ActionChunkDataset(Dataset):
    """State-conditioned action chunks that never cross episode boundaries."""

    def __init__(
        self,
        archive: dict[str, np.ndarray],
        episode_ids: np.ndarray,
        horizon: int,
        stats: NormalizationStats | None = None,
        observation_horizon: int = 2,
        action_stats: ActionNormalizationStats | None = None,
    ) -> None:
        self.horizon = int(horizon)
        self.observation_horizon = int(observation_horizon)
        mask = np.isin(archive["episode_ids"], episode_ids)
        selected = np.flatnonzero(mask)
        if len(selected) == 0:
            raise ValueError("Dataset split has no transitions")
        self.observations = archive["observations"].astype(np.float32)
        self.actions = archive["actions"].astype(np.float32)
        self.all_episode_ids = archive["episode_ids"].astype(np.int64)
        self.indices = selected
        counts = {
            episode_id: int(np.sum(self.all_episode_ids[selected] == episode_id))
            for episode_id in np.unique(self.all_episode_ids[selected])
        }
        self.sample_weights = np.asarray(
            [1.0 / counts[self.all_episode_ids[index]] for index in selected],
            dtype=np.float64,
        )
        self.stats = stats or NormalizationStats.from_observations(
            self.observations[selected]
        )
        self.action_stats = action_stats

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        start = int(self.indices[index])
        episode_id = self.all_episode_ids[start]
        chunk = []
        valid = []
        last_action = self.actions[start]
        for offset in range(self.horizon):
            position = start + offset
            if (
                position < len(self.actions)
                and self.all_episode_ids[position] == episode_id
            ):
                last_action = self.actions[position]
                valid.append(1.0)
            else:
                valid.append(0.0)
            normalized_action = (
                self.action_stats.normalize(last_action)
                if self.action_stats is not None
                else last_action
            )
            chunk.append(normalized_action)
        history = []
        for offset in reversed(range(self.observation_horizon)):
            position = start - offset
            if (
                position < 0
                or self.all_episode_ids[position] != episode_id
            ):
                position = start
                while (
                    position > 0
                    and self.all_episode_ids[position - 1] == episode_id
                ):
                    position -= 1
            history.append(self.stats.normalize(self.observations[position]))
        observation = np.concatenate(history)
        return (
            torch.from_numpy(observation.astype(np.float32)),
            torch.from_numpy(np.asarray(chunk, dtype=np.float32)),
            torch.from_numpy(np.asarray(valid, dtype=np.float32)),
        )
