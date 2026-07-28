from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class GaussianActor(nn.Module):
    def __init__(
        self, observation_dim: int, action_dim: int, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mean_network = mlp(observation_dim, action_dim, hidden_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.mean_network(observations)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    def sample(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        raw = distribution.rsample()
        action = torch.tanh(raw)
        correction = torch.log(1.0 - action.square() + 1e-6)
        log_probability = (
            distribution.log_prob(raw) - correction
        ).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_probability, entropy

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            action = torch.tanh(self.mean_network(tensor))
        else:
            action, _, _ = self.sample(tensor)
        return action[0].cpu().numpy().astype(np.float32)

    def config(self) -> dict[str, int]:
        return {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        }


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = mlp(observation_dim, 1, hidden_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


def discounted_returns(
    rewards: list[float], gamma: float, device: torch.device
) -> torch.Tensor:
    values = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        values.append(running)
    values.reverse()
    return torch.as_tensor(values, dtype=torch.float32, device=device)

