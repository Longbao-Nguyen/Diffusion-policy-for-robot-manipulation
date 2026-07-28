from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10_000) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device) * -scale
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.dimension % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int, dilation: int) -> None:
        super().__init__()
        self.condition = nn.Linear(condition_dim, channels * 2)
        self.network = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(
                channels, channels, 3, padding=dilation, dilation=dilation
            ),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        conditioned = value * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return value + self.network(conditioned)


class TemporalNoisePredictor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        observation_horizon: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        condition_dim = observation_dim * observation_horizon + time_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.input_projection = nn.Conv1d(action_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(hidden_dim, condition_dim, dilation)
            for dilation in (1, 2, 4, 8)
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, action_dim, 1),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        observations: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time = self.time_embedding(timesteps)
        condition = torch.cat([observations, time], dim=-1)
        value = self.input_projection(noisy_actions.transpose(1, 2))
        for block in self.blocks:
            value = block(value, condition)
        return self.output_projection(value).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        observation_horizon: int = 2,
        action_horizon: int = 8,
        diffusion_steps: int = 50,
        hidden_dim: int = 256,
        noise_schedule: str = "linear",
        sample_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.observation_horizon = observation_horizon
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.diffusion_steps = diffusion_steps
        self.hidden_dim = hidden_dim
        self.noise_schedule = noise_schedule
        self.sample_clip = float(sample_clip)
        self.noise_predictor = TemporalNoisePredictor(
            observation_dim,
            observation_horizon,
            action_dim,
            action_horizon,
            hidden_dim=hidden_dim,
        )
        if noise_schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, diffusion_steps)
        elif noise_schedule == "cosine":
            # Nichol & Dhariwal cosine schedule. Unlike the short linear
            # schedule, this makes the terminal training distribution nearly
            # pure Gaussian noise, matching how sampling is initialized.
            steps = torch.arange(diffusion_steps + 1, dtype=torch.float32)
            s = 0.008
            alpha_bar_curve = torch.cos(
                ((steps / diffusion_steps + s) / (1.0 + s)) * math.pi / 2.0
            ).square()
            alpha_bar_curve = alpha_bar_curve / alpha_bar_curve[0]
            betas = 1.0 - alpha_bar_curve[1:] / alpha_bar_curve[:-1]
            betas = betas.clamp(1e-5, 0.999)
        else:
            raise ValueError(f"Unsupported noise schedule: {noise_schedule}")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def training_loss(
        self,
        observations: torch.Tensor,
        action_chunks: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = observations.shape[0]
        timesteps = torch.randint(
            0, self.diffusion_steps, (batch,), device=observations.device
        )
        noise = torch.randn_like(action_chunks)
        alpha_bar = self.alpha_bars[timesteps].view(batch, 1, 1)
        noisy = alpha_bar.sqrt() * action_chunks + (1.0 - alpha_bar).sqrt() * noise
        predicted = self.noise_predictor(noisy, observations, timesteps)
        squared_error = (predicted - noise).square()
        if valid_mask is None:
            return squared_error.mean()
        weights = valid_mask.unsqueeze(-1)
        return (squared_error * weights).sum() / (
            weights.sum() * action_chunks.shape[-1]
        ).clamp_min(1.0)

    @torch.no_grad()
    def sample(
        self,
        normalized_observations: torch.Tensor,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch = normalized_observations.shape[0]
        actions = torch.randn(
            batch,
            self.action_horizon,
            self.action_dim,
            device=normalized_observations.device,
            generator=generator,
        )
        for step in reversed(range(self.diffusion_steps)):
            timesteps = torch.full(
                (batch,),
                step,
                device=normalized_observations.device,
                dtype=torch.long,
            )
            predicted_noise = self.noise_predictor(
                actions, normalized_observations, timesteps
            )
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            if deterministic:
                # DDIM update with eta=0.
                predicted_clean = (
                    actions - torch.sqrt(1.0 - alpha_bar) * predicted_noise
                ) / torch.sqrt(alpha_bar)
                predicted_clean = predicted_clean.clamp(
                    -self.sample_clip, self.sample_clip
                )
                if step > 0:
                    previous_alpha_bar = self.alpha_bars[step - 1]
                    actions = (
                        torch.sqrt(previous_alpha_bar) * predicted_clean
                        + torch.sqrt(1.0 - previous_alpha_bar) * predicted_noise
                    )
                else:
                    actions = predicted_clean
            else:
                mean = (
                    actions
                    - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
                    * predicted_noise
                ) / torch.sqrt(alpha)
                if step > 0:
                    actions = mean + torch.sqrt(self.betas[step]) * torch.randn_like(
                        actions
                    )
                else:
                    actions = mean
        return actions.clamp(-self.sample_clip, self.sample_clip)

    def config(self) -> dict[str, int | float | str]:
        return {
            "observation_dim": self.observation_dim,
            "observation_horizon": self.observation_horizon,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "diffusion_steps": self.diffusion_steps,
            "hidden_dim": self.hidden_dim,
            "noise_schedule": self.noise_schedule,
            "sample_clip": self.sample_clip,
        }


class DiffusionPolicyRunner:
    def __init__(
        self,
        model: DiffusionPolicy,
        observation_mean: np.ndarray,
        observation_std: np.ndarray,
        device: torch.device,
        execute_horizon: int = 1,
        stochastic_sampling: bool = False,
        seed: int = 0,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
        observation_clip: float | None = None,
        force_gripper_open: bool = False,
    ) -> None:
        self.model = model.to(device).eval()
        self.mean = torch.as_tensor(
            observation_mean, dtype=torch.float32, device=device
        )
        self.std = torch.as_tensor(
            observation_std, dtype=torch.float32, device=device
        )
        self.device = device
        self.execute_horizon = max(1, int(execute_horizon))
        self.stochastic_sampling = stochastic_sampling
        self._queue: list[np.ndarray] = []
        self._history: list[np.ndarray] = []
        self._seed = int(seed)
        self._generator = torch.Generator(device=device)
        self.action_mean = torch.as_tensor(
            np.zeros(model.action_dim, dtype=np.float32)
            if action_mean is None
            else action_mean,
            dtype=torch.float32,
            device=device,
        )
        self.action_std = torch.as_tensor(
            np.ones(model.action_dim, dtype=np.float32)
            if action_std is None
            else action_std,
            dtype=torch.float32,
            device=device,
        )
        self.observation_clip = observation_clip
        self.force_gripper_open = bool(force_gripper_open)

    def reset(self) -> None:
        self._queue.clear()
        self._history.clear()
        self._generator.manual_seed(self._seed)

    @torch.no_grad()
    def predict(self, observation: np.ndarray) -> np.ndarray:
        # Keep a true consecutive observation history even when multiple
        # actions from the previous chunk are executed open-loop.
        self._history.append(np.asarray(observation, dtype=np.float32))
        self._history = self._history[-self.model.observation_horizon :]
        if not self._queue:
            padded = [self._history[0]] * (
                self.model.observation_horizon - len(self._history)
            ) + self._history
            value = torch.as_tensor(
                np.stack(padded), dtype=torch.float32, device=self.device
            )
            normalized = ((value - self.mean) / self.std).reshape(1, -1)
            if self.observation_clip is not None:
                normalized = normalized.clamp(
                    -float(self.observation_clip), float(self.observation_clip)
                )
            chunk = self.model.sample(
                normalized,
                deterministic=not self.stochastic_sampling,
                generator=self._generator,
            )[0]
            count = min(self.execute_horizon, chunk.shape[0])
            denormalized = chunk[:count] * self.action_std + self.action_mean
            denormalized = denormalized.clamp(-1.0, 1.0)
            if self.force_gripper_open:
                denormalized[:, -1] = 1.0
            self._queue = [
                action.detach().cpu().numpy().astype(np.float32)
                for action in denormalized
            ]
        return self._queue.pop(0)
