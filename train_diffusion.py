from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from arl_robot.constants import SUPPORTED_TASKS
from arl_robot.demo_data import (
    ActionNormalizationStats,
    ActionChunkDataset,
    NormalizationStats,
    load_demo_archive,
    split_episode_ids,
)
from arl_robot.diffusion import DiffusionPolicy
from arl_robot.io_utils import (
    append_csv,
    runtime_metadata,
    set_global_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--observation-horizon", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--noise-schedule", choices=("linear", "cosine"), default="cosine"
    )
    parser.add_argument("--observation-std-floor", type=float, default=0.05)
    parser.add_argument("--action-std-floor", type=float, default=0.05)
    parser.add_argument("--observation-clip", type=float, default=5.0)
    parser.add_argument("--sample-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def validation_loss(
    model: DiffusionPolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for observations, actions, mask in loader:
        losses.append(
            float(
                model.training_loss(
                    observations.to(device),
                    actions.to(device),
                    mask.to(device),
                ).item()
            )
        )
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    archive = load_demo_archive(args.dataset)
    split = split_episode_ids(archive["episode_ids"], args.seed)
    train_mask = np.isin(archive["episode_ids"], split["train"])
    stats = NormalizationStats.from_observations(
        archive["observations"][train_mask], args.observation_std_floor
    )
    action_stats = ActionNormalizationStats.from_actions(
        archive["actions"][train_mask], args.action_std_floor
    )
    train_dataset = ActionChunkDataset(
        archive,
        split["train"],
        args.action_horizon,
        stats,
        args.observation_horizon,
        action_stats,
    )
    validation_dataset = ActionChunkDataset(
        archive,
        split["validation"],
        args.action_horizon,
        stats,
        args.observation_horizon,
        action_stats,
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = DiffusionPolicy(
        observation_dim=archive["observations"].shape[1],
        observation_horizon=args.observation_horizon,
        action_dim=archive["actions"].shape[1],
        action_horizon=args.action_horizon,
        diffusion_steps=args.diffusion_steps,
        hidden_dim=args.hidden_dim,
        noise_schedule=args.noise_schedule,
        sample_clip=args.sample_clip,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-6
    )
    metrics_path = args.output / "training_metrics.csv"
    best_validation = float("inf")
    started = time.time()
    global_step = 0

    config = {
        **vars(args),
        "dataset": str(args.dataset.resolve()),
        "output": str(args.output.resolve()),
        "device_resolved": str(device),
        "model": model.config(),
        "split_episode_ids": split,
        "normalization": stats.as_dict(),
        "action_normalization": action_stats.as_dict(),
        "runtime": runtime_metadata(),
    }
    write_json(args.output / "run_config.json", config)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        epoch_started = time.time()
        for observations, action_chunks, valid_mask in train_loader:
            observations = observations.to(device)
            action_chunks = action_chunks.to(device)
            valid_mask = valid_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(
                observations, action_chunks, valid_mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_model.parameters(), model.parameters()
                ):
                    ema_parameter.mul_(0.995).add_(parameter, alpha=0.005)
            losses.append(float(loss.item()))
            global_step += 1
        val_loss = validation_loss(ema_model, validation_loader, device)
        train_loss = float(np.mean(losses))
        append_csv(
            metrics_path,
            {
                "algorithm": "diffusion",
                "task": args.task,
                "seed": args.seed,
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "epoch_seconds": time.time() - epoch_started,
                "elapsed_seconds": time.time() - started,
            },
        )
        checkpoint = {
            "model_state_dict": ema_model.state_dict(),
            "model_config": model.config(),
            "observation_mean": stats.observation_mean,
            "observation_std": stats.observation_std,
            "action_mean": action_stats.action_mean,
            "action_std": action_stats.action_std,
            "observation_clip": args.observation_clip,
            "task": args.task,
            "robot": "panda",
            "seed": args.seed,
            "epoch": epoch,
            "validation_loss": val_loss,
            "ema_decay": 0.995,
        }
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if val_loss < best_validation:
            best_validation = val_loss
            torch.save(checkpoint, args.output / "checkpoint.pt")
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"validation_loss={val_loss:.6f}",
            flush=True,
        )
    write_json(
        args.output / "training_summary.json",
        {
            "algorithm": "diffusion",
            "task": args.task,
            "seed": args.seed,
            "best_validation_loss": best_validation,
            "epochs": args.epochs,
            "global_steps": global_step,
            "elapsed_seconds": time.time() - started,
            "checkpoint": str((args.output / "checkpoint.pt").resolve()),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
