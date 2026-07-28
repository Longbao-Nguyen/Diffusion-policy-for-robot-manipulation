from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arl_robot.io_utils import write_json


ALGORITHM_ORDER = ["diffusion", "ppo", "sac", "reinforce"]
TASK_ORDER = ["reach_target", "push_button", "slide_block_to_target"]
COLORS = {
    "diffusion": "#4C78A8",
    "ppo": "#F58518",
    "sac": "#54A24B",
    "reinforce": "#E45756",
}


def is_primary_result(path: Path, root: Path) -> bool:
    """Exclude smoke-test and generated plot artifacts from final results."""
    relative = path.relative_to(root)
    run_name = relative.parts[0] if relative.parts else ""
    return "_failed_" not in run_name and not any(
        path.is_relative_to(root / excluded)
        for excluded in ("smoke", "plots")
    )


def is_canonical_result_file(path: Path, root: Path, filename: str) -> bool:
    """Accept only run-level files and the canonical run/evaluation files.

    Evaluation shards and archived evaluation directories are intentionally
    retained for provenance, but must not be counted as independent runs.
    """
    if not is_primary_result(path, root):
        return False
    relative = path.relative_to(root)
    if filename == "episode_metrics.csv":
        return len(relative.parts) == 2 or (
            len(relative.parts) == 3 and relative.parts[1] == "evaluation"
        )
    return len(relative.parts) == 2


def numeric_success(series: pd.Series) -> pd.Series:
    """Normalize CSV boolean and numeric success encodings to float."""
    normalized = series.astype(str).str.strip().str.lower().replace(
        {"true": "1", "false": "0"}
    )
    return pd.to_numeric(normalized, errors="coerce")


def keep_last_environment_step_segment(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop partial data preceding the last restarted training segment."""
    if "environment_steps" not in frame or frame.empty:
        return frame
    steps = pd.to_numeric(frame["environment_steps"], errors="coerce")
    resets = np.flatnonzero(steps.diff().fillna(0).to_numpy() < 0)
    if len(resets):
        return frame.iloc[int(resets[-1]) :].reset_index(drop=True)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_matching(root: Path, filename: str) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    sources = []
    for path in sorted(root.rglob(filename)):
        if not is_canonical_result_file(path, root, filename):
            continue
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            continue
        frame = keep_last_environment_step_segment(frame)
        frame["source_file"] = str(path.resolve())
        frames.append(frame)
        sources.append(str(path.resolve()))
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        sources,
    )


def read_json_matching(root: Path, filename: str) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    sources = []
    for path in sorted(root.rglob(filename)):
        if not is_canonical_result_file(path, root, filename):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload["source_file"] = str(path.resolve())
        rows.append(payload)
        sources.append(str(path.resolve()))
    return pd.DataFrame(rows), sources


def save_figure(
    figure: plt.Figure,
    output: Path,
    name: str,
    manifest: list[dict],
    sources: list[str],
    description: str,
) -> None:
    target = output / name
    figure.tight_layout()
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    manifest.append(
        {
            "plot": str(target.resolve()),
            "description": description,
            "source_files": sources,
        }
    )


def grouped_bar(
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    filename: str,
    manifest: list[dict],
    sources: list[str],
    success_only: bool = False,
) -> None:
    frame = data.copy()
    if frame.empty or metric not in frame:
        return
    if success_only:
        if "success" not in frame:
            return
        frame = frame[frame["success"].astype(float) > 0.5]
    if frame.empty:
        return
    per_run = (
        frame.groupby(["source_file", "task", "algorithm", "seed"])[metric]
        .mean()
        .reset_index()
    )
    grouped = (
        per_run.groupby(["task", "algorithm"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASK_ORDER))
    width = 0.19
    for index, algorithm in enumerate(ALGORITHM_ORDER):
        subset = grouped[grouped["algorithm"] == algorithm].set_index("task")
        means = [subset["mean"].get(task, np.nan) for task in TASK_ORDER]
        stds = [subset["std"].get(task, 0.0) for task in TASK_ORDER]
        axis.bar(
            x + (index - 1.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=algorithm,
            color=COLORS[algorithm],
        )
    axis.set_xticks(x, [task.replace("_", "\n") for task in TASK_ORDER])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, output, filename, manifest, sources, title)


def learning_curves(
    episodes: pd.DataFrame,
    output: Path,
    manifest: list[dict],
    sources: list[str],
) -> None:
    required = {"algorithm", "task", "environment_steps", "episode_reward"}
    if episodes.empty or not required.issubset(episodes.columns):
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for axis, task in zip(axes, TASK_ORDER):
        task_data = episodes[episodes["task"] == task].copy()
        for algorithm in ("ppo", "sac", "reinforce"):
            subset = task_data[task_data["algorithm"] == algorithm]
            if subset.empty:
                continue
            runs = []
            grid = np.linspace(
                subset["environment_steps"].min(),
                subset["environment_steps"].max(),
                100,
            )
            for _, run in subset.groupby("source_file"):
                run = run.sort_values("environment_steps")
                rolling = run["episode_reward"].rolling(
                    20, min_periods=1
                ).mean()
                runs.append(
                    np.interp(grid, run["environment_steps"], rolling)
                )
            values = np.vstack(runs)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            axis.plot(grid, mean, label=algorithm, color=COLORS[algorithm])
            axis.fill_between(
                grid,
                mean - std,
                mean + std,
                color=COLORS[algorithm],
                alpha=0.18,
            )
        axis.set_title(task.replace("_", " "))
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Rolling mean episode reward")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=len(labels))
    save_figure(
        figure,
        output,
        "learning_curves_reward.png",
        manifest,
        sources,
        "Online learning curves: rolling mean reward ± standard deviation",
    )

    success_figure, success_axes = plt.subplots(
        1, 3, figsize=(15, 4.5), sharey=True
    )
    for axis, task in zip(success_axes, TASK_ORDER):
        task_data = episodes[episodes["task"] == task].copy()
        for algorithm in ("ppo", "sac", "reinforce"):
            subset = task_data[task_data["algorithm"] == algorithm]
            if subset.empty or "success" not in subset:
                continue
            grid = np.linspace(
                subset["environment_steps"].min(),
                subset["environment_steps"].max(),
                100,
            )
            runs = []
            for _, run in subset.groupby("source_file"):
                run = run.sort_values("environment_steps")
                rolling = run["success"].astype(float).rolling(
                    20, min_periods=1
                ).mean()
                runs.append(
                    np.interp(grid, run["environment_steps"], rolling)
                )
            values = np.vstack(runs)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            axis.plot(
                grid,
                mean,
                color=COLORS[algorithm],
                label=algorithm,
                linewidth=2,
            )
            axis.fill_between(
                grid,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=COLORS[algorithm],
                alpha=0.18,
            )
        axis.set_title(task.replace("_", " "))
        axis.set_xlabel("Environment steps")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    success_axes[0].set_ylabel("Rolling success rate")
    handles, labels = success_axes[0].get_legend_handles_labels()
    if handles:
        success_figure.legend(
            handles, labels, loc="upper center", ncol=len(labels)
        )
    save_figure(
        success_figure,
        output,
        "sample_efficiency_success.png",
        manifest,
        sources,
        "Online sample efficiency measured by rolling success rate",
    )


def diffusion_losses(
    training: pd.DataFrame,
    output: Path,
    manifest: list[dict],
    sources: list[str],
) -> None:
    if training.empty or "train_loss" not in training:
        return
    frame = training[training["algorithm"] == "diffusion"]
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, task in zip(axes, TASK_ORDER):
        subset = frame[frame["task"] == task]
        for source, run in subset.groupby("source_file"):
            axis.plot(
                run["epoch"],
                run["train_loss"],
                color=COLORS["diffusion"],
                alpha=0.35,
            )
            axis.plot(
                run["epoch"],
                run["validation_loss"],
                color="#B279A2",
                alpha=0.35,
            )
        axis.set_title(task.replace("_", " "))
        axis.set_xlabel("Epoch")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Noise prediction MSE")
    axes[-1].plot([], [], color=COLORS["diffusion"], label="train")
    axes[-1].plot([], [], color="#B279A2", label="validation")
    axes[-1].legend()
    save_figure(
        figure,
        output,
        "diffusion_training_loss.png",
        manifest,
        sources,
        "Diffusion Policy train and validation losses",
    )


def representative_trajectories(
    root: Path,
    output: Path,
    manifest: list[dict],
) -> None:
    records = []
    for path in sorted(root.rglob("trajectories.npz")):
        relative = path.relative_to(root)
        if not (
            is_primary_result(path, root)
            and len(relative.parts) == 3
            and relative.parts[1] == "evaluation"
        ):
            continue
        config_path = path.parent / "evaluation_config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append((config["task"], config["algorithm"], path))
    for task in TASK_ORDER:
        selected = {}
        for record_task, algorithm, path in records:
            if record_task == task and algorithm not in selected:
                selected[algorithm] = path
        if not selected:
            continue
        figure, (trajectory_axis, action_axis) = plt.subplots(
            1, 2, figsize=(12, 5)
        )
        used_sources = []
        for algorithm in ALGORITHM_ORDER:
            path = selected.get(algorithm)
            if path is None:
                continue
            with np.load(path, allow_pickle=False) as archive:
                episode_ids = archive["episode_ids"]
                first_episode = int(episode_ids.min())
                mask = episode_ids == first_episode
                observations = archive["observations"][mask]
                actions = archive["actions"][mask]
            positions = observations[:, 15:18]
            trajectory_axis.plot(
                positions[:, 0],
                positions[:, 1],
                label=algorithm,
                color=COLORS[algorithm],
            )
            changes = np.zeros(len(actions), dtype=np.float32)
            if len(actions) > 1:
                changes[1:] = np.linalg.norm(np.diff(actions, axis=0), axis=1)
            action_axis.plot(
                changes,
                label=algorithm,
                color=COLORS[algorithm],
            )
            used_sources.append(str(path.resolve()))
        trajectory_axis.set_xlabel("End-effector x (m)")
        trajectory_axis.set_ylabel("End-effector y (m)")
        trajectory_axis.set_title("Representative end-effector path")
        trajectory_axis.axis("equal")
        trajectory_axis.grid(alpha=0.25)
        action_axis.set_xlabel("Timestep")
        action_axis.set_ylabel("||a_t - a_(t-1)||")
        action_axis.set_title("Representative action changes")
        action_axis.grid(alpha=0.25)
        action_axis.legend()
        title = f"Representative trajectories: {task}"
        figure.suptitle(title)
        save_figure(
            figure,
            output,
            f"representative_trajectory_{task}.png",
            manifest,
            used_sources,
            title,
        )


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    evaluation, evaluation_sources = read_matching(
        args.input, "episode_metrics.csv"
    )
    if "success" in evaluation:
        evaluation["success"] = numeric_success(evaluation["success"])
    training, training_sources = read_matching(
        args.input, "training_metrics.csv"
    )
    summaries, summary_sources = read_json_matching(
        args.input, "training_summary.json"
    )

    # A training run and its evaluation can both contain episode_metrics.csv.
    # Evaluation rows have inference latency; use those for final comparisons.
    final_evaluation = (
        evaluation[evaluation["mean_inference_ms"].notna()].copy()
        if "mean_inference_ms" in evaluation
        else pd.DataFrame()
    )
    online_episodes = (
        evaluation[evaluation["environment_steps"].notna()].copy()
        if "environment_steps" in evaluation
        else pd.DataFrame()
    )
    if not final_evaluation.empty:
        final_evaluation.to_csv(
            data_dir / "combined_evaluation_episodes.csv", index=False
        )
        metric_columns = [
            column
            for column in (
                "success",
                "episode_reward",
                "episode_length",
                "final_distance",
                "collision_rate",
                "action_smoothness",
                "mean_inference_ms",
            )
            if column in final_evaluation
        ]
        per_run = (
            final_evaluation.groupby(
                ["source_file", "task", "algorithm", "seed"]
            )[metric_columns]
            .mean()
            .reset_index()
        )
        aggregate = per_run.groupby(["task", "algorithm"])[
            metric_columns
        ].agg(["mean", "std", "count"])
        aggregate.columns = [
            f"{metric}_{statistic}"
            for metric, statistic in aggregate.columns
        ]
        aggregate.reset_index().to_csv(
            data_dir / "evaluation_summary_mean_std.csv", index=False
        )
    if not online_episodes.empty:
        online_episodes.to_csv(
            data_dir / "combined_online_training_episodes.csv", index=False
        )
    if not training.empty:
        training.to_csv(data_dir / "combined_training_metrics.csv", index=False)
    if not summaries.empty:
        summaries.to_csv(data_dir / "combined_training_summaries.csv", index=False)

    manifest: list[dict] = []
    metrics = [
        (
            "success",
            "Success rate",
            "Success rate by task and algorithm",
            "success_rate.png",
            False,
        ),
        (
            "episode_reward",
            "Mean episode reward",
            "Episode reward by task and algorithm",
            "episode_reward.png",
            False,
        ),
        (
            "episode_length",
            "Steps",
            "Steps to success",
            "steps_to_success.png",
            True,
        ),
        (
            "final_distance",
            "Distance (m)",
            "Final distance to target",
            "final_distance.png",
            False,
        ),
        (
            "collision_rate",
            "Collision rate",
            "Collision rate",
            "collision_rate.png",
            False,
        ),
        (
            "action_smoothness",
            "Mean squared action change",
            "Action smoothness",
            "action_smoothness.png",
            False,
        ),
        (
            "mean_inference_ms",
            "Milliseconds / action",
            "Inference latency",
            "inference_latency.png",
            False,
        ),
    ]
    for metric, ylabel, title, filename, success_only in metrics:
        grouped_bar(
            final_evaluation,
            metric,
            ylabel,
            title,
            args.output,
            filename,
            manifest,
            evaluation_sources,
            success_only,
        )
    learning_curves(
        online_episodes,
        args.output,
        manifest,
        evaluation_sources,
    )
    diffusion_losses(
        training,
        args.output,
        manifest,
        training_sources,
    )
    if not summaries.empty and "elapsed_seconds" in summaries:
        grouped_bar(
            summaries,
            "elapsed_seconds",
            "Training time (seconds)",
            "Training wall-clock time",
            args.output,
            "training_time.png",
            manifest,
            summary_sources,
        )
    representative_trajectories(args.input, args.output, manifest)
    write_json(
        args.output / "plot_manifest.json",
        {
            "input_root": str(args.input.resolve()),
            "plots": manifest,
            "combined_data": [
                str(path.resolve()) for path in sorted(data_dir.glob("*.csv"))
            ],
        },
    )
    print(
        json.dumps(
            {
                "plots_written": len(manifest),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
