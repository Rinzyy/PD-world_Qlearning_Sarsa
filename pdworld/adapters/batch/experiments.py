from __future__ import annotations

import json
from pathlib import Path

from pdworld.adapters.batch.analysis import (
    generate_exp2_attractive_paths_from_snapshots,
    plot_cumulative_reward,
    plot_episode_lengths,
    plot_q_heatmap,
    save_episode_lengths_csv,
    save_q_snapshot_csv,
    save_timeseries_csv,
)
from pdworld.core.constants import DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_SEEDS
from pdworld.adapters.batch.runner import run_steps
from pdworld.core.types import LearnerType, Policy, RunConfig, RunResult


def get_experiment_config(exp_id: int, seed: int) -> RunConfig:
    if exp_id == 1:
        return RunConfig(
            alpha=DEFAULT_ALPHA,
            gamma=DEFAULT_GAMMA,
            total_steps=8000,
            schedule=[(1, 4000, Policy.PRANDOM), (4001, 8000, Policy.PGREEDY)],
            seed=seed,
            learner=LearnerType.Q_LEARNING,
            snapshot_steps={"mid": 4000, "final": 8000},
            experiment_id=1,
        )

    if exp_id == 2:
        return RunConfig(
            alpha=DEFAULT_ALPHA,
            gamma=DEFAULT_GAMMA,
            total_steps=8000,
            schedule=[(1, 200, Policy.PRANDOM), (201, 8000, Policy.PEXPLOIT)],
            seed=seed,
            learner=LearnerType.Q_LEARNING,
            snapshot_steps={"final": 8000},
            capture_first_full_dropoff=True,
            capture_first_terminal=True,
            experiment_id=2,
        )

    if exp_id == 3:
        return RunConfig(
            alpha=DEFAULT_ALPHA,
            gamma=DEFAULT_GAMMA,
            total_steps=8000,
            schedule=[(1, 200, Policy.PRANDOM), (201, 8000, Policy.PEXPLOIT)],
            seed=seed,
            learner=LearnerType.SARSA,
            snapshot_steps={"final": 8000},
            experiment_id=3,
        )

    raise ValueError(f"Unsupported experiment id: {exp_id}")


def write_run_artifacts(exp_id: int, seed: int, result: RunResult, output_root: Path) -> Path:
    run_dir = output_root / f"exp{exp_id}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_timeseries_csv(result.step_rewards, result.cumulative_rewards, run_dir / "timeseries.csv")
    save_episode_lengths_csv(result.episode_lengths, run_dir / "episode_lengths.csv")

    for snapshot_name, q_values in result.q_snapshots.items():
        save_q_snapshot_csv(q_values, run_dir / f"q_{snapshot_name}.csv")
        plot_q_heatmap(
            q_values,
            run_dir / f"q_{snapshot_name}.png",
            title=f"Experiment {exp_id}, seed {seed} - Q snapshot: {snapshot_name}",
        )

    plot_cumulative_reward(
        result.cumulative_rewards,
        run_dir / "cumulative_reward.png",
        title=f"Experiment {exp_id}, seed {seed} - Cumulative Reward",
    )
    plot_episode_lengths(
        result.episode_lengths,
        run_dir / "episode_lengths.png",
        title=f"Experiment {exp_id}, seed {seed} - Episode Length",
    )

    if exp_id == 2:
        generate_exp2_attractive_paths_from_snapshots(result.q_snapshots, run_dir)

    metadata = {
        "experiment_id": exp_id,
        "seed": seed,
        "terminal_hits": result.terminal_hits,
        "episodes_completed": len(result.episode_lengths),
        "final_cumulative_reward": result.cumulative_rewards[-1] if result.cumulative_rewards else 0.0,
        "available_snapshots": sorted(result.q_snapshots.keys()),
        **result.metadata,
    }

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return run_dir


def run_experiment(exp_id: int, seed: int, output_root: Path) -> tuple[RunResult, Path]:
    config = get_experiment_config(exp_id, seed)
    result = run_steps(config)
    run_dir = write_run_artifacts(exp_id, seed, result, output_root)
    return result, run_dir


def run_all_experiments(output_root: Path, seeds: list[int] | None = None) -> list[dict[str, object]]:
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)

    summary: list[dict[str, object]] = []
    for exp_id in (1, 2, 3):
        for seed in seeds:
            result, run_dir = run_experiment(exp_id, seed, output_root)
            summary.append(
                {
                    "experiment": exp_id,
                    "seed": seed,
                    "terminal_hits": result.terminal_hits,
                    "episodes_completed": len(result.episode_lengths),
                    "run_dir": str(run_dir),
                }
            )
    return summary
