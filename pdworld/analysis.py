from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="xdg-cache-")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplconfig-")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pdworld.constants import ACTIONS, ACTION_TO_INDEX, GRID_SIZE, MOVE_ACTIONS
from pdworld.state_mapping import id_to_state, state_to_id


def save_timeseries_csv(step_rewards: list[int], cumulative_rewards: list[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward", "cumulative_reward"])
        for idx, (reward, cumulative) in enumerate(zip(step_rewards, cumulative_rewards), start=1):
            writer.writerow([idx, reward, cumulative])


def save_episode_lengths_csv(episode_lengths: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "operator_applications"])
        for idx, length in enumerate(episode_lengths, start=1):
            writer.writerow([idx, length])


def save_q_snapshot_csv(q_values: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["state_id", *[action.value for action in ACTIONS]])
        for state_id in range(q_values.shape[0]):
            writer.writerow([state_id, *q_values[state_id, :].tolist()])


def load_q_snapshot_csv(input_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] != "state_id":
            raise ValueError(f"Invalid Q snapshot CSV format: {input_path}")
        for row in reader:
            rows.append([float(v) for v in row[1:]])
    return np.array(rows, dtype=float)


def plot_cumulative_reward(cumulative_rewards: list[float], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(cumulative_rewards) + 1), cumulative_rewards, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward (Bank Account)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_episode_lengths(episode_lengths: list[int], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if episode_lengths:
        ax.plot(range(1, len(episode_lengths) + 1), episode_lengths, marker="o", markersize=2, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Operator Applications to Terminal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_q_heatmap(q_values: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(q_values, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Action")
    ax.set_ylabel("State ID")
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels([a.value for a in ACTIONS], rotation=30, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def attractive_move_grid(q_values: np.ndarray, carrying: int) -> list[list[str]]:
    arrows = {
        MOVE_ACTIONS[0]: "↑",
        MOVE_ACTIONS[1]: "↓",
        MOVE_ACTIONS[2]: "→",
        MOVE_ACTIONS[3]: "←",
    }

    grid: list[list[str]] = [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    move_indices = [ACTION_TO_INDEX[action] for action in MOVE_ACTIONS]

    for row in range(1, GRID_SIZE + 1):
        for col in range(1, GRID_SIZE + 1):
            state_id = state_to_id(row, col, carrying)
            move_values = q_values[state_id, move_indices]
            best_idx = int(np.argmax(move_values))
            best_action = MOVE_ACTIONS[best_idx]
            grid[row - 1][col - 1] = arrows[best_action]

    return grid


def plot_attractive_path_grid(grid: list[list[str]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.5, GRID_SIZE + 0.5)
    ax.set_ylim(0.5, GRID_SIZE + 0.5)
    ax.set_xticks(range(1, GRID_SIZE + 1))
    ax.set_yticks(range(1, GRID_SIZE + 1))
    ax.grid(True, linewidth=0.8, alpha=0.7)

    for row in range(1, GRID_SIZE + 1):
        for col in range(1, GRID_SIZE + 1):
            arrow = grid[row - 1][col - 1]
            ax.text(col, GRID_SIZE - row + 1, arrow, ha="center", va="center", fontsize=16)

    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_exp2_attractive_paths_from_snapshots(
    snapshots: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    for snapshot_name in ("first_full_dropoff", "first_terminal", "final"):
        if snapshot_name not in snapshots:
            continue
        q_values = snapshots[snapshot_name]
        for carrying in (0, 1):
            grid = attractive_move_grid(q_values, carrying)
            output_path = output_dir / f"attractive_paths_{snapshot_name}_x{carrying}.png"
            title = f"Attractive Paths ({snapshot_name}, carrying={carrying})"
            plot_attractive_path_grid(grid, output_path, title)


def regenerate_exp2_attractive_paths(exp2_seed_dir: Path) -> None:
    snapshots: dict[str, np.ndarray] = {}
    for name in ("first_full_dropoff", "first_terminal", "final"):
        path = exp2_seed_dir / f"q_{name}.csv"
        if path.exists():
            snapshots[name] = load_q_snapshot_csv(path)
    generate_exp2_attractive_paths_from_snapshots(snapshots, exp2_seed_dir)
