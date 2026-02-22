from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Callable, Literal

import numpy as np

if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="xdg-cache-")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplconfig-")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from pdworld.core.constants import ACTIONS, ACTION_TO_INDEX, GRID_SIZE, MOVE_ACTIONS
from pdworld.core.state_mapping import id_to_state, state_to_id


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


def export_q_subset_moves(
    q_values: np.ndarray,
    output_path: Path,
    carrying: int,
    agg: Literal["max", "mean"] = "max",
) -> None:
    """
    Exports a report-friendly subset of the Q-table, aggregating over s, t, u,
    and outputting only columns for N, S, E, W actions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    move_indices = [ACTION_TO_INDEX[action] for action in MOVE_ACTIONS]
    header = ["row", "col", *[action.value for action in MOVE_ACTIONS]]
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for row in range(1, GRID_SIZE + 1):
            for col in range(1, GRID_SIZE + 1):
                agg_vals = []
                for s in (0, 1):
                    for t in (0, 1):
                        for u in (0, 1):
                            state_id = state_to_id(row, col, carrying, s, t, u)
                            agg_vals.append(q_values[state_id, move_indices])
                
                if agg == "max":
                    final_vals = np.max(agg_vals, axis=0)
                else:
                    final_vals = np.mean(agg_vals, axis=0)
                    
                writer.writerow([row, col, *final_vals.tolist()])


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


def plot_q_triangles_grid(q_values: np.ndarray, carrying: int, output_path: Path, title: str) -> None:
    """
    Plots an attractive 5x5 grid with triangles for each action N,S,E,W shaded 
    by their Q-value, and an arrow pointing in the best move's direction.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#EBE5D9")
    ax.set_facecolor("white")
    
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    
    move_indices = [ACTION_TO_INDEX[action] for action in MOVE_ACTIONS]
    
    # Calculate global max/min for scaling colors 
    valid_vals = []
    for row in range(1, GRID_SIZE + 1):
        for col in range(1, GRID_SIZE + 1):
            move_values_agg = []
            for s in (0, 1):
                for t in (0, 1):
                    for u in (0, 1):
                        state_id = state_to_id(row, col, carrying, s, t, u)
                        move_values_agg.append(q_values[state_id, move_indices])
            
            agg = np.mean(move_values_agg, axis=0)
            
            if row == 1: agg[0] = -np.inf
            if row == GRID_SIZE: agg[1] = -np.inf
            if col == GRID_SIZE: agg[2] = -np.inf
            if col == 1: agg[3] = -np.inf
            
            valid_vals.extend([float(v) for v in agg if v != -np.inf])
                
    max_v = max(valid_vals) if valid_vals else 1
    min_v = min(valid_vals) if valid_vals else -1
    vmax = max(abs(max_v), abs(min_v), 1)

    # Draw the grid cells
    for row in range(1, GRID_SIZE + 1):
        for col in range(1, GRID_SIZE + 1):
            move_values_agg = []
            for s in (0, 1):
                for t in (0, 1):
                    for u in (0, 1):
                        state_id = state_to_id(row, col, carrying, s, t, u)
                        move_values_agg.append(q_values[state_id, move_indices])

            agg = np.mean(move_values_agg, axis=0)

            if row == 1: agg[0] = -np.inf
            if row == GRID_SIZE: agg[1] = -np.inf
            if col == GRID_SIZE: agg[2] = -np.inf
            if col == 1: agg[3] = -np.inf
            
            x_left   = col - 1
            x_right  = col
            y_top    = row - 1
            y_bottom = row
            center_x = (x_left + x_right) / 2
            center_y = (y_top + y_bottom) / 2
            
            valid_moves = [v for v in agg if v != -np.inf]
            best_idx = int(np.argmax(agg)) if valid_moves else -1
            
            poly_n = Polygon([(x_left, y_top), (x_right, y_top), (center_x, center_y)], closed=True)
            poly_s = Polygon([(x_left, y_bottom), (x_right, y_bottom), (center_x, center_y)], closed=True)
            poly_e = Polygon([(x_right, y_top), (x_right, y_bottom), (center_x, center_y)], closed=True)
            poly_w = Polygon([(x_left, y_top), (x_left, y_bottom), (center_x, center_y)], closed=True)
            polys = [poly_n, poly_s, poly_e, poly_w]
            
            for idx, poly in enumerate(polys):
                val = agg[idx]
                if val == -np.inf:
                    color = "white" # White edge wall
                    text_color = "white"
                    text = ""
                else:
                    if val == 0:
                        color = (1.0, 1.0, 1.0) # Pure white
                        text_color = "#000000"
                    elif val > 0:
                        intensity = val / vmax if vmax > 0 else 0
                        # Pure white parsing to bright rich green
                        color = (1.0 - 1.0 * intensity, 1.0 - 0.5 * intensity, 1.0 - 1.0 * intensity)
                        text_color = "#000000" if intensity < 0.5 else "#ffffff"
                    else:
                        intensity = abs(val) / vmax if vmax > 0 else 0
                        # Pure white parsing to rich bright red
                        color = (1.0 - 0.2 * intensity, 1.0 - 1.0 * intensity, 1.0 - 1.0 * intensity)
                        text_color = "#000000" if intensity < 0.5 else "#ffffff"
                    
                    text = f"{val:.2f}"
                
                poly.set_facecolor(color)
                poly.set_edgecolor('#111111')
                poly.set_linewidth(1.5)
                ax.add_patch(poly)
                
                if text:
                    tx, ty = center_x, center_y
                    if idx == 0: ty -= 0.28
                    elif idx == 1: ty += 0.28
                    elif idx == 2: tx += 0.28
                    elif idx == 3: tx -= 0.28
                    ax.text(tx, ty, text, color=text_color, ha="center", va="center", fontsize=8, fontweight='bold', fontfamily='sans-serif')
                    
    # Draw thicker black exterior rectangle
    rect = Rectangle((0, 0), GRID_SIZE, GRID_SIZE, fill=False, edgecolor='#111111', linewidth=4)
    ax.add_patch(rect)

    fig.suptitle(title, fontsize=16, fontweight='bold', fontfamily='serif', color="#3e2a14", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.9]) # type: ignore
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

