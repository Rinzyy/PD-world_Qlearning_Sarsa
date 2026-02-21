from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    PICKUP = "PICKUP"
    DROPOFF = "DROPOFF"


class Policy(str, Enum):
    PRANDOM = "PRANDOM"
    PEXPLOIT = "PEXPLOIT"
    PGREEDY = "PGREEDY"


class LearnerType(str, Enum):
    Q_LEARNING = "Q_LEARNING"
    SARSA = "SARSA"


@dataclass(frozen=True)
class PDWorldState:
    row: int
    col: int
    carrying: bool
    pickup_counts: tuple[int, int, int]
    dropoff_counts: tuple[int, int, int]


@dataclass
class RunConfig:
    alpha: float = 0.3
    gamma: float = 0.5
    total_steps: int = 0
    schedule: list[tuple[int, int, Policy]] = field(default_factory=list)
    seed: int = 7
    learner: LearnerType = LearnerType.Q_LEARNING
    snapshot_steps: dict[str, int] = field(default_factory=dict)
    capture_first_full_dropoff: bool = False
    capture_first_terminal: bool = False
    experiment_id: int = 0
    initial_state: PDWorldState | None = None
    reset_state: PDWorldState | None = None


@dataclass
class RunResult:
    cumulative_rewards: list[float]
    step_rewards: list[int]
    episode_lengths: list[int]
    terminal_hits: int
    q_snapshots: dict[str, object]
    metadata: dict[str, object]
