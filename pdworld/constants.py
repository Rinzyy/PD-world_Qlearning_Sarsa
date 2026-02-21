from __future__ import annotations

from pdworld.types import Action

GRID_SIZE = 5

PICKUP_LOCATIONS: tuple[tuple[int, int], ...] = ((1, 1), (3, 3), (5, 5))
DROPOFF_LOCATIONS: tuple[tuple[int, int], ...] = ((5, 1), (5, 3), (2, 5))

INITIAL_POSITION = (1, 5)
INITIAL_PICKUP_COUNTS = (5, 5, 5)
INITIAL_DROPOFF_COUNTS = (0, 0, 0)

MOVE_REWARD = -1
PICKUP_REWARD = 13
DROPOFF_REWARD = 13

DEFAULT_ALPHA = 0.3
DEFAULT_GAMMA = 0.5
DEFAULT_SEEDS = (7, 19)

ACTIONS: tuple[Action, ...] = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
    Action.PICKUP,
    Action.DROPOFF,
)

MOVE_ACTIONS: tuple[Action, ...] = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
)

ACTION_TO_INDEX: dict[Action, int] = {action: idx for idx, action in enumerate(ACTIONS)}
INDEX_TO_ACTION: dict[int, Action] = {idx: action for action, idx in ACTION_TO_INDEX.items()}
