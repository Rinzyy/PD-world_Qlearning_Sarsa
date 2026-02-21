from __future__ import annotations

from pdworld.core.constants import (
    DROPOFF_LOCATIONS,
    DROPOFF_REWARD,
    GRID_SIZE,
    INITIAL_DROPOFF_COUNTS,
    INITIAL_PICKUP_COUNTS,
    INITIAL_POSITION,
    MOVE_REWARD,
    PICKUP_LOCATIONS,
    PICKUP_REWARD,
)
from pdworld.core.types import Action, PDWorldState

_PICKUP_INDEX = {loc: idx for idx, loc in enumerate(PICKUP_LOCATIONS)}
_DROPOFF_INDEX = {loc: idx for idx, loc in enumerate(DROPOFF_LOCATIONS)}

_MOVEMENT = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.EAST: (0, 1),
    Action.WEST: (0, -1),
}


def reset_world() -> PDWorldState:
    return PDWorldState(
        row=INITIAL_POSITION[0],
        col=INITIAL_POSITION[1],
        carrying=False,
        pickup_counts=INITIAL_PICKUP_COUNTS,
        dropoff_counts=INITIAL_DROPOFF_COUNTS,
    )


def is_terminal(state: PDWorldState) -> bool:
    return (not state.carrying) and all(count == 5 for count in state.dropoff_counts)


def applicable_actions(state: PDWorldState) -> list[Action]:
    actions: list[Action] = []

    for action, (dr, dc) in _MOVEMENT.items():
        next_row = state.row + dr
        next_col = state.col + dc
        if 1 <= next_row <= GRID_SIZE and 1 <= next_col <= GRID_SIZE:
            actions.append(action)

    pos = (state.row, state.col)
    if not state.carrying and pos in _PICKUP_INDEX:
        pickup_idx = _PICKUP_INDEX[pos]
        if state.pickup_counts[pickup_idx] >= 1:
            actions.append(Action.PICKUP)

    if state.carrying and pos in _DROPOFF_INDEX:
        dropoff_idx = _DROPOFF_INDEX[pos]
        if state.dropoff_counts[dropoff_idx] < 5:
            actions.append(Action.DROPOFF)

    return actions


def apply_action(state: PDWorldState, action: Action) -> tuple[PDWorldState, int, bool]:
    allowed = applicable_actions(state)
    if action not in allowed:
        raise ValueError(f"Action {action} is not applicable in state {state}")

    if action in _MOVEMENT:
        dr, dc = _MOVEMENT[action]
        next_state = PDWorldState(
            row=state.row + dr,
            col=state.col + dc,
            carrying=state.carrying,
            pickup_counts=state.pickup_counts,
            dropoff_counts=state.dropoff_counts,
        )
        return next_state, MOVE_REWARD, is_terminal(next_state)

    if action == Action.PICKUP:
        idx = _PICKUP_INDEX[(state.row, state.col)]
        next_pickups = list(state.pickup_counts)
        next_pickups[idx] -= 1
        next_state = PDWorldState(
            row=state.row,
            col=state.col,
            carrying=True,
            pickup_counts=tuple(next_pickups),
            dropoff_counts=state.dropoff_counts,
        )
        return next_state, PICKUP_REWARD, is_terminal(next_state)

    idx = _DROPOFF_INDEX[(state.row, state.col)]
    next_dropoffs = list(state.dropoff_counts)
    next_dropoffs[idx] += 1
    next_state = PDWorldState(
        row=state.row,
        col=state.col,
        carrying=False,
        pickup_counts=state.pickup_counts,
        dropoff_counts=tuple(next_dropoffs),
    )
    return next_state, DROPOFF_REWARD, is_terminal(next_state)
