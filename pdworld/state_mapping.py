from __future__ import annotations

from pdworld.constants import GRID_SIZE
from pdworld.types import PDWorldState

NUM_RL_STATES = GRID_SIZE * GRID_SIZE * 2


def rl_state(world_state: PDWorldState) -> tuple[int, int, int]:
    return (world_state.row, world_state.col, int(world_state.carrying))


def state_to_id(row: int, col: int, carrying: int | bool) -> int:
    if not (1 <= row <= GRID_SIZE and 1 <= col <= GRID_SIZE):
        raise ValueError(f"Invalid coordinates: ({row}, {col})")
    x = int(carrying)
    if x not in (0, 1):
        raise ValueError(f"Invalid carrying flag: {carrying}")
    cell_index = (row - 1) * GRID_SIZE + (col - 1)
    return cell_index * 2 + x


def id_to_state(state_id: int) -> tuple[int, int, int]:
    if not (0 <= state_id < NUM_RL_STATES):
        raise ValueError(f"Invalid state id: {state_id}")
    cell_index, x = divmod(state_id, 2)
    row = (cell_index // GRID_SIZE) + 1
    col = (cell_index % GRID_SIZE) + 1
    return row, col, x


def world_state_to_id(world_state: PDWorldState) -> int:
    return state_to_id(world_state.row, world_state.col, world_state.carrying)
