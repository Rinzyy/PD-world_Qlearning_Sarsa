from __future__ import annotations

from pdworld.core.constants import GRID_SIZE
from pdworld.core.types import PDWorldState

NUM_RL_STATES = 400


def rl_state(world_state: PDWorldState) -> tuple[int, int, int, int, int, int]:
    x = int(world_state.carrying)
    if x == 0:
        s = int(world_state.pickup_counts[0] > 0)
        t = int(world_state.pickup_counts[1] > 0)
        u = int(world_state.pickup_counts[2] > 0)
    else:
        s = int(world_state.dropoff_counts[0] < 5)
        t = int(world_state.dropoff_counts[1] < 5)
        u = int(world_state.dropoff_counts[2] < 5)
    return (world_state.row, world_state.col, x, s, t, u)


def state_to_id(row: int, col: int, carrying: int | bool, s: int | bool, t: int | bool, u: int | bool) -> int:
    if not (1 <= row <= GRID_SIZE and 1 <= col <= GRID_SIZE):
        raise ValueError(f"Invalid coordinates: ({row}, {col})")
    x = int(carrying)
    if x not in (0, 1):
        raise ValueError(f"Invalid carrying flag: {carrying}")
    
    cell_index = (row - 1) * GRID_SIZE + (col - 1)
    stu_index = int(s) * 4 + int(t) * 2 + int(u)
    
    return (cell_index * 2 + x) * 8 + stu_index


def id_to_state(state_id: int) -> tuple[int, int, int, int, int, int]:
    if not (0 <= state_id < NUM_RL_STATES):
        raise ValueError(f"Invalid state id: {state_id}")
    stu_index = state_id % 8
    rest = state_id // 8
    
    x = rest % 2
    cell_index = rest // 2
    
    u = stu_index % 2
    stu_index //= 2
    t = stu_index % 2
    s = stu_index % 2
    
    row = (cell_index // GRID_SIZE) + 1
    col = (cell_index % GRID_SIZE) + 1
    return row, col, x, s, t, u


def world_state_to_id(world_state: PDWorldState) -> int:
    x = int(world_state.carrying)
    if x == 0:
        s = world_state.pickup_counts[0] > 0
        t = world_state.pickup_counts[1] > 0
        u = world_state.pickup_counts[2] > 0
    else:
        s = world_state.dropoff_counts[0] < 5
        t = world_state.dropoff_counts[1] < 5
        u = world_state.dropoff_counts[2] < 5
    return state_to_id(world_state.row, world_state.col, world_state.carrying, s, t, u)
