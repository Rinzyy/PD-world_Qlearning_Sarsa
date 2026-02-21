from __future__ import annotations

from pdworld.core.qtable import QTable
from pdworld.core.types import Action


def q_learning_update(
    q_table: QTable,
    state_id: int,
    action: Action,
    reward: float,
    next_state_id: int,
    next_applicable_actions: list[Action],
    alpha: float,
    gamma: float,
    terminal: bool,
) -> float:
    old = q_table.get(state_id, action)
    bootstrap = 0.0 if terminal else q_table.max_value(next_state_id, next_applicable_actions)
    target = reward + gamma * bootstrap
    updated = old + alpha * (target - old)
    q_table.set(state_id, action, updated)
    return updated


def sarsa_update(
    q_table: QTable,
    state_id: int,
    action: Action,
    reward: float,
    next_state_id: int,
    next_action: Action | None,
    alpha: float,
    gamma: float,
    terminal: bool,
) -> float:
    old = q_table.get(state_id, action)
    bootstrap = 0.0 if terminal or next_action is None else q_table.get(next_state_id, next_action)
    target = reward + gamma * bootstrap
    updated = old + alpha * (target - old)
    q_table.set(state_id, action, updated)
    return updated
