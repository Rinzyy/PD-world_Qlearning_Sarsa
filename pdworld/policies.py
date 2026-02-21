from __future__ import annotations

import numpy as np

from pdworld.qtable import QTable
from pdworld.types import Action, PDWorldState, Policy
from pdworld.world import applicable_actions


def choose_action(
    policy: Policy,
    world_state: PDWorldState,
    state_id: int,
    q_table: QTable,
    rng: np.random.Generator,
) -> Action:
    actions = applicable_actions(world_state)

    # Spec rule: if pickup/dropoff is applicable, must take it.
    if Action.DROPOFF in actions:
        return Action.DROPOFF
    if Action.PICKUP in actions:
        return Action.PICKUP

    if policy == Policy.PRANDOM:
        return rng.choice(np.array(actions, dtype=object)).item()

    if policy == Policy.PGREEDY:
        return q_table.best_action(state_id, actions, rng)

    if policy == Policy.PEXPLOIT:
        best = q_table.best_action(state_id, actions, rng)
        if len(actions) == 1:
            return best
        if rng.random() < 0.8:
            return best
        others = [action for action in actions if action != best]
        return rng.choice(np.array(others, dtype=object)).item()

    raise ValueError(f"Unknown policy: {policy}")
