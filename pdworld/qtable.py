from __future__ import annotations

import numpy as np

from pdworld.constants import ACTION_TO_INDEX, ACTIONS
from pdworld.state_mapping import NUM_RL_STATES
from pdworld.types import Action


class QTable:
    def __init__(self, num_states: int = NUM_RL_STATES, num_actions: int = len(ACTIONS)) -> None:
        self.values = np.zeros((num_states, num_actions), dtype=float)

    def get(self, state_id: int, action: Action) -> float:
        return float(self.values[state_id, ACTION_TO_INDEX[action]])

    def set(self, state_id: int, action: Action, value: float) -> None:
        self.values[state_id, ACTION_TO_INDEX[action]] = value

    def max_value(self, state_id: int, applicable_actions: list[Action]) -> float:
        if not applicable_actions:
            return 0.0
        indices = [ACTION_TO_INDEX[action] for action in applicable_actions]
        return float(np.max(self.values[state_id, indices]))

    def best_action(self, state_id: int, applicable_actions: list[Action], rng: np.random.Generator) -> Action:
        if not applicable_actions:
            raise ValueError("No applicable actions available")

        max_val = self.max_value(state_id, applicable_actions)
        best = [action for action in applicable_actions if self.get(state_id, action) == max_val]
        choice = rng.choice(np.array(best, dtype=object))
        return choice.item() if hasattr(choice, "item") else choice

    def snapshot(self) -> np.ndarray:
        return self.values.copy()
