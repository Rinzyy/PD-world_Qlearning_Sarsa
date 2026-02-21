import numpy as np

from pdworld.qtable import QTable
from pdworld.state_mapping import world_state_to_id
from pdworld.types import Action, PDWorldState, Policy
from pdworld.policies import choose_action


def test_forced_pickup_overrides_policy() -> None:
    state = PDWorldState(row=1, col=1, carrying=False, pickup_counts=(1, 5, 5), dropoff_counts=(0, 0, 0))
    q = QTable()
    rng = np.random.default_rng(0)

    for policy in (Policy.PRANDOM, Policy.PEXPLOIT, Policy.PGREEDY):
        action = choose_action(policy, state, world_state_to_id(state), q, rng)
        assert action == Action.PICKUP


def test_forced_dropoff_overrides_policy() -> None:
    state = PDWorldState(row=5, col=1, carrying=True, pickup_counts=(0, 0, 0), dropoff_counts=(4, 0, 0))
    q = QTable()
    rng = np.random.default_rng(0)

    for policy in (Policy.PRANDOM, Policy.PEXPLOIT, Policy.PGREEDY):
        action = choose_action(policy, state, world_state_to_id(state), q, rng)
        assert action == Action.DROPOFF


def test_pexploit_explores_non_best_action() -> None:
    state = PDWorldState(row=2, col=2, carrying=False, pickup_counts=(5, 5, 5), dropoff_counts=(0, 0, 0))
    state_id = world_state_to_id(state)
    q = QTable()
    q.set(state_id, Action.NORTH, 10.0)

    rng = np.random.default_rng(2)
    actions = [choose_action(Policy.PEXPLOIT, state, state_id, q, rng) for _ in range(300)]
    assert Action.NORTH in actions
    assert any(action != Action.NORTH for action in actions)


def test_tie_breaking_is_seed_deterministic() -> None:
    state = PDWorldState(row=2, col=2, carrying=False, pickup_counts=(5, 5, 5), dropoff_counts=(0, 0, 0))
    state_id = world_state_to_id(state)
    q = QTable()
    q.set(state_id, Action.NORTH, 5.0)
    q.set(state_id, Action.SOUTH, 5.0)

    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    action_a = choose_action(Policy.PGREEDY, state, state_id, q, rng_a)
    action_b = choose_action(Policy.PGREEDY, state, state_id, q, rng_b)

    assert action_a == action_b
    assert action_a in (Action.NORTH, Action.SOUTH)
