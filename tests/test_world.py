from pdworld.types import Action, PDWorldState
from pdworld.world import applicable_actions, apply_action, is_terminal, reset_world


def test_reset_world_initial_state() -> None:
    state = reset_world()
    assert (state.row, state.col) == (1, 5)
    assert state.carrying is False
    assert state.pickup_counts == (5, 5, 5)
    assert state.dropoff_counts == (0, 0, 0)


def test_movement_boundaries_and_rewards() -> None:
    state = PDWorldState(row=1, col=1, carrying=False, pickup_counts=(5, 5, 5), dropoff_counts=(0, 0, 0))
    actions = applicable_actions(state)
    assert Action.NORTH not in actions
    assert Action.WEST not in actions
    assert Action.SOUTH in actions
    assert Action.EAST in actions

    next_state, reward, terminal = apply_action(state, Action.SOUTH)
    assert (next_state.row, next_state.col) == (2, 1)
    assert reward == -1
    assert terminal is False


def test_pickup_and_dropoff_mutation() -> None:
    pickup_state = PDWorldState(row=1, col=1, carrying=False, pickup_counts=(2, 5, 5), dropoff_counts=(0, 0, 0))
    assert Action.PICKUP in applicable_actions(pickup_state)

    carrying_state, pickup_reward, _ = apply_action(pickup_state, Action.PICKUP)
    assert carrying_state.carrying is True
    assert carrying_state.pickup_counts == (1, 5, 5)
    assert pickup_reward == 13

    dropoff_state = PDWorldState(row=5, col=1, carrying=True, pickup_counts=(1, 5, 5), dropoff_counts=(3, 0, 0))
    assert Action.DROPOFF in applicable_actions(dropoff_state)

    next_state, drop_reward, _ = apply_action(dropoff_state, Action.DROPOFF)
    assert next_state.carrying is False
    assert next_state.dropoff_counts == (4, 0, 0)
    assert drop_reward == 13


def test_terminal_condition_exactness() -> None:
    non_terminal = PDWorldState(row=2, col=2, carrying=False, pickup_counts=(0, 0, 0), dropoff_counts=(5, 5, 4))
    assert is_terminal(non_terminal) is False

    terminal = PDWorldState(row=2, col=2, carrying=False, pickup_counts=(0, 0, 0), dropoff_counts=(5, 5, 5))
    assert is_terminal(terminal) is True

    carrying_not_terminal = PDWorldState(
        row=2,
        col=2,
        carrying=True,
        pickup_counts=(0, 0, 0),
        dropoff_counts=(5, 5, 5),
    )
    assert is_terminal(carrying_not_terminal) is False
