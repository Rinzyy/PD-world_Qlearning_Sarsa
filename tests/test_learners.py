from pdworld.learners import q_learning_update, sarsa_update
from pdworld.qtable import QTable
from pdworld.types import Action


def test_q_learning_update_numeric() -> None:
    q = QTable()
    q.set(0, Action.NORTH, 1.0)
    q.set(1, Action.SOUTH, 4.0)

    updated = q_learning_update(
        q_table=q,
        state_id=0,
        action=Action.NORTH,
        reward=2.0,
        next_state_id=1,
        next_applicable_actions=[Action.SOUTH],
        alpha=0.3,
        gamma=0.5,
        terminal=False,
    )

    assert abs(updated - 1.9) < 1e-9


def test_q_learning_terminal_bootstrap_zero() -> None:
    q = QTable()
    q.set(0, Action.NORTH, 1.0)

    updated = q_learning_update(
        q_table=q,
        state_id=0,
        action=Action.NORTH,
        reward=2.0,
        next_state_id=1,
        next_applicable_actions=[Action.SOUTH],
        alpha=0.3,
        gamma=0.5,
        terminal=True,
    )

    assert abs(updated - 1.3) < 1e-9


def test_sarsa_update_numeric() -> None:
    q = QTable()
    q.set(0, Action.NORTH, 1.0)
    q.set(1, Action.EAST, 3.0)

    updated = sarsa_update(
        q_table=q,
        state_id=0,
        action=Action.NORTH,
        reward=2.0,
        next_state_id=1,
        next_action=Action.EAST,
        alpha=0.3,
        gamma=0.5,
        terminal=False,
    )

    assert abs(updated - 1.75) < 1e-9


def test_sarsa_terminal_bootstrap_zero() -> None:
    q = QTable()
    q.set(0, Action.NORTH, 1.0)

    updated = sarsa_update(
        q_table=q,
        state_id=0,
        action=Action.NORTH,
        reward=2.0,
        next_state_id=1,
        next_action=Action.EAST,
        alpha=0.3,
        gamma=0.5,
        terminal=True,
    )

    assert abs(updated - 1.3) < 1e-9
