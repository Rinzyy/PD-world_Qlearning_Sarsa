from pathlib import Path

from pdworld.experiments import get_experiment_config
from pdworld.constants import ACTION_TO_INDEX
from pdworld.experiments import run_experiment
from pdworld.runner import policy_for_step, run_steps
from pdworld.state_mapping import world_state_to_id
from pdworld.types import Action, LearnerType, PDWorldState, Policy, RunConfig


def test_policy_schedule_switching() -> None:
    schedule = [(1, 3, Policy.PRANDOM), (4, 6, Policy.PGREEDY)]
    assert policy_for_step(schedule, 1) == Policy.PRANDOM
    assert policy_for_step(schedule, 3) == Policy.PRANDOM
    assert policy_for_step(schedule, 4) == Policy.PGREEDY
    assert policy_for_step(schedule, 6) == Policy.PGREEDY


def test_episode_reset_keeps_q_table() -> None:
    near_terminal = PDWorldState(
        row=2,
        col=5,
        carrying=True,
        pickup_counts=(0, 0, 0),
        dropoff_counts=(5, 5, 4),
    )

    config = RunConfig(
        total_steps=2,
        schedule=[(1, 2, Policy.PRANDOM)],
        learner=LearnerType.Q_LEARNING,
        seed=7,
        snapshot_steps={"final": 2},
        initial_state=near_terminal,
    )

    result = run_steps(config)
    state_id = world_state_to_id(near_terminal)
    q_value = result.q_snapshots["final"][state_id, ACTION_TO_INDEX[Action.DROPOFF]]

    assert result.terminal_hits >= 1
    assert result.episode_lengths[0] == 1
    assert q_value != 0.0


def test_exp2_checkpoint_capture_fires() -> None:
    near_terminal = PDWorldState(
        row=2,
        col=5,
        carrying=True,
        pickup_counts=(0, 0, 0),
        dropoff_counts=(5, 5, 4),
    )

    config = RunConfig(
        total_steps=3,
        schedule=[(1, 3, Policy.PRANDOM)],
        learner=LearnerType.Q_LEARNING,
        seed=11,
        snapshot_steps={"final": 3},
        capture_first_full_dropoff=True,
        capture_first_terminal=True,
        initial_state=near_terminal,
    )

    result = run_steps(config)
    assert "first_full_dropoff" in result.q_snapshots
    assert "first_terminal" in result.q_snapshots
    assert result.metadata["snapshot_steps"]["first_full_dropoff"] == 1
    assert result.metadata["snapshot_steps"]["first_terminal"] == 1


def test_smoke_experiment_output_files(tmp_path: Path) -> None:
    _, run_dir = run_experiment(exp_id=1, seed=7, output_root=tmp_path)
    assert (run_dir / "timeseries.csv").exists()
    assert (run_dir / "episode_lengths.csv").exists()
    assert (run_dir / "q_mid.csv").exists()
    assert (run_dir / "q_final.csv").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "cumulative_reward.png").exists()


def test_required_seed_runs_are_not_identical() -> None:
    for exp_id in (1, 2, 3):
        config_a = get_experiment_config(exp_id=exp_id, seed=7)
        config_b = get_experiment_config(exp_id=exp_id, seed=19)
        result_a = run_steps(config_a)
        result_b = run_steps(config_b)

        assert result_a.step_rewards != result_b.step_rewards
