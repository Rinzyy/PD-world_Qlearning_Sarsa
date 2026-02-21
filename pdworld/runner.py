from __future__ import annotations

from dataclasses import asdict

import numpy as np

from pdworld.learners import q_learning_update, sarsa_update
from pdworld.policies import choose_action
from pdworld.qtable import QTable
from pdworld.state_mapping import world_state_to_id
from pdworld.types import LearnerType, Policy, RunConfig, RunResult
from pdworld.world import applicable_actions, reset_world


def policy_for_step(schedule: list[tuple[int, int, Policy]], step: int) -> Policy:
    for start, end, policy in schedule:
        if start <= step <= end:
            return policy
    raise ValueError(f"No policy defined for step {step}")


def run_steps(config: RunConfig) -> RunResult:
    rng = np.random.default_rng(config.seed)
    q_table = QTable()

    if config.initial_state is not None:
        world_state = config.initial_state
    else:
        world_state = reset_world()

    if config.reset_state is not None:
        reset_target = config.reset_state
    else:
        reset_target = reset_world()

    step_rewards: list[int] = []
    cumulative_rewards: list[float] = []
    episode_lengths: list[int] = []
    q_snapshots: dict[str, np.ndarray] = {}
    snapshot_steps_taken: dict[str, int] = {}

    cumulative_bank = 0.0
    terminal_hits = 0
    current_episode_length = 0

    captured_first_full_dropoff = False
    captured_first_terminal = False

    for step in range(1, config.total_steps + 1):
        policy = policy_for_step(config.schedule, step)
        state_id = world_state_to_id(world_state)
        action = choose_action(policy, world_state, state_id, q_table, rng)

        next_world, reward, terminal = (
            __import__("pdworld.world", fromlist=["apply_action"]).apply_action(world_state, action)
        )
        next_state_id = world_state_to_id(next_world)

        if config.learner == LearnerType.Q_LEARNING:
            q_learning_update(
                q_table=q_table,
                state_id=state_id,
                action=action,
                reward=reward,
                next_state_id=next_state_id,
                next_applicable_actions=applicable_actions(next_world),
                alpha=config.alpha,
                gamma=config.gamma,
                terminal=terminal,
            )
        elif config.learner == LearnerType.SARSA:
            if terminal:
                next_action = None
            else:
                next_policy = policy_for_step(config.schedule, min(step + 1, config.total_steps))
                next_action = choose_action(next_policy, next_world, next_state_id, q_table, rng)

            sarsa_update(
                q_table=q_table,
                state_id=state_id,
                action=action,
                reward=reward,
                next_state_id=next_state_id,
                next_action=next_action,
                alpha=config.alpha,
                gamma=config.gamma,
                terminal=terminal,
            )
        else:
            raise ValueError(f"Unsupported learner: {config.learner}")

        current_episode_length += 1
        cumulative_bank += reward
        step_rewards.append(reward)
        cumulative_rewards.append(cumulative_bank)

        for snapshot_name, snapshot_step in config.snapshot_steps.items():
            if step == snapshot_step and snapshot_name not in q_snapshots:
                q_snapshots[snapshot_name] = q_table.snapshot()
                snapshot_steps_taken[snapshot_name] = step

        if config.capture_first_full_dropoff and (not captured_first_full_dropoff):
            if any(count == 5 for count in next_world.dropoff_counts):
                q_snapshots["first_full_dropoff"] = q_table.snapshot()
                snapshot_steps_taken["first_full_dropoff"] = step
                captured_first_full_dropoff = True

        if terminal:
            terminal_hits += 1
            episode_lengths.append(current_episode_length)
            current_episode_length = 0

            if config.capture_first_terminal and (not captured_first_terminal):
                q_snapshots["first_terminal"] = q_table.snapshot()
                snapshot_steps_taken["first_terminal"] = step
                captured_first_terminal = True

            world_state = reset_target
        else:
            world_state = next_world

    if "final" not in q_snapshots:
        q_snapshots["final"] = q_table.snapshot()
        snapshot_steps_taken["final"] = config.total_steps

    metadata = {
        "config": {
            "alpha": config.alpha,
            "gamma": config.gamma,
            "total_steps": config.total_steps,
            "seed": config.seed,
            "learner": config.learner.value,
            "schedule": [(s, e, p.value) for s, e, p in config.schedule],
            "experiment_id": config.experiment_id,
        },
        "terminal_hits": terminal_hits,
        "episodes_completed": len(episode_lengths),
        "snapshot_steps": snapshot_steps_taken,
        "episode_in_progress_length": current_episode_length,
    }

    return RunResult(
        cumulative_rewards=cumulative_rewards,
        step_rewards=step_rewards,
        episode_lengths=episode_lengths,
        terminal_hits=terminal_hits,
        q_snapshots=q_snapshots,
        metadata=metadata,
    )
