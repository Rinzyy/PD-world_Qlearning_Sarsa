from pdworld.state_mapping import NUM_RL_STATES, id_to_state, state_to_id


def test_unique_state_ids_count() -> None:
    ids = {state_to_id(row, col, carrying) for row in range(1, 6) for col in range(1, 6) for carrying in (0, 1)}
    assert len(ids) == NUM_RL_STATES


def test_roundtrip_state_mapping() -> None:
    for row in range(1, 6):
        for col in range(1, 6):
            for carrying in (0, 1):
                state_id = state_to_id(row, col, carrying)
                decoded = id_to_state(state_id)
                assert decoded == (row, col, carrying)
