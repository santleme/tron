from training.arena.engine import step
from training.arena.seeds import derive_seed, PCG32
from training.arena.vector import VectorArena


def test_vectorized_headless_runner_matches_scalar_transitions():
    batch = VectorArena.create(count=12, width=12, height=8, master_seed=123, start_index=7)
    assert len({(state.agents[0].position, state.agents[1].position) for state in batch.states}) > 1
    actions = [(1, 3)] * len(batch.states)
    expected = [step(state, action) for state, action in zip(batch.states, actions)]
    actual = batch.step(actions)
    assert actual == expected
    assert batch.states == [item.state for item in expected]


def test_initialization_uses_episode_for_positions_and_agent_streams_for_headings():
    master, arena = 88, 4
    batch = VectorArena.create(count=1, width=7, height=5, master_seed=master, start_index=arena)
    state = batch.states[0]
    episode, seed0, seed1 = batch.stream_seeds[0]
    positions_rng = PCG32.from_seed(episode)
    headings = [PCG32.from_seed(seed0).next_bounded(4), PCG32.from_seed(seed1).next_bounded(4)]
    count = 35
    limit = (1 << 32) - ((1 << 32) % count)
    def draw_cell():
        value = positions_rng.next_u32()
        while value >= limit:
            value = positions_rng.next_u32()
        return value % count
    cells = [draw_cell()]
    cells.append(draw_cell())
    while cells[1] == cells[0]:
        cells[1] = draw_cell()
    assert [a.position for a in state.agents] == [(i % 7, i // 7) for i in cells]
    assert [a.heading for a in state.agents] == headings
    assert episode == derive_seed(master, "episode", arena)
