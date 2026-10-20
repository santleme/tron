from training.arena.engine import DELTA, initialize_episode
from training.arena.heuristics import HEURISTIC_STYLES, heuristic_action


def test_training_heuristics_only_choose_legal_non_reverse_moves():
    for seed in range(32, 64):
        state = initialize_episode(12, 12, seed, seed)[0]
        occupied = {cell for agent in state.agents for cell in agent.trail}
        for seat in (0, 1):
            for style in HEURISTIC_STYLES:
                action = heuristic_action(state, seat, style)
                assert action in range(4)
                agent = state.agents[seat]
                assert action != (agent.heading + 2) % 4
                dx, dy = DELTA[action]
                destination = (agent.position[0] + dx, agent.position[1] + dy)
                assert 0 <= destination[0] < state.width
                assert 0 <= destination[1] < state.height
                assert destination not in occupied


def test_training_heuristics_are_deterministic_and_distinct_styles_exist():
    state = initialize_episode(12, 12, 444, 0)[0]
    assert len(HEURISTIC_STYLES) >= 3
    for style in HEURISTIC_STYLES:
        assert heuristic_action(state, 0, style) == heuristic_action(state, 0, style)
