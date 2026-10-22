import pytest

from training.arena.engine import Agent, State
from training.search.bitboard import CompactState, from_arena_state, legal_actions
from training.search.matrix import solve_zero_sum
from training.search.planners import decoupled_ucb1_tuned, fixed_depth, fixed_depth_pair, regret_matching


def test_zero_sum_matrix_solver_matching_pennies_and_dominant_action():
    value, row, column = solve_zero_sum(((1, -1), (-1, 1)))
    assert value == pytest.approx(0.0)
    assert row == pytest.approx((0.5, 0.5))
    assert column == pytest.approx((0.5, 0.5))

    value, row, column = solve_zero_sum(((3, 2), (1, 0)))
    assert value == pytest.approx(2.0)
    assert row == pytest.approx((1.0, 0.0))
    assert column == pytest.approx((0.0, 1.0))


def test_zero_sum_matrix_solver_handles_rectangular_and_single_action_games():
    value, row, column = solve_zero_sum(((0, 2, 1), (3, -1, 2)))
    assert sum(row) == pytest.approx(1.0)
    assert sum(column) == pytest.approx(1.0)
    assert all(probability >= 0 for probability in row + column)
    assert value == pytest.approx(1.0)

    value, row, column = solve_zero_sum(((0, 1),))
    assert value == pytest.approx(0.0)
    assert row == (1.0,)
    assert column == pytest.approx((1.0, 0.0))


def test_zero_sum_matrix_solver_rejects_non_finite_or_malformed_input():
    with pytest.raises(ValueError, match="rectangular"):
        solve_zero_sum(((1, 2), (3,)))
    with pytest.raises(ValueError, match="finite"):
        solve_zero_sum(((float("nan"),),))


def _open_state():
    return from_arena_state(State(
        6, 6,
        (Agent((1, 2), 1, ((1, 2),)), Agent((4, 3), 3, ((4, 3),))),
    ))


@pytest.mark.parametrize("planner", (fixed_depth, decoupled_ucb1_tuned, regret_matching))
def test_simultaneous_planners_return_legal_deterministic_bounded_results(planner):
    state = _open_state()
    first = planner(state, player=0, max_depth=3, edge_budget=96, seed=17)
    repeated = planner(state, player=0, max_depth=3, edge_budget=96, seed=17)
    assert first == repeated
    assert first.action in legal_actions(state, 0)
    assert len(first.policy) == 4
    assert sum(first.policy) == pytest.approx(1.0)
    assert -1.0 <= first.value <= 1.0
    assert first.transition_edges <= 96
    assert first.leaf_evaluations >= 0


def test_simultaneous_planners_respect_player_swap_and_seeded_policy_output():
    state = _open_state()
    for planner in (fixed_depth, decoupled_ucb1_tuned, regret_matching):
        left = planner(state, player=0, max_depth=2, edge_budget=64, seed=91)
        right = planner(state, player=1, max_depth=2, edge_budget=64, seed=91)
        assert right.action in legal_actions(state, 1)
        assert sum(right.policy) == pytest.approx(1.0)
        assert -1.0 <= right.value <= 1.0
        assert left.transition_edges <= 64
        assert right.transition_edges <= 64


def test_fixed_depth_pair_reuses_one_search_and_matches_separate_seat_results():
    state = _open_state()
    paired = fixed_depth_pair(state, max_depth=3, edge_budget=96, seed0=91, seed1=91)
    separate = tuple(fixed_depth(state, player=seat, max_depth=3,
                                 edge_budget=96, seed=91)
                     for seat in (0, 1))
    assert paired == separate
    assert paired[0].transition_edges == paired[1].transition_edges
    assert paired[0].leaf_evaluations == paired[1].leaf_evaluations


def test_trapped_agent_is_resolved_by_the_canonical_invalid_action_before_cutoff():
    trapped = Agent((1, 1), 0, ((0, 1), (1, 0), (2, 1), (1, 2), (1, 1)))
    other = Agent((3, 3), 3, ((3, 3),))
    state = from_arena_state(State(4, 4, (trapped, other)))
    assert legal_actions(state, 0) == ()
    for planner in (fixed_depth, decoupled_ucb1_tuned, regret_matching):
        result = planner(state, player=0, max_depth=3, edge_budget=96, seed=5)
        assert result.action == -1
        assert result.value == pytest.approx(-1.0)
        assert result.transition_edges <= 96


def test_mcts_expands_a_separated_root_to_return_a_policy():
    width, height = 6, 3
    occupied = sum(1 << (y * width + 2) for y in range(height))
    heads = (width + 0, width + 4)
    occupied |= (1 << heads[0]) | (1 << heads[1])
    state = CompactState(width, height, heads, (1, 3), occupied)
    for planner in (decoupled_ucb1_tuned, regret_matching):
        result = planner(state, player=0, max_depth=2, edge_budget=16, seed=19)
        assert result.action in legal_actions(state, 0)
        assert result.transition_edges > 0
