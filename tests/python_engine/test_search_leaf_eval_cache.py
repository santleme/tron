"""Regression coverage for fixed-depth leaf evaluation reuse."""

from collections import Counter

import pytest

from training.search.bitboard import CompactState
from training.search import planners


def test_fixed_depth_evaluates_repeated_exact_leaf_state_once(monkeypatch):
    root = CompactState(4, 4, (5, 10), (1, 3), (1 << 5) | (1 << 10))
    shared_leaf = CompactState(4, 4, (4, 11), (0, 2),
                               (1 << 4) | (1 << 11) | (1 << 5) | (1 << 10))
    evaluation_calls = Counter()
    transition_calls = []

    def actions(_state, _player):
        return (0, 1)

    def transition(state, joint_action):
        transition_calls.append((state.key, tuple(joint_action)))
        return shared_leaf, ((), ()), (None, None)

    def evaluate(state, _player):
        evaluation_calls[state.key] += 1
        return -0.25 if state.key == root.key else 0.375

    monkeypatch.setattr(planners, "_actions", actions)
    monkeypatch.setattr(planners, "transition", transition)
    monkeypatch.setattr(planners, "_is_separated", lambda state: state.key == shared_leaf.key)
    monkeypatch.setattr(planners, "_has_no_move", lambda _state: False)
    monkeypatch.setattr(planners, "evaluate", evaluate)

    result = planners.fixed_depth(root, player=0, max_depth=1, edge_budget=4, seed=23)

    assert result.action == 1
    assert result.policy == pytest.approx((0.0, 1.0, 0.0, 0.0))
    assert result.value == pytest.approx(0.375)
    assert result.transition_edges == 4
    assert transition_calls == [
        (root.key, (0, 0)),
        (root.key, (0, 1)),
        (root.key, (1, 0)),
        (root.key, (1, 1)),
    ]
    assert evaluation_calls == Counter({root.key: 1, shared_leaf.key: 1})
