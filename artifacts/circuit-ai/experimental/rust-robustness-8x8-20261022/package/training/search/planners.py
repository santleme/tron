"""Bounded simultaneous-move planners over the compact arena state."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import log, sqrt
from typing import Literal

from .bitboard import CompactState, legal_actions, transition
from .matrix import solve_zero_sum
from .territory import evaluate, flood_region


@dataclass(frozen=True, slots=True)
class SearchResult:
    action: int
    policy: tuple[float, float, float, float]
    value: float
    transition_edges: int
    leaf_evaluations: int
    completed_depth: int
    simulations: int
    nodes: int


def _actions(state: CompactState, player: int) -> tuple[int, ...]:
    actions = legal_actions(state, player)
    return actions if actions else (-1,)


def _is_separated(state: CompactState) -> bool:
    return not (flood_region(state, state.heads[0]) & flood_region(state, state.heads[1]))


def _has_no_move(state: CompactState) -> bool:
    return not legal_actions(state, 0) or not legal_actions(state, 1)


def _sample_action(actions: tuple[int, ...], probabilities: tuple[float, ...], rng: random.Random) -> int:
    point = rng.random()
    cumulative = 0.0
    for action, probability in zip(actions, probabilities):
        cumulative += probability
        if point < cumulative:
            return action
    return actions[-1]


def _public_policy(actions: tuple[int, ...], probabilities: tuple[float, ...]) -> tuple[float, float, float, float]:
    policy = [0.0] * 4
    for action, probability in zip(actions, probabilities):
        if 0 <= action < 4:
            policy[action] = probability
    return tuple(policy)  # type: ignore[return-value]


def _root_result(state: CompactState, player: int, strategy: tuple[float, ...], actions: tuple[int, ...],
                 value: float, edges: int, leaves: int, depth: int, simulations: int, nodes: int,
                 seed: int) -> SearchResult:
    if not actions or actions == (-1,):
        return SearchResult(-1, (0.0, 0.0, 0.0, 0.0), value, edges, leaves, depth, simulations, nodes)
    rng = random.Random(seed + player * 0x9E3779B1)
    action = _sample_action(actions, strategy, rng)
    public = _public_policy(actions, strategy)
    return SearchResult(action, public, value if player == 0 else -value, edges, leaves, depth, simulations, nodes)


class _BudgetExceeded(Exception):
    pass


def fixed_depth(state: CompactState, *, player: int = 0, max_depth: int = 6,
                edge_budget: int = 4096, seed: int = 0) -> SearchResult:
    """Return one player's result from the shared simultaneous-move search."""
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    return fixed_depth_pair(state, max_depth=max_depth, edge_budget=edge_budget,
                            seed0=seed, seed1=seed)[player]


def fixed_depth_pair(state: CompactState, *, max_depth: int = 6,
                     edge_budget: int = 4096, seed0: int = 0,
                     seed1: int | None = None) -> tuple[SearchResult, SearchResult]:
    """Search a state once and return both seat policies from each payoff matrix.

    ``seed0`` and ``seed1`` only affect the sampled actions. The policy/value
    targets and search work are shared, which is useful when labeling both
    players from one state.
    """
    if max_depth < 1 or edge_budget < 1:
        raise ValueError("max_depth and edge_budget must be positive")
    if seed1 is None:
        seed1 = seed0
    root_actions = (_actions(state, 0), _actions(state, 1))
    if state.result is not None:
        value = evaluate(state, 0)
        return tuple(_root_result(state, player, (), (), value, 0, 0, 0, 0, 0,
                                  seed0 if player == 0 else seed1)
                     for player in (0, 1))  # type: ignore[return-value]
    edges = 0
    leaves = 0
    visited: set[tuple[object, ...]] = set()
    completed_depth = 0
    best_value = evaluate(state, 0)
    best_strategies = tuple(tuple(1.0 / len(actions) for _ in actions)
                            for actions in root_actions)
    cache: dict[tuple[tuple[object, ...], int], float] = {}
    leaf_cache: dict[tuple[object, ...], float] = {state.key: best_value}

    def leaf_value(current: CompactState) -> float:
        nonlocal leaves
        leaves += 1
        key = current.key
        if key not in leaf_cache:
            leaf_cache[key] = evaluate(current, 0)
        return leaf_cache[key]

    def solve(current: CompactState, depth: int) -> float:
        nonlocal edges, leaves
        if current.result is not None:
            return leaf_value(current)
        if depth == 0 or _is_separated(current):
            if _has_no_move(current):
                actions0, actions1 = _actions(current, 0), _actions(current, 1)
                forced_matrix = []
                for action0 in actions0:
                    row = []
                    for action1 in actions1:
                        if edges >= edge_budget:
                            raise _BudgetExceeded
                        edges += 1
                        successor, _, _ = transition(current, (action0, action1))
                        row.append(leaf_value(successor))
                    forced_matrix.append(row)
                value, _, _ = solve_zero_sum(forced_matrix)
                return value
            return leaf_value(current)
        key = (current.key, depth)
        if key in cache:
            return cache[key]
        visited.add(key)
        actions0, actions1 = _actions(current, 0), _actions(current, 1)
        matrix: list[list[float]] = []
        for action0 in actions0:
            row = []
            for action1 in actions1:
                if edges >= edge_budget:
                    raise _BudgetExceeded
                edges += 1
                successor, _, _ = transition(current, (action0, action1))
                row.append(solve(successor, depth - 1))
            matrix.append(row)
        value, _, _ = solve_zero_sum(matrix)
        cache[key] = value
        return value

    for depth in range(1, max_depth + 1):
        before_edges = edges
        try:
            actions0, actions1 = root_actions
            matrix = []
            for action0 in actions0:
                row = []
                for action1 in actions1:
                    if edges >= edge_budget:
                        raise _BudgetExceeded
                    edges += 1
                    successor, _, _ = transition(state, (action0, action1))
                    row.append(solve(successor, depth - 1))
                matrix.append(row)
            value, strategy0, strategy1 = solve_zero_sum(matrix)
            best_value = value
            best_strategies = (strategy0, strategy1)
            completed_depth = depth
        except _BudgetExceeded:
            # An incomplete iteration cannot replace the previous complete one.
            if edges < before_edges:
                raise AssertionError("edge accounting moved backwards")
            break
    return tuple(_root_result(state, player, best_strategies[player], root_actions[player],
                              best_value, edges, leaves, completed_depth, 0,
                              len(visited), seed0 if player == 0 else seed1)
                 for player in (0, 1))  # type: ignore[return-value]


@dataclass(slots=True)
class _Node:
    state: CompactState
    actions: tuple[tuple[int, ...], tuple[int, ...]]
    visits: int = 0
    action_visits: list[list[int]] = field(default_factory=list)
    reward_sum: list[list[float]] = field(default_factory=list)
    reward_square_sum: list[list[float]] = field(default_factory=list)
    regret: list[list[float]] = field(default_factory=list)
    strategy_sum: list[list[float]] = field(default_factory=list)

    @classmethod
    def create(cls, state: CompactState) -> _Node:
        acts = (_actions(state, 0), _actions(state, 1))
        return cls(state, acts, 0,
                   [[0] * len(acts[0]), [0] * len(acts[1])],
                   [[0.0] * len(acts[0]), [0.0] * len(acts[1])],
                   [[0.0] * len(acts[0]), [0.0] * len(acts[1])],
                   [[0.0] * len(acts[0]), [0.0] * len(acts[1])],
                   [[0.0] * len(acts[0]), [0.0] * len(acts[1])])

    def rm_policy(self, player: int) -> tuple[float, ...]:
        regrets = self.regret[player]
        positive = [max(0.0, regret) for regret in regrets]
        total = sum(positive)
        base = ([value / total for value in positive] if total else
                [1.0 / len(positive)] * len(positive))
        uniform = 1.0 / len(base)
        return tuple(0.99 * probability + 0.01 * uniform for probability in base)


def _ucb1_tuned(node: _Node, player: int, rng: random.Random) -> tuple[int, float]:
    counts = node.action_visits[player]
    unvisited = [i for i, count in enumerate(counts) if count == 0]
    if unvisited:
        index = unvisited[rng.randrange(len(unvisited))]
        return index, 1.0 / len(unvisited)
    total = max(1, node.visits)
    indices = []
    for i, count in enumerate(counts):
        mean = node.reward_sum[player][i] / count
        variance = max(0.0, node.reward_square_sum[player][i] / count - mean * mean)
        ln = log(total) if total > 1 else 0.0
        bound = variance + sqrt(2.0 * ln / count)
        index = mean + sqrt((ln / count) * min(0.25, bound))
        indices.append(index)
    best = max(indices)
    candidates = [i for i, value in enumerate(indices) if abs(value - best) <= 1e-12]
    return candidates[rng.randrange(len(candidates))], 1.0 / len(candidates)


def _terminal_or_cutoff(state: CompactState, depth: int, max_depth: int) -> bool:
    if state.result is not None:
        return True
    # A trapped rider still has to submit INVALID; resolve that exact joint
    # transition before applying a non-terminal territory cutoff.
    if _has_no_move(state):
        return False
    # The root still needs one joint action matrix to produce a policy even if
    # the heads are already in separate components. Child states can use the
    # territorial evaluator as soon as separation is observed.
    return depth >= max_depth or (depth > 0 and _is_separated(state))


def _matrix_mcts(state: CompactState, *, player: int, max_depth: int, edge_budget: int, seed: int,
                 algorithm: Literal["ducb1t", "regret_matching"]) -> SearchResult:
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    if state.result is not None:
        return _root_result(state, player, (), (), evaluate(state, player), 0, 0, 0, 0, 0, seed)
    if edge_budget < max_depth + 1:
        raise ValueError("edge_budget must allow a depth-limited path and a forced no-move transition")

    rngs = (random.Random(seed), random.Random(seed ^ 0xD1B54A32D192ED03))
    nodes: dict[tuple[object, ...], _Node] = {state.key: _Node.create(state)}
    edges = leaves = simulations = 0
    while edge_budget - edges >= max_depth + 1:
        current = state
        depth = 0
        path: list[tuple[_Node, tuple[int, int], tuple[tuple[float, ...], tuple[float, ...]]]] = []
        while not _terminal_or_cutoff(current, depth, max_depth):
            node = nodes.get(current.key)
            if node is None:
                node = _Node.create(current)
                nodes[current.key] = node
            if algorithm == "ducb1t":
                selected = (_ucb1_tuned(node, 0, rngs[0]), _ucb1_tuned(node, 1, rngs[1]))
                indices = (selected[0][0], selected[1][0])
                probabilities = (
                    tuple(1.0 if i == indices[0] else 0.0 for i in range(len(node.actions[0]))),
                    tuple(1.0 if i == indices[1] else 0.0 for i in range(len(node.actions[1]))),
                )
            else:
                probabilities = (node.rm_policy(0), node.rm_policy(1))
                indices = (_sample_index(probabilities[0], rngs[0]),
                           _sample_index(probabilities[1], rngs[1]))
                for p in (0, 1):
                    for i, probability in enumerate(probabilities[p]):
                        node.strategy_sum[p][i] += probability
            action_pair = (node.actions[0][indices[0]], node.actions[1][indices[1]])
            path.append((node, indices, probabilities))
            current, _, _ = transition(current, action_pair)
            edges += 1
            depth += 1
        value = evaluate(current, 0)
        leaves += 1
        for node, selected, probabilities in reversed(path):
            node.visits += 1
            for p in (0, 1):
                utility = value if p == 0 else -value
                normalized = (utility + 1.0) * 0.5
                index = selected[p]
                node.action_visits[p][index] += 1
                node.reward_sum[p][index] += normalized
                node.reward_square_sum[p][index] += normalized * normalized
                if algorithm == "regret_matching":
                    probability = probabilities[p][index]
                    estimated_selected = utility / probability
                    baseline = probability * estimated_selected
                    for action_index, action_probability in enumerate(probabilities[p]):
                        estimate = estimated_selected if action_index == index else 0.0
                        node.regret[p][action_index] += estimate - baseline
            nodes.setdefault(node.state.key, node)
        simulations += 1

    root = nodes[state.key]
    if algorithm == "ducb1t":
        visits = root.action_visits[player]
        total = sum(visits)
        strategy = tuple(count / total for count in visits) if total else tuple(1.0 / len(visits) for _ in visits)
        player_value = 2.0 * sum(root.reward_sum[player]) / max(1, total) - 1.0
        value = player_value if player == 0 else -player_value
    else:
        strategy_total = sum(root.strategy_sum[player])
        if strategy_total:
            strategy = tuple(value / strategy_total for value in root.strategy_sum[player])
        else:
            strategy = tuple(1.0 / len(root.actions[player]) for _ in root.actions[player])
        visits = sum(root.action_visits[player])
        player_value = 2.0 * sum(root.reward_sum[player]) / max(1, visits) - 1.0
        value = player_value if player == 0 else -player_value
    return _root_result(state, player, strategy, root.actions[player], value, edges, leaves, 0,
                        simulations, len(nodes), seed)


def _sample_index(probabilities: tuple[float, ...], rng: random.Random) -> int:
    point = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if point < cumulative:
            return index
    return len(probabilities) - 1


def decoupled_ucb1_tuned(state: CompactState, *, player: int = 0, max_depth: int = 6,
                         edge_budget: int = 4096, seed: int = 0) -> SearchResult:
    """Stacked simultaneous matrix-MCTS with marginal UCB1-Tuned selection."""
    return _matrix_mcts(state, player=player, max_depth=max_depth, edge_budget=edge_budget,
                        seed=seed, algorithm="ducb1t")


def regret_matching(state: CompactState, *, player: int = 0, max_depth: int = 6,
                    edge_budget: int = 4096, seed: int = 0) -> SearchResult:
    """Stacked simultaneous matrix-MCTS using positive cumulative regrets."""
    return _matrix_mcts(state, player=player, max_depth=max_depth, edge_budget=edge_budget,
                        seed=seed, algorithm="regret_matching")
