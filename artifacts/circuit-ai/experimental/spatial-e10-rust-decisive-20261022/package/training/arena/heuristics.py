"""Small, held-out-safe heuristic opponents for G008 training only.

These policies use immediate legality, local mobility, or opponent distance;
none performs the flood-fill reachable-area search reserved for final holdout.
"""
from __future__ import annotations

from training.arena.engine import DELTA

HEURISTIC_STYLES = ("cruise", "local-mobility", "interceptor")


def heuristic_action(state, seat: int, style: str) -> int | str:
    """Choose an action with one of three bounded local heuristics."""
    if seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    if style not in HEURISTIC_STYLES:
        raise ValueError(f"unknown training heuristic style: {style!r}")
    if state.result is not None:
        return "INVALID"
    me, opponent = state.agents[seat], state.agents[1 - seat]
    occupied = {cell for agent in state.agents for cell in agent.trail}
    candidates = []
    for action, (dx, dy) in enumerate(DELTA):
        if action == (me.heading + 2) % 4:
            continue
        destination = (me.position[0] + dx, me.position[1] + dy)
        x, y = destination
        if not (0 <= x < state.width and 0 <= y < state.height) or destination in occupied:
            continue
        forward_rank = (action - me.heading) % 4
        # Straight is preferred for ties, then clockwise, then counterclockwise.
        tie_rank = {0: 2, 1: 1, 3: 0}[forward_rank]
        if style == "cruise":
            score = (int(action == me.heading), tie_rank)
        elif style == "local-mobility":
            # One-ply degree only: count currently free neighbors of the proposed
            # destination. This is deliberately not a flood-fill territory score.
            mobility = sum(
                0 <= x + nx < state.width and 0 <= y + ny < state.height
                and (x + nx, y + ny) not in occupied and (x + nx, y + ny) != destination
                for nx, ny in DELTA
            )
            score = (mobility, tie_rank)
        else:  # interceptor: prefer approach moves while respecting occupied trails.
            distance = abs(x - opponent.position[0]) + abs(y - opponent.position[1])
            score = (-distance, tie_rank)
        candidates.append((score, action))
    return max(candidates)[1] if candidates else "INVALID"
