"""Compact bitset rules kernel matching :mod:`training.arena.engine`.

Cells use row-major bit positions. The occupancy bitset is sufficient for
future transitions; trail ordering is not part of the rules after occupancy
has been established.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from training.arena.engine import Agent, State

UP, RIGHT, DOWN, LEFT = range(4)
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
CAUSES = ("invalid_action", "reverse_action", "out_of_bounds", "occupied_head", "occupied_trail", "same_destination", "head_swap")


@dataclass(frozen=True, slots=True)
class CompactState:
    width: int
    height: int
    heads: tuple[int, int]
    headings: tuple[int, int]
    occupied: int
    alive: tuple[bool, bool] = (True, True)
    result: str | None = None

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("board dimensions must be at least 2")
        size = self.width * self.height
        if any(cell < 0 or cell >= size for cell in self.heads):
            raise ValueError("head outside board")
        if any(heading not in range(4) for heading in self.headings):
            raise ValueError("heading must be in [0, 3]")
        if any(not self.occupied & (1 << cell) for cell in self.heads):
            raise ValueError("every head must be occupied")
        if self.occupied >> size:
            raise ValueError("occupied bit outside board")
        if self.result not in (None, "agent_0_win", "agent_1_win", "draw"):
            raise ValueError("invalid game result")

    @property
    def board_mask(self) -> int:
        return (1 << (self.width * self.height)) - 1

    @property
    def key(self) -> tuple[int, int, tuple[int, int], tuple[int, int], int, tuple[bool, bool], str | None]:
        """Exact transposition identity; tick and trail order do not affect future rules."""
        return self.width, self.height, self.heads, self.headings, self.occupied, self.alive, self.result

    def to_arena_state(self, trails: tuple[tuple[int, ...], tuple[int, ...]]) -> State:
        """Build a canonical diagnostic state from complete per-agent trails."""
        if len(trails) != 2:
            raise ValueError("exactly two trails are required")
        if occupied_from_trails(trails) != self.occupied:
            raise ValueError("diagnostic trails do not match compact occupancy")
        agents = tuple(
            Agent(_xy(self.heads[i], self.width), self.headings[i],
                  tuple(_xy(cell, self.width) for cell in trails[i]), self.alive[i])
            for i in range(2)
        )
        return State(self.width, self.height, agents, result=self.result)


def _xy(cell: int, width: int) -> tuple[int, int]:
    return cell % width, cell // width


def from_arena_state(state: State) -> CompactState:
    occupied = 0
    for agent in state.agents:
        for x, y in agent.trail:
            occupied |= 1 << (y * state.width + x)
    heads = tuple(a.position[1] * state.width + a.position[0] for a in state.agents)
    return CompactState(state.width, state.height, heads,
                        tuple(a.heading for a in state.agents), occupied,
                        tuple(a.alive for a in state.agents), state.result)


def _destination(state: CompactState, head: int, action: int) -> tuple[int, int]:
    x, y = _xy(head, state.width)
    dx, dy = DELTAS[action]
    x, y = x + dx, y + dy
    return x, y


def action_mask(state: CompactState, player: int) -> int:
    """Bit i is set iff action i passes the individual canonical legality checks.

    Joint destination and head-swap collisions remain joint-action properties
    and are intentionally not hidden by a per-player mask.
    """
    if player not in (0, 1) or state.result is not None or not state.alive[player]:
        return 0
    mask = 0
    for action in range(4):
        if action == (state.headings[player] + 2) % 4:
            continue
        x, y = _destination(state, state.heads[player], action)
        if x < 0 or x >= state.width or y < 0 or y >= state.height:
            continue
        dest = y * state.width + x
        if state.occupied & (1 << dest):
            continue
        mask |= 1 << action
    return mask


def legal_actions(state: CompactState, player: int) -> tuple[int, ...]:
    mask = action_mask(state, player)
    return tuple(action for action in range(4) if mask & (1 << action))


def transition(state: CompactState, actions: Sequence[object]):
    """Resolve both proposals simultaneously with canonical cause ordering.

    Returns ``(next_state, causes, destinations)``. Destinations are ``(x, y)``
    coordinates, including out-of-bounds proposals; ``None`` means the action
    was invalid or a reversal.
    """
    if len(actions) != 2:
        raise ValueError("exactly two actions are required")
    if state.result is not None:
        return state, ((), ()), (None, None)

    causes: list[list[str]] = [[], []]
    destinations: list[tuple[int, int] | None] = [None, None]
    for i, (heading, head) in enumerate(zip(state.headings, state.heads)):
        action = actions[i]
        if not state.alive[i] or type(action) is not int or action not in range(4):
            causes[i].append("invalid_action")
            continue
        if action == (heading + 2) % 4:
            causes[i].append("reverse_action")
            continue
        x, y = _destination(state, head, action)
        destinations[i] = (x, y)
        if x < 0 or x >= state.width or y < 0 or y >= state.height:
            causes[i].append("out_of_bounds")
        else:
            dest = y * state.width + x
            if not state.occupied & (1 << dest):
                continue
            if dest in state.heads:
                causes[i].append("occupied_head")
            else:
                causes[i].append("occupied_trail")

    d0, d1 = destinations
    head_positions = tuple(_xy(head, state.width) for head in state.heads)
    if d0 is not None and d1 is not None and d0 == d1:
        causes[0].append("same_destination")
        causes[1].append("same_destination")
    if d0 is not None and d1 is not None and d0 == head_positions[1] and d1 == head_positions[0]:
        causes[0].append("head_swap")
        causes[1].append("head_swap")

    dead = tuple(bool(causes[i]) or not state.alive[i] for i in range(2))
    heads = list(state.heads)
    headings = list(state.headings)
    occupied = state.occupied
    for i, dest in enumerate(destinations):
        if not dead[i] and dest is not None:
            x, y = dest
            heads[i] = y * state.width + x
            headings[i] = actions[i]
            occupied |= 1 << heads[i]
    if dead[0] and dead[1]:
        result = "draw"
    elif dead[0]:
        result = "agent_1_win"
    elif dead[1]:
        result = "agent_0_win"
    else:
        result = None
    next_state = CompactState(state.width, state.height, tuple(heads), tuple(headings), occupied,  # type: ignore[arg-type]
                              tuple(not value for value in dead), result)  # type: ignore[arg-type]
    return next_state, (tuple(causes[0]), tuple(causes[1])), tuple(destinations)


def occupied_from_trails(trails: Sequence[Sequence[int]]) -> int:
    """Utility for fixtures and data adapters."""
    occupied = 0
    for trail in trails:
        for cell in trail:
            occupied |= 1 << cell
    return occupied
