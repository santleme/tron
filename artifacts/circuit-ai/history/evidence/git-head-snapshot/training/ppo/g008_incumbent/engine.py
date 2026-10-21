"""Deterministic, headless Light Cycles v1 game rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

UP, RIGHT, DOWN, LEFT = range(4)
ACTION_ORDER = ("UP", "RIGHT", "DOWN", "LEFT")
DELTA = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True)
class Agent:
    position: tuple[int, int]
    heading: int
    trail: tuple[tuple[int, int], ...]
    alive: bool = True


@dataclass(frozen=True)
class State:
    width: int
    height: int
    agents: tuple[Agent, Agent]
    tick: int = 0
    result: str | None = None

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("board dimensions must be at least 2")
        if len(self.agents) != 2:
            raise ValueError("exactly two agents are required")
        if self.tick < 0:
            raise ValueError("tick cannot be negative")
        for agent in self.agents:
            if agent.heading not in range(4):
                raise ValueError("heading must be in [0, 3]")
            if not agent.trail or agent.position != agent.trail[-1]:
                raise ValueError("trail must be non-empty and end at the head")


@dataclass(frozen=True)
class Transition:
    state: State
    causes: tuple[tuple[str, ...], tuple[str, ...]]
    destinations: tuple[tuple[int, int] | None, tuple[int, int] | None]


def _action(value: object) -> int | None:
    # bool is deliberately rejected despite being an int subclass.
    return value if type(value) is int and 0 <= value < 4 else None


def step(state: State, actions: Sequence[object]) -> Transition:
    """Resolve a simultaneous tick from a single immutable pre-tick snapshot."""
    if len(actions) != 2:
        raise ValueError("exactly two actions are required")
    if state.result is not None:
        return Transition(state, ((), ()), (None, None))

    agents = state.agents
    occupied = {cell for agent in agents for cell in agent.trail}
    normalized = tuple(_action(action) for action in actions)
    destinations: list[tuple[int, int] | None] = [None, None]
    causes: list[list[str]] = [[], []]

    # First compute and classify both proposals without changing state.
    for i, (agent, action) in enumerate(zip(agents, normalized)):
        if not agent.alive:
            causes[i].append("invalid_action")
            continue
        if action is None:
            causes[i].append("invalid_action")
            continue
        if action == (agent.heading + 2) % 4:
            causes[i].append("reverse_action")
            continue
        dx, dy = DELTA[action]
        destination = (agent.position[0] + dx, agent.position[1] + dy)
        destinations[i] = destination
        x, y = destination
        if x < 0 or x >= state.width or y < 0 or y >= state.height:
            causes[i].append("out_of_bounds")
        elif destination in occupied:
            head_cells = {a.position for a in agents}
            causes[i].append("occupied_head" if destination in head_cells else "occupied_trail")

    # Joint collisions are collected after individual tests and before mutation.
    d0, d1 = destinations
    if d0 is not None and d1 is not None and d0 == d1:
        causes[0].append("same_destination")
        causes[1].append("same_destination")
    if d0 is not None and d1 is not None and d0 == agents[1].position and d1 == agents[0].position:
        causes[0].append("head_swap")
        causes[1].append("head_swap")

    dead = [bool(causes[i]) or not agents[i].alive for i in range(2)]
    updated: list[Agent] = []
    for i, agent in enumerate(agents):
        action = normalized[i]
        if not dead[i] and destinations[i] is not None:
            dest = destinations[i]
            updated.append(Agent(dest, action, agent.trail + (dest,), True))
        else:
            updated.append(Agent(agent.position, agent.heading, agent.trail, False))

    if dead[0] and dead[1]:
        result = "draw"
    elif dead[0]:
        result = "agent_1_win"
    elif dead[1]:
        result = "agent_0_win"
    else:
        result = None
    next_state = State(state.width, state.height, tuple(updated), state.tick + 1, result)
    return Transition(next_state, (tuple(causes[0]), tuple(causes[1])), tuple(destinations))


def observe(state: State, agent_index: int) -> list[list[list[int]]]:
    """Return lc-observation/1 as a row-major [height][width][12] tensor."""
    if agent_index not in (0, 1):
        raise ValueError("agent_index must be 0 or 1")
    me, opponent = state.agents[agent_index], state.agents[1 - agent_index]
    result = [[[0] * 12 for _ in range(state.width)] for _ in range(state.height)]
    for x, y in me.trail:
        result[y][x][2] = 1
    for x, y in opponent.trail:
        result[y][x][3] = 1
    x, y = me.position
    result[y][x][0] = 1
    result[y][x][4 + me.heading] = 1
    x, y = opponent.position
    result[y][x][1] = 1
    result[y][x][8 + opponent.heading] = 1
    return result


def choose_action(logits: Sequence[float]) -> int:
    """Deterministic argmax; ties resolve to the lowest action index."""
    if len(logits) != 4:
        raise ValueError("policy must return four action scores")
    import math

    if any(not math.isfinite(float(value)) for value in logits):
        raise ValueError("policy scores must be finite")
    return max(range(4), key=lambda index: logits[index])


def initial_state(width: int, height: int, episode_rng: object,
                  agent_rngs: Sequence[object] | None = None) -> State:
    """Spawn distinct row-major cells; episode stream places heads, agent streams pick headings."""
    cell_count = width * height
    if width < 2 or height < 2 or cell_count < 2:
        raise ValueError("board dimensions must be at least 2")
    draw = getattr(episode_rng, "next_u32", None)
    if draw is None:
        raise TypeError("episode_rng must provide next_u32()")
    if agent_rngs is None:
        # Convenience for deterministic direct tests. Production initialization
        # passes independent derived agent streams (see initialize_episode).
        agent_rngs = (episode_rng, episode_rng)
    if len(agent_rngs) != 2 or any(not hasattr(rng, "next_u32") for rng in agent_rngs):
        raise ValueError("exactly two agent RNG streams are required")
    # Rejection sampling avoids modulo bias and guarantees two distinct heads.
    limit = (1 << 32) - ((1 << 32) % cell_count)

    def index() -> int:
        value = draw()
        while value >= limit:
            value = draw()
        return value % cell_count

    first = index()
    second = index()
    while second == first:
        second = index()
    positions = (first, second)
    agents = []
    for i, flat in enumerate(positions):
        heading = agent_rngs[i].next_bounded(4) if hasattr(agent_rngs[i], "next_bounded") else agent_rngs[i].next_u32() % 4
        pos = (flat % width, flat // width)
        agents.append(Agent(pos, heading, (pos,)))
    return State(width, height, tuple(agents))


def initialize_episode(width: int, height: int, master_seed: int, arena_index: int = 0):
    """Return ``(state, stream seeds)`` for a reproducible independent arena."""
    from .seeds import PCG32, stream_set

    seeds = stream_set(master_seed, arena_index)
    streams = tuple(PCG32.from_seed(seed) for seed in seeds)
    state = initial_state(width, height, streams[0], streams[1:])
    return state, seeds
