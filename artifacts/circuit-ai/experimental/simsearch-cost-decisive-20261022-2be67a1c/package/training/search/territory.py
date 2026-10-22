"""Deterministic bitset territory, component and articulation analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .bitboard import CompactState, action_mask
from .native.territory import articulation_count as _native_articulation_count


def _expand(cells: int, width: int, height: int) -> int:
    size = width * height
    board = (1 << size) - 1
    first_col = sum(1 << (y * width) for y in range(height))
    last_col = sum(1 << (y * width + width - 1) for y in range(height))
    return ((cells >> width) | ((cells << width) & board) |
            ((cells & ~first_col) >> 1) | (((cells & ~last_col) << 1) & board))


def flood_region(state: CompactState, start: int, *, allow_occupied_start: bool = True) -> int:
    """Return a bitset for cells reachable from ``start`` without crossing trails."""
    if start < 0 or start >= state.width * state.height:
        return 0
    start_bit = 1 << start
    if state.occupied & start_bit and not allow_occupied_start:
        return 0
    free = state.board_mask & ~state.occupied
    seen = start_bit
    frontier = start_bit
    while frontier:
        frontier = _expand(frontier, state.width, state.height) & free & ~seen
        seen |= frontier
    return seen


def components(state: CompactState) -> tuple[int, ...]:
    """Return disjoint connected components of currently empty cells."""
    remaining = state.board_mask & ~state.occupied
    result: list[int] = []
    while remaining:
        seed = remaining & -remaining
        region = flood_region(state, seed.bit_length() - 1, allow_occupied_start=False) & remaining
        result.append(region)
        remaining &= ~region
    return tuple(result)


def _distances(state: CompactState, player: int) -> list[int]:
    size = state.width * state.height
    distance = [-1] * size
    start = state.heads[player]
    other = state.heads[1 - player]
    blocked = state.occupied & ~(1 << start)
    blocked |= 1 << other
    queue = deque([start])
    distance[start] = 0
    while queue:
        cell = queue.popleft()
        x, y = cell % state.width, cell // state.width
        for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if nx < 0 or nx >= state.width or ny < 0 or ny >= state.height:
                continue
            nxt = ny * state.width + nx
            if distance[nxt] >= 0 or blocked & (1 << nxt):
                continue
            distance[nxt] = distance[cell] + 1
            queue.append(nxt)
    return distance


def voronoi(state: CompactState) -> tuple[int, int, int]:
    """Count cells closer to player 0, closer to player 1, and tied/unreachable."""
    d0, d1 = _distances(state, 0), _distances(state, 1)
    owned = [0, 0, 0]
    playable = (state.board_mask & ~state.occupied) | (1 << state.heads[0]) | (1 << state.heads[1])
    for cell, (a, b) in enumerate(zip(d0, d1)):
        if not playable & (1 << cell):
            continue
        if a < 0 and b < 0 or a == b:
            owned[2] += 1
        elif b < 0 or (a >= 0 and a < b):
            owned[0] += 1
        else:
            owned[1] += 1
    return tuple(owned)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ChamberTree:
    """Biconnected chambers and articulation links for one rider's component."""

    chambers: tuple[int, ...]
    articulation_points: tuple[int, ...]
    links: tuple[tuple[int, int], ...]


def chamber_tree(state: CompactState, player: int, *, reachable: int | None = None) -> ChamberTree:
    """Build the block-cut tree of reachable cells, rooted at one agent's head."""
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    start = state.heads[player]
    reachable = flood_region(state, start) if reachable is None else reachable
    other = state.heads[1 - player]
    reachable &= ~(1 << other)
    if not reachable & (1 << start):
        return ChamberTree((), (), ())

    def neighbors(cell: int):
        x, y = cell % state.width, cell // state.width
        for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if 0 <= nx < state.width and 0 <= ny < state.height:
                nxt = ny * state.width + nx
                if reachable & (1 << nxt):
                    yield nxt

    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {start: None}
    children: dict[int, int] = {start: 0}
    articulation: set[int] = set()
    edge_stack: list[tuple[int, int]] = []
    chamber_masks: list[int] = []
    def pop_chamber(stop: tuple[int, int]) -> None:
        mask = 0
        while edge_stack:
            edge = edge_stack.pop()
            mask |= (1 << edge[0]) | (1 << edge[1])
            if edge == stop:
                break
        if mask:
            chamber_masks.append(mask)

    clock = 1
    discovery[start] = low[start] = clock
    # Iterative DFS avoids recursion limits on large open boards.
    stack = [(start, iter(neighbors(start)))]
    while stack:
        cell, adjacent = stack[-1]
        try:
            nxt = next(adjacent)
        except StopIteration:
            stack.pop()
            ancestor = parent[cell]
            if ancestor is None:
                if children[cell] > 1:
                    articulation.add(cell)
                continue
            low[ancestor] = min(low[ancestor], low[cell])
            if low[cell] >= discovery[ancestor]:
                if parent[ancestor] is not None:
                    articulation.add(ancestor)
                pop_chamber((ancestor, cell))
            continue

        if nxt not in discovery:
            parent[nxt] = cell
            children[cell] = children.get(cell, 0) + 1
            children[nxt] = 0
            edge_stack.append((cell, nxt))
            clock += 1
            discovery[nxt] = low[nxt] = clock
            stack.append((nxt, iter(neighbors(nxt))))
        elif nxt != parent[cell] and discovery[nxt] < discovery[cell]:
            edge_stack.append((cell, nxt))
            low[cell] = min(low[cell], discovery[nxt])
    if not chamber_masks:
        chamber_masks.append(1 << start)
    articulation_points = tuple(sorted(articulation))
    links = tuple(
        (index, point)
        for index, mask in enumerate(chamber_masks)
        for point in articulation_points
        if mask & (1 << point)
    )
    return ChamberTree(tuple(chamber_masks), articulation_points, links)


def evaluate(state: CompactState, player: int = 0) -> float:
    """Return a bounded, antisymmetric territorial value from ``player``'s view."""
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    if state.result is not None:
        if state.result == "draw":
            return 0.0
        winner = 0 if state.result == "agent_0_win" else 1
        return 1.0 if player == winner else -1.0
    regions = (flood_region(state, state.heads[0]), flood_region(state, state.heads[1]))
    sizes = (regions[0].bit_count(), regions[1].bit_count())
    if not regions[0] & regions[1]:
        # Once trails disconnect the heads, each rider's survival is bounded
        # by its own component; exact draws remain possible on equal components.
        delta = (sizes[0] - sizes[1]) / max(1, state.width * state.height)
        value = max(-0.95, min(0.95, delta * 2.0))
    else:
        a, b, _ = voronoi(state)
        mobility = (action_mask(state, 0).bit_count() - action_mask(state, 1).bit_count())
        counts = tuple(_native_articulation_count(
            regions[player], state.width, state.height, state.heads[player],
            state.heads[1 - player]) for player in (0, 1))
        if counts[0] is None or counts[1] is None:
            counts = tuple(len(chamber_tree(state, player, reachable=regions[player]).articulation_points)
                           for player in (0, 1))
        articulation = counts[0] - counts[1]
        scale = max(1, state.width * state.height)
        value = ((a - b) / scale + 0.035 * mobility + 0.006 * articulation)
        value = max(-0.95, min(0.95, value))
    return value if player == 0 else -value
