import json
import random
import ctypes
import shutil
import subprocess
from itertools import product
from pathlib import Path

from training.arena.engine import Agent, State, step
from training.search.bitboard import CompactState, action_mask, from_arena_state, transition
from training.search.territory import chamber_tree, components, evaluate, flood_region, voronoi


ROOT = Path(__file__).resolve().parents[2]
ACTION_INDEX = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}


def _fixture_state(case):
    positions = [tuple(position) for position in case["positions"]]
    trails = [[position] for position in positions]
    for x, y in case.get("occupied_cells", []):
        trails[0].insert(0, (x, y))
    agents = tuple(
        Agent(positions[i], ACTION_INDEX[case["headings"][i]], tuple(trails[i]))
        for i in range(2)
    )
    return State(case["width"], case["height"], agents)


def _occupied_from_state(state):
    occupied = 0
    for agent in state.agents:
        for x, y in agent.trail:
            occupied |= 1 << (y * state.width + x)
    return occupied


def test_compact_transition_matches_canonical_collision_fixtures():
    vectors = json.loads((ROOT / "shared/fixtures/collision-transitions-1.json").read_text())
    for case in vectors["cases"]:
        original = _fixture_state(case)
        compact = from_arena_state(original)
        action_pairs = set(product((-1, 0, 1, 2, 3, 4), repeat=2))
        action_pairs.add(tuple(case["actions"]))
        for actions in action_pairs:
            expected = step(original, actions)
            actual, causes, destinations = transition(compact, actions)
            assert [list(items) for items in causes] == [list(items) for items in expected.causes], (case["id"], actions)
            assert actual.result == expected.state.result, (case["id"], actions)
            assert actual.heads == tuple(a.position[1] * original.width + a.position[0] for a in expected.state.agents)
            assert actual.headings == tuple(a.heading for a in expected.state.agents)
            assert actual.alive == tuple(a.alive for a in expected.state.agents)
            assert actual.occupied == _occupied_from_state(expected.state)
            assert destinations == expected.destinations


def test_compact_action_masks_match_individual_canonical_legality():
    vectors = json.loads((ROOT / "shared/fixtures/collision-transitions-1.json").read_text())
    deltas = ((0, -1), (1, 0), (0, 1), (-1, 0))
    for case in vectors["cases"]:
        state = from_arena_state(_fixture_state(case))
        for player in (0, 1):
            expected = 0
            head = state.heads[player]
            x, y = head % state.width, head // state.width
            for action, (dx, dy) in enumerate(deltas):
                nx, ny = x + dx, y + dy
                if action == (state.headings[player] + 2) % 4:
                    continue
                if not (0 <= nx < state.width and 0 <= ny < state.height):
                    continue
                destination = ny * state.width + nx
                if state.occupied & (1 << destination):
                    continue
                expected |= 1 << action
            assert action_mask(state, player) == expected, case["id"]


def test_compact_terminal_state_is_idempotent():
    state = State(4, 4, (Agent((0, 0), 0, ((0, 0),)), Agent((3, 3), 3, ((3, 3),))))
    first, _, _ = transition(from_arena_state(state), (0, 3))
    assert first.result == "agent_1_win"
    following, causes, destinations = transition(first, (1, 3))
    assert following == first
    assert causes == ((), ())
    assert destinations == (None, None)


def test_off_board_proposals_keep_coordinates_instead_of_aliasing_other_rows():
    state = State(4, 4, (Agent((0, 2), 3, ((0, 2),)), Agent((3, 3), 3, ((3, 3),))))
    compact = from_arena_state(state)
    expected = step(state, (3, 3))
    actual, causes, destinations = transition(compact, (3, 3))
    assert destinations == expected.destinations == ((-1, 2), (2, 3))
    assert causes == expected.causes == (("out_of_bounds",), ())
    assert actual.result == expected.state.result == "agent_1_win"


def test_bitset_flood_fill_components_and_voronoi_are_consistent():
    width = height = 4
    # Vertical occupied wall separates the left and right empty components.
    occupied = sum(1 << (y * width + 1) for y in range(height)) | (1 << 0) | (1 << 15)
    state = from_arena_state(State(
        width, height,
        (Agent((0, 0), 1, ((0, 0),)), Agent((3, 3), 3, ((3, 3),))),
    ))
    state = type(state)(state.width, state.height, state.heads, state.headings, occupied)
    regions = components(state)
    assert len(regions) == 2
    # A complete barrier produces two disjoint reachable regions from the heads.
    left = flood_region(state, state.heads[0])
    right = flood_region(state, state.heads[1])
    assert not left & right
    own0, own1, _ = voronoi(state)
    assert own0 + own1 <= width * height
    assert -1.0 <= evaluate(state, 0) <= 1.0
    assert evaluate(state, 0) == -evaluate(state, 1)


def test_chamber_tree_finds_articulation_chain_and_block_links():
    width, height = 5, 3
    corridor = sum(1 << (width + x) for x in range(width))
    heads = (width, width + width - 1)
    occupied = ((1 << (width * height)) - 1) & ~corridor
    occupied |= (1 << heads[0]) | (1 << heads[1])
    state = from_arena_state(State(
        width, height,
        (Agent((0, 1), 1, ((0, 1),)), Agent((4, 1), 3, ((4, 1),))),
    ))
    state = type(state)(state.width, state.height, state.heads, state.headings, occupied)
    tree = chamber_tree(state, 0)
    assert tree.articulation_points == (6, 7)
    assert len(tree.chambers) == 3
    assert len(tree.links) == 4
    assert chamber_tree(state, 0, reachable=flood_region(state, state.heads[0])) == tree


def test_rust_articulation_count_matches_python_on_seeded_boards(tmp_path):
    rustc = shutil.which("rustc")
    if not rustc:
        import pytest
        pytest.skip("rustc is unavailable")
    source = ROOT / "training/search/native/territory.rs"
    library_path = tmp_path / "libcircuit_territory.so"
    subprocess.run([rustc, "--crate-type=cdylib", "--edition=2021", "-C", "opt-level=3",
                    str(source), "-o", str(library_path)], check=True, capture_output=True)
    library = ctypes.CDLL(str(library_path))
    count = library.circuit_articulation_count
    count.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
                      ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                      ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32)]
    count.restype = ctypes.c_int32

    rng = random.Random(0xA471C)
    for width, height in ((5, 3), (8, 8), (13, 11), (17, 13)):
        size = width * height
        for _ in range(40):
            start = rng.randrange(size)
            other = (start + rng.randrange(1, size)) % size
            reachable = sum(1 << cell for cell in range(size) if rng.random() < 0.64)
            reachable |= 1 << start | 1 << other
            # Build a structurally valid compact state; Python and Rust both
            # receive the same precomputed reachable mask.
            occupied = ((1 << size) - 1) & ~reachable
            occupied |= 1 << start | 1 << other
            state = CompactState(width, height, (start, other), (1, 3), occupied)
            expected = len(chamber_tree(state, 0, reachable=reachable).articulation_points)
            words_count = (size + 63) // 64
            words = (ctypes.c_uint64 * words_count)(
                *((reachable >> (64 * index)) & ((1 << 64) - 1)
                  for index in range(words_count)))
            actual = ctypes.c_uint32()
            assert count(words, words_count, width, height, start, other,
                         ctypes.byref(actual)) == 0
            assert actual.value == expected, (width, height, start, other, reachable)


def test_exact_transposition_key_includes_all_rule_relevant_fields():
    state = from_arena_state(State(4, 4, (Agent((0, 0), 1, ((0, 0),)), Agent((3, 3), 3, ((3, 3),)))))
    assert state.key != type(state)(4, 4, state.heads, (2, 3), state.occupied).key
    assert state.key != type(state)(4, 4, state.heads, state.headings, state.occupied | (1 << 5)).key
    assert state.key != type(state)(5, 4, (0, 19), state.headings, state.occupied | (1 << 19)).key


def test_compact_transition_matches_canonical_on_seeded_reachable_states():
    rng = random.Random(0xC1AC017)
    actions = (-1, 0, 1, 2, 3, 4)
    checked = 0
    for width, height in ((4, 4), (5, 3), (7, 6)):
        for _ in range(12):
            state = State(
                width, height,
                (Agent((1, 1), 1, ((1, 1),)),
                 Agent((width - 2, height - 2), 3, ((width - 2, height - 2),))),
            )
            for _ in range(rng.randrange(4)):
                if state.result is not None:
                    break
                compact = from_arena_state(state)
                legal = [
                    tuple(action for action in range(4) if action_mask(compact, player) & (1 << action))
                    for player in (0, 1)
                ]
                pair = tuple(rng.choice(options) if options else -1 for options in legal)
                state = step(state, pair).state
            if state.result is not None:
                continue
            compact = from_arena_state(state)
            for joint_action in product(actions, repeat=2):
                expected = step(state, joint_action)
                actual, causes, destinations = transition(compact, joint_action)
                assert causes == expected.causes
                assert destinations == expected.destinations
                assert actual.result == expected.state.result
                assert actual.heads == tuple(a.position[1] * width + a.position[0] for a in expected.state.agents)
                assert actual.headings == tuple(a.heading for a in expected.state.agents)
                assert actual.alive == tuple(a.alive for a in expected.state.agents)
                assert actual.occupied == _occupied_from_state(expected.state)
                checked += 1
    assert checked >= 1000
