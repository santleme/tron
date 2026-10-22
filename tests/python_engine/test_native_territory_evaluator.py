"""Differential tests for the optional native full territory evaluator."""

from __future__ import annotations

import random
import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

from training.arena.engine import Agent, State, step
from training.search.bitboard import CompactState, action_mask, from_arena_state
from training.search import territory
from training.search.native import territory as native_territory


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def native_evaluate_value(tmp_path_factory):
    """Compile and load the real Rust kernel without touching its repo loader."""
    rustc = shutil.which("rustc")
    if not rustc:
        pytest.skip("rustc is unavailable; native differential checks require Rust")
    library_path = tmp_path_factory.mktemp("native-territory") / "libcircuit_territory.so"
    subprocess.run(
        [rustc, "--crate-type=cdylib", "--edition=2021", "-C", "opt-level=3",
         str(ROOT / "training/search/native/territory.rs"), "-o", str(library_path)],
        check=True, capture_output=True, text=True,
    )
    library = ctypes.CDLL(str(library_path))
    patch = pytest.MonkeyPatch()
    patch.setattr(native_territory, "_library", library)
    patch.setattr(native_territory, "_load_attempted", True)
    try:
        yield native_territory.evaluate_value
    finally:
        patch.undo()


def _state(width: int, height: int, heads: tuple[int, int],
           headings: tuple[int, int], occupied_cells=()) -> CompactState:
    occupied = sum(1 << cell for cell in (*heads, *occupied_cells))
    return CompactState(width, height, heads, headings, occupied)


def _edge_cases() -> tuple[CompactState, ...]:
    # Open connected territory with a symmetric, tied Voronoi boundary.
    open_board = _state(7, 7, (8, 40), (1, 3))

    # A complete vertical trail wall separates the riders' reachable regions.
    separated = _state(7, 5, (15, 19), (1, 3),
                       (3, 10, 17, 24, 31))

    # Both riders start at corners, where board bounds remove two moves each.
    boundary = _state(6, 5, (0, 29), (3, 1), (1, 28))

    # Same geometry, different headings: reverse-action restrictions alter
    # the legal mobility term even though occupancy and territory are fixed.
    reverse_restricted = _state(7, 7, (16, 32), (1, 3), (15, 33))

    # The four-cell ring leaves a playable center unreachable from both heads;
    # the open symmetric remainder also has equal-distance Voronoi cells.
    unreachable_and_tied = _state(5, 5, (0, 24), (1, 3),
                                  (7, 11, 13, 17))

    # Scattered trails constrain local movement while leaving both components
    # connected. This covers occupied trail cells as well as open space.
    trails = _state(8, 6, (17, 30), (1, 2), (9, 10, 18, 22, 23, 29, 37))
    return (open_board, separated, boundary, reverse_restricted,
            unreachable_and_tied, trails)


def _generated_canonical_states() -> tuple[CompactState, ...]:
    """Generate reproducible, reachable states through the canonical engine."""
    states = []
    seed_rng = random.Random(0x51A7E)
    for _ in range(128):
        width, height = seed_rng.choice(((5, 5), (7, 6), (8, 8), (9, 7)))
        first = (seed_rng.randrange(1, width - 1), seed_rng.randrange(1, height - 1))
        second = (seed_rng.randrange(width), seed_rng.randrange(height))
        while second == first:
            second = (seed_rng.randrange(width), seed_rng.randrange(height))
        state = State(width, height, (
            Agent(first, seed_rng.randrange(4), (first,)),
            Agent(second, seed_rng.randrange(4), (second,)),
        ))
        rollout_rng = random.Random(seed_rng.randrange(1 << 32))
        for _ in range(rollout_rng.randrange(1, 15)):
            compact = from_arena_state(state)
            choices = [tuple(action for action in range(4)
                             if action_mask(compact, player) & (1 << action))
                       for player in (0, 1)]
            actions = tuple(rollout_rng.choice(options) if options else -1
                            for options in choices)
            state = step(state, actions).state
            if state.result is not None:
                break
        if state.result is None:
            states.append(from_arena_state(state))
    assert len(states) >= 64
    return tuple(states)


def _python_value_without_articulation_acceleration(state: CompactState) -> float:
    # The reference path must use the Python chamber-tree count, not the
    # optional native articulation helper shared by the new full evaluator.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(territory, "_native_evaluate_value", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(territory, "_native_articulation_count",
                            lambda *_args, **_kwargs: None)
        return territory.evaluate(state, player=0)


@pytest.mark.parametrize("state", _edge_cases(), ids=(
    "connected-open", "separated-components", "boundary-mobility",
    "reverse-restrictions", "unreachable-and-tied-voronoi", "trails",
))
def test_native_evaluator_matches_python_on_territory_edge_cases(state, native_evaluate_value):
    expected = _python_value_without_articulation_acceleration(state)
    actual = native_evaluate_value(
        state.occupied, state.width, state.height, state.heads, state.headings)
    assert actual is not None
    assert actual == pytest.approx(expected, abs=1e-12)
    assert territory.evaluate(state, player=0) == pytest.approx(actual, abs=1e-12)
    assert territory.evaluate(state, player=1) == pytest.approx(-actual, abs=1e-12)


def test_reverse_restrictions_change_value_through_legal_mobility(native_evaluate_value):
    geometry = _edge_cases()[3]
    alternate = CompactState(geometry.width, geometry.height, geometry.heads,
                             (3, geometry.headings[1]), geometry.occupied)
    assert action_mask(geometry, 0).bit_count() != action_mask(alternate, 0).bit_count()
    assert _python_value_without_articulation_acceleration(geometry) != pytest.approx(
        _python_value_without_articulation_acceleration(alternate), abs=1e-12)
    assert native_evaluate_value(
        geometry.occupied, geometry.width, geometry.height,
        geometry.heads, geometry.headings) == pytest.approx(
            _python_value_without_articulation_acceleration(geometry), abs=1e-12)
    assert native_evaluate_value(
        alternate.occupied, alternate.width, alternate.height,
        alternate.heads, alternate.headings) == pytest.approx(
            _python_value_without_articulation_acceleration(alternate), abs=1e-12)


@pytest.mark.parametrize("state", _generated_canonical_states(), ids=lambda s: f"seeded-{s.width}x{s.height}-{s.occupied.bit_count()}")
def test_native_evaluator_matches_python_on_fixed_seed_reachable_states(state, native_evaluate_value):
    expected = _python_value_without_articulation_acceleration(state)
    actual = native_evaluate_value(
        state.occupied, state.width, state.height, state.heads, state.headings)
    assert actual is not None
    assert actual == pytest.approx(expected, abs=1e-12)
