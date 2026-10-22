"""ctypes adapter for the dependency-free Rust simultaneous planner."""

from __future__ import annotations

import ctypes
import random
import struct
from pathlib import Path

from .bitboard import CompactState


_LIBRARY: ctypes.CDLL | None = None


def _library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        path = Path(__file__).parent / "native" / "libcircuit_planner.so"
        lib = ctypes.CDLL(str(path))
        lib.circuit_input_ptr.restype = ctypes.POINTER(ctypes.c_uint8)
        lib.circuit_output_ptr.restype = ctypes.POINTER(ctypes.c_double)
        lib.circuit_plan.restype = ctypes.c_int
        _LIBRARY = lib
    return _LIBRARY


def ensure_loaded() -> None:
    """Load the shared object before a tournament starts its decision clock."""
    _library()


def plan(state: CompactState, *, player: int, max_depth: int, edge_budget: int,
         seed: int) -> tuple[int, int, int, int]:
    """Return sampled legal action, elapsed-independent search counts, and value.

    The shared library uses a single static packet, so each tournament worker
    process may call it sequentially; callers must not share it across threads.
    """
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    packet = bytearray(46)
    packet[:8] = bytes((state.width, state.height, *state.heads, *state.headings,
                        {None: 0, "agent_0_win": 1, "agent_1_win": 2, "draw": 3}[state.result],
                        int(state.alive[0]) | (int(state.alive[1]) << 1)))
    packet[8] = max_depth
    packet[10:14] = struct.pack("<I", edge_budget)
    for cell in range(state.width * state.height):
        if state.occupied & (1 << cell):
            packet[14 + cell // 8] |= 1 << (cell % 8)

    lib = _library()
    input_ptr = lib.circuit_input_ptr()
    for index, value in enumerate(packet):
        input_ptr[index] = value
    status = lib.circuit_plan()
    if status != 0:
        raise RuntimeError(f"Rust planner ABI failed with status {status}")
    output = lib.circuit_output_ptr()
    mask = int(output[9 + player])
    probabilities = [float(output[1 + player * 4 + action]) for action in range(4)]
    legal = [action for action in range(4) if mask & (1 << action)]
    if not legal:
        return -1, int(output[12]), int(output[13]), int(output[14])
    total = sum(probabilities[action] for action in legal)
    if total <= 0:
        probabilities = [1.0 / len(legal) if action in legal else 0.0 for action in range(4)]
    else:
        probabilities = [probability / total if action in legal else 0.0
                          for action, probability in enumerate(probabilities)]
    rng = random.Random(seed + player * 0x9E3779B1)
    point = rng.random()
    action = legal[-1]
    cumulative = 0.0
    for candidate in legal:
        cumulative += probabilities[candidate]
        if point < cumulative:
            action = candidate
            break
    return action, int(output[12]), int(output[13]), int(output[14])
