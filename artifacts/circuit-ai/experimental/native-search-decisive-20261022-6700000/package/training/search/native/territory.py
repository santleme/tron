"""Optional ctypes bridge to the tiny native territory kernel."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

_library = None
_evaluate_function = None
_load_attempted = False


def _load():
    global _library, _evaluate_function, _load_attempted
    if _load_attempted:
        return _library
    _load_attempted = True
    library = Path(__file__).with_name("libcircuit_territory.so")
    if not library.is_file():
        return None
    try:
        loaded = ctypes.CDLL(str(library))
        function = loaded.circuit_articulation_count
        function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
                             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32)]
        function.restype = ctypes.c_int32
    except (OSError, AttributeError):
        return None
    try:
        _evaluate_function = loaded.circuit_evaluate_value
        _evaluate_function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
                                       ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                       ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        _evaluate_function.restype = ctypes.c_double
    except AttributeError:
        _evaluate_function = None
    _library = loaded
    return _library


def articulation_count(reachable: int, width: int, height: int,
                       start: int, other: int) -> int | None:
    """Count articulation vertices, or return ``None`` when native code is absent."""
    library = _load()
    if library is None:
        return None
    size = width * height
    words_count = (size + 63) // 64
    words = (ctypes.c_uint64 * words_count)(
        *((reachable >> (64 * index)) & ((1 << 64) - 1)
          for index in range(words_count)))
    result = ctypes.c_uint32()
    status = library.circuit_articulation_count(
        words, words_count, width, height, start, other, ctypes.byref(result))
    if status != 0:
        raise ValueError("native articulation kernel rejected board arguments")
    return int(result.value)


def evaluate_value(occupied: int, width: int, height: int,
                   heads: tuple[int, int], headings: tuple[int, int]) -> float | None:
    """Evaluate a live compact board in one native call, if the kernel exists.

    The value is from player 0's perspective and follows
    ``training.search.territory.evaluate``. Malformed compact states are
    rejected instead of crossing the FFI boundary.
    """
    global _evaluate_function
    library = _load()
    if library is None:
        return None
    if _evaluate_function is None:
        try:
            _evaluate_function = library.circuit_evaluate_value
            _evaluate_function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
                                           ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                           ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
            _evaluate_function.restype = ctypes.c_double
        except AttributeError:
            return None
    if (type(width) is not int or type(height) is not int or width < 2 or height < 2
            or width * height > 4096):
        raise ValueError("native evaluator requires a board between 4 and 4096 cells")
    size = width * height
    if type(occupied) is not int or occupied < 0 or occupied >> size:
        raise ValueError("occupied bits must fit the board")
    if (len(heads) != 2 or len(headings) != 2
            or any(type(cell) is not int or cell < 0 or cell >= size for cell in heads)
            or heads[0] == heads[1]
            or any(type(heading) is not int or heading not in range(4) for heading in headings)
            or any(not occupied & (1 << cell) for cell in heads)):
        raise ValueError("invalid heads or headings for native evaluator")
    words_count = (size + 63) // 64
    words = (ctypes.c_uint64 * words_count)(
        *((occupied >> (64 * index)) & ((1 << 64) - 1)
          for index in range(words_count)))
    value = _evaluate_function(
        words, words_count, width, height, heads[0], heads[1],
        headings[0], headings[1])
    if not math.isfinite(value):
        raise ValueError("native evaluator rejected board arguments")
    return float(value)


def available() -> bool:
    return _load() is not None

