"""Optional ctypes bridge to the tiny native territory kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path

_library = None
_load_attempted = False


def _load():
    global _library, _load_attempted
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


def available() -> bool:
    return _load() is not None

