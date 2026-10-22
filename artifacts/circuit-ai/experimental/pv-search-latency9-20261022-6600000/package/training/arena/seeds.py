"""Cross-runtime seed derivation and PCG-XSH-RR 64/32 implementation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

MASK64 = (1 << 64) - 1
MULTIPLIER = 6364136223846793005
SEED_PREFIX = b"lc-arena/seed/v1/"


def _uint64(value: int, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MASK64:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


@dataclass(frozen=True)
class StreamSeed:
    label: str
    initstate: int
    initseq: int
    digest_hex: str
    preimage_hex: str


def derive_seed(master_seed: int, label: str, *indices: int) -> StreamSeed:
    """Derive an independent PCG seed using the lc-arena/seed/v1 domain."""
    _uint64(master_seed, "master_seed")
    if label not in ("episode", "agent/0", "agent/1"):
        raise ValueError("unsupported stream label")
    preimage = SEED_PREFIX + label.encode("utf-8") + b"\0"
    preimage += struct.pack("<Q", master_seed)
    for index in indices:
        preimage += struct.pack("<Q", _uint64(index, "stream index"))
    digest = hashlib.sha256(preimage).digest()
    initstate, initseq = struct.unpack("<QQ", digest[:16])
    return StreamSeed(label, initstate, initseq, digest.hex(), preimage.hex())


class PCG32:
    """Reference PCG-XSH-RR 64/32, seeded per the v1 contract."""

    def __init__(self, initstate: int, initseq: int):
        self.state = 0
        self.increment = ((initseq << 1) | 1) & MASK64
        self.next_u32()
        self.state = (self.state + _uint64(initstate, "initstate")) & MASK64
        self.next_u32()

    @classmethod
    def from_seed(cls, seed: StreamSeed) -> "PCG32":
        return cls(seed.initstate, seed.initseq)

    def next_u32(self) -> int:
        old = self.state
        self.state = (old * MULTIPLIER + self.increment) & MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rotation = (old >> 59) & 31
        return ((xorshifted >> rotation) | (xorshifted << ((-rotation) & 31))) & 0xFFFFFFFF

    def next_bounded(self, bound: int) -> int:
        """Uniform integer in [0, bound), using PCG's unbiased threshold method."""
        if type(bound) is not int or bound <= 0 or bound > (1 << 32):
            raise ValueError("bound must be in [1, 2**32]")
        threshold = ((-bound) & 0xFFFFFFFF) % bound
        while True:
            value = self.next_u32()
            if value >= threshold:
                return value % bound


def stream_set(master_seed: int, arena_index: int) -> tuple[StreamSeed, StreamSeed, StreamSeed]:
    _uint64(arena_index, "arena_index")
    return (
        derive_seed(master_seed, "episode", arena_index),
        derive_seed(master_seed, "agent/0", arena_index, 0),
        derive_seed(master_seed, "agent/1", arena_index, 1),
    )
