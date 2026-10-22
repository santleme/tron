"""Generate reproducible policy/value distillation examples for search.

All states and transitions come from the canonical arena engine.  The compact
bitboard is used only by the fixed-depth teacher to evaluate a canonical state.
This module deliberately knows nothing about Kaggle submission or protected
evaluation sets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Iterator, Literal

from training.arena.engine import State, initialize_episode, observe, step
from training.arena.heuristics import HEURISTIC_STYLES, heuristic_action

from .bitboard import action_mask, from_arena_state
from .planners import fixed_depth_pair

PROTECTED_SEED_RANGES = (
    (0, 31),
    (1_000_000, 1_000_255),
    (2_000_000, 2_999_999),
    (3_000_000, 3_999_999),
    (4_000_000, 4_000_255),
    (5_000_000, 5_000_271),
)


@dataclass(frozen=True, slots=True)
class DistillConfig:
    """Bounded inputs for teacher-data generation."""

    seeds: tuple[int, ...]
    width: int = 12
    height: int = 12
    edge_budget: int = 56
    max_depth: int = 3
    heuristic_styles: tuple[str, ...] = HEURISTIC_STYLES

    def __post_init__(self) -> None:
        if (not self.seeds or any(type(seed) is not int or not 0 <= seed < (1 << 53)
                                  for seed in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("seeds must be a non-empty sequence of unique integers in [0, 2^53)")
        if any(start <= seed <= end for seed in self.seeds
               for start, end in PROTECTED_SEED_RANGES):
            raise ValueError("seed list overlaps protected G008 development/evaluation root ranges")
        if (type(self.width) is not int or type(self.height) is not int
                or self.width < 2 or self.height < 2):
            raise ValueError("board dimensions must be integers of at least two")
        if (type(self.max_depth) is not int or not 1 <= self.max_depth <= 255
                or type(self.edge_budget) is not int
                or not self.max_depth + 1 <= self.edge_budget <= 0xFFFFFFFF):
            raise ValueError("max_depth must fit the search ABI and edge_budget its u32 counter")
        if (not self.heuristic_styles
                or any(style not in HEURISTIC_STYLES for style in self.heuristic_styles)):
            raise ValueError("heuristic_styles must use the permitted development heuristics")


def _seed_for(seed: int, tick: int, seat: int) -> int:
    raw = f"circuit-ai-distill-v1\0{seed}\0{tick}\0{seat}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _iter_seed_examples(config: DistillConfig, game_index: int, seed: int) -> Iterator[dict[str, Any]]:
    style = config.heuristic_styles[game_index % len(config.heuristic_styles)]
    teacher_seat = game_index & 1
    state, _ = initialize_episode(config.width, config.height, seed, 0)
    tick_limit = config.width * config.height + 1
    while state.result is None:
        compact = from_arena_state(state)
        teacher_results = {}
        legal_seats = tuple(seat for seat in (0, 1) if action_mask(compact, seat))
        if legal_seats:
            paired = fixed_depth_pair(
                compact, max_depth=config.max_depth,
                edge_budget=config.edge_budget,
                seed0=_seed_for(seed, state.tick, 0),
                seed1=_seed_for(seed, state.tick, 1),
            )
            teacher_results.update((seat, paired[seat]) for seat in legal_seats)
        for seat in legal_seats:
            yield _record_with_result(state, seed, style, seat, teacher_results[seat],
                                      config.edge_budget, teacher_seat)
        actions: list[int | str] = ["INVALID", "INVALID"]
        actions[teacher_seat] = (teacher_results[teacher_seat].action
                                 if teacher_seat in teacher_results else "INVALID")
        actions[1 - teacher_seat] = heuristic_action(state, 1 - teacher_seat, style)
        state = step(state, actions).state
        if state.tick > tick_limit:
            raise RuntimeError("canonical episode exceeded the board occupancy bound")


def _seed_example_batch(task: tuple[DistillConfig, int, int]) -> list[dict[str, Any]]:
    config, game_index, seed = task
    return list(_iter_seed_examples(config, game_index, seed))


def iter_examples(config: DistillConfig, *, workers: int = 1) -> Iterator[dict[str, Any]]:
    """Yield ordered seat-relative labels; independent seed games may use processes.

    A bounded window of at most ``workers`` seed results is in flight. Every
    worker uses the canonical simulator and paired fixed-depth teacher.
    """
    if not isinstance(config, DistillConfig):
        raise TypeError("config must be DistillConfig")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive; runtime parallelism is clamped to available CPUs")
    workers = min(workers, os.cpu_count() or 1)
    if workers == 1:
        for game_index, seed in enumerate(config.seeds):
            yield from _iter_seed_examples(config, game_index, seed)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(config.seeds), workers):
            batch = [(config, index, config.seeds[index])
                     for index in range(start, min(start + workers, len(config.seeds)))]
            for examples in pool.map(_seed_example_batch, batch, chunksize=1):
                yield from examples


def _record_with_result(state: State, seed: int, style: str, seat: int, result: Any,
                        edge_budget: int,
                        trajectory_teacher_seat: int) -> dict[str, Any]:
    compact = from_arena_state(state)
    mask = action_mask(compact, seat)
    policy = tuple(float(value) for value in result.policy)
    if (mask == 0 or len(policy) != 4
            or any(not math.isfinite(value) or value < -1e-8 for value in policy)
            or any(policy[action] > 1e-8 and not mask & (1 << action) for action in range(4))
            or not math.isclose(sum(policy), 1.0, rel_tol=1e-7, abs_tol=1e-7)):
        raise RuntimeError("fixed-depth teacher produced an invalid policy target")
    return {
        "observation": observe(state, seat),
        "policy": list(policy),
        "value": float(max(-1.0, min(1.0, result.value))),
        "legal_mask": [int(bool(mask & (1 << action))) for action in range(4)],
        "seed": seed,
        "tick": state.tick,
        "seat": seat,
        "opponent": (f"heuristic:{style}" if seat == trajectory_teacher_seat else "fixed-depth"),
        "trajectory_teacher_seat": trajectory_teacher_seat,
        "teacher": "fixed-depth",
        "teacher_depth": result.completed_depth,
        "teacher_edge_budget": edge_budget,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _source_provenance() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    sources = ("training/arena/engine.py", "training/arena/heuristics.py",
               "training/arena/seeds.py", "training/search/bitboard.py",
               "training/search/matrix.py", "training/search/planners.py",
               "training/search/territory.py", "training/search/native/territory.py",
               "training/search/native/territory.rs", "training/search/distill.py")
    return {name: _sha256(root / name) for name in sources}


def _jsonl_records(records: Iterable[dict[str, Any]], destination: Path) -> int:
    count = 0
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")) + "\n")
            count += 1
        output.flush()
        os.fsync(output.fileno())
    return count


def _npz_records(records: Iterable[dict[str, Any]], destination: Path,
                 width: int, height: int, max_samples: int) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NPZ output requires numpy; choose format='jsonl' otherwise") from exc
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    fields = {
        "observations": (np.uint8, (height, width, 12)),
        "policies": (np.float32, (4,)),
        "values": (np.float32, ()),
        "legal_masks": (np.uint8, (4,)),
        "seeds": (np.int64, ()),
        "ticks": (np.int32, ()),
        "seats": (np.int8, ()),
        "trajectory_teacher_seats": (np.int8, ()),
        "teacher_depths": (np.int8, ()),
        "teacher_edge_budgets": (np.int16, ()),
        "opponents": ("U32", ()),
        "teachers": ("U16", ()),
    }
    with TemporaryDirectory(prefix="circuit-distill-npz-") as temporary:
        arrays = {
            name: np.lib.format.open_memmap(
                Path(temporary) / f"{name}.npy", mode="w+", dtype=dtype,
                shape=(max_samples, *shape))
            for name, (dtype, shape) in fields.items()
        }
        count = 0
        for row in records:
            if count >= max_samples:
                raise RuntimeError("sample count exceeded conservative episode bound")
            observation = np.asarray(row["observation"], dtype=np.uint8)
            if observation.shape != (height, width, 12):
                raise RuntimeError("canonical observations have unexpected channel dimensions")
            arrays["observations"][count] = observation
            for name, key in (("policies", "policy"), ("values", "value"),
                              ("legal_masks", "legal_mask"), ("seeds", "seed"),
                              ("ticks", "tick"), ("seats", "seat"),
                              ("trajectory_teacher_seats", "trajectory_teacher_seat"),
                              ("teacher_depths", "teacher_depth"),
                              ("teacher_edge_budgets", "teacher_edge_budget"),
                              ("opponents", "opponent"), ("teachers", "teacher")):
                arrays[name][count] = row[key]
            count += 1
        if not count:
            raise ValueError("refusing to write an empty distillation dataset")
        for array in arrays.values():
            array.flush()
        # Write standard .npy members directly from disk-backed arrays so data
        # generation does not retain the full observation set in RAM.
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, array in arrays.items():
                with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, array[:count], allow_pickle=False)
    return count


def write_dataset(config: DistillConfig, output_dir: str | Path, *,
                  format: Literal["jsonl", "npz"] = "jsonl",
                  experiment_id: str | None = None, workers: int = 1) -> dict[str, Any]:
    """Generate examples and write the dataset plus a hash-pinned manifest.

    The target directory must not already exist; this prevents accidental
    overwrite of artifacts from another exploratory run.
    """
    if format not in ("jsonl", "npz"):
        raise ValueError("format must be 'jsonl' or 'npz'")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    workers = min(workers, os.cpu_count() or 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_hash = hashlib.sha256(_canonical(asdict(config))).hexdigest()
    experiment_id = experiment_id or f"distill-{stamp}-{config_hash[:8]}"
    if not experiment_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in experiment_id):
        raise ValueError("experiment_id must contain only letters, digits, dot, underscore, or dash")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    data_path = destination / ("examples.jsonl" if format == "jsonl" else "examples.npz")
    count = (_jsonl_records(iter_examples(config, workers=workers), data_path) if format == "jsonl" else
             _npz_records(iter_examples(config, workers=workers), data_path, config.width, config.height,
                          len(config.seeds) * 2 * (config.width * config.height + 1)))
    if count == 0:
        raise ValueError("refusing to write an empty distillation dataset")
    manifest = {
        "format": "circuit-ai-distillation-dataset/1",
        "sample_schema": {
            "id": "circuit-ai-policy-value-sample/1",
            "observation": {"shape": [config.height, config.width, 12],
                            "dtype": "uint8", "perspective": "sample seat"},
            "policy": {"shape": [4], "dtype": "float32",
                       "meaning": "fixed-depth mixed strategy, mass only on legal actions"},
            "value": {"shape": [], "dtype": "float32", "perspective": "sample seat",
                      "range": [-1.0, 1.0]},
            "legal_mask": {"shape": [4], "dtype": "uint8", "values": [0, 1]},
        },
        "experiment_id": experiment_id,
        "run_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "exploratory search-policy distillation",
        "protection_scope": "development seeds only; no official/final/reserved split data accessed",
        "config": asdict(config),
        "config_sha256": config_hash,
        "teacher": {"algorithm": "fixed-depth", "max_depth": config.max_depth,
                    "edge_budget": config.edge_budget},
        "generation_workers": workers,
        "opponents": [f"heuristic:{style}" for style in config.heuristic_styles],
        "sample_count": count,
        "data": {"file": data_path.name, "format": format, "sha256": _sha256(data_path)},
        "source_sha256": _source_provenance(),
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    manifest_path = destination / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return manifest

