"""Validated loading boundary for G008 split-manifest consumers.

G007's ``lc-eval/1`` protocol remains independent. G008 train, validation,
and final consumers should obtain rows through this module so the frozen
manifest is validated before any split rows are exposed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from training.eval.provenance import ProvenanceError, validate_manifest


_ALLOWED_USAGES = frozenset({
    "train", "imitation", "reward_design", "benchmark", "validation",
    "checkpoint_selection", "final_eval",
})


def load_manifest(path: str | Path, *,
                  expanded_schedules: Mapping[str, Mapping[str, Any]] | None = None,
                  realized_schedules: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Read a manifest only when its per-run expanded/realized schedules validate.

    Missing artifact maps deliberately fail closed: hash-shaped references alone
    are not sufficient to expose manifest rows.
    """
    try:
        with Path(path).open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot load G008 split manifest: {exc}") from exc
    validate_manifest(manifest, expanded_schedules=expanded_schedules,
                      realized_schedules=realized_schedules)
    return manifest


def rows_for_usage(manifest: Mapping[str, Any], usage: str, *,
                   expanded_schedules: Mapping[str, Mapping[str, Any]] | None = None,
                   realized_schedules: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], ...]:
    """Validate schedule ancestry first, then expose rows authorized for usage."""
    if usage not in _ALLOWED_USAGES:
        raise ValueError(f"unknown G008 manifest usage: {usage}")
    validate_manifest(manifest, expanded_schedules=expanded_schedules,
                      realized_schedules=realized_schedules)
    return tuple(dict(row) for row in manifest["rows"] if usage in row["usages"])


def load_selection_rows(path: str | Path, *,
                        expanded_schedules: Mapping[str, Mapping[str, Any]] | None = None,
                        realized_schedules: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], ...]:
    """Load only checkpoint-selection rows from a validated manifest.

    This is the prescribed input boundary for future checkpoint selectors; it
    does not prevent callers from bypassing this module and importing raw data.
    """
    return rows_for_usage(load_manifest(path, expanded_schedules=expanded_schedules,
                                        realized_schedules=realized_schedules),
                          "checkpoint_selection", expanded_schedules=expanded_schedules,
                          realized_schedules=realized_schedules)
