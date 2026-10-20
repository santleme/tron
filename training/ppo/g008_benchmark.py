"""Frozen single-baseline G008 principal throughput measurement.

This harness measures the current trainer as-is. It does not compare a candidate,
claim an optimization, or imply that any throughput gate has been met.
"""
from __future__ import annotations

import argparse
from dataclasses import MISSING, fields
from functools import lru_cache
import json
from pathlib import Path
import shutil
from statistics import median
from typing import Any

from training.eval.provenance import benchmark_run_keys, build_benchmark_run_schedules
from training.ppo.protocol import file_sha256
from training.ppo.train import TrainConfig, train

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "g008-principal-benchmark.json"
CONFIG_FORMAT = "circuit-ai-principal-benchmark/1"
MEASURED_REPEATS = 5
WARMUP_REPEATS = 2
WARMUP_UPDATES = 1
FROZEN_PRINCIPAL_CONFIG_SHA256 = "0ec65b7978e26ecb13b893e76ec358d48c41428db96b0a60dd810f9f24f0e484"


def load_principal_config(path: str | Path = CONFIG_PATH) -> tuple[TrainConfig, dict[str, Any]]:
    """Load the frozen principal config and fail closed on shape or digest drift."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"format", "source", "config_sha256", "config"}:
        raise ValueError("principal config must have exactly format, source, config_sha256, and config")
    if document["format"] != CONFIG_FORMAT or not isinstance(document["source"], str) or not document["source"]:
        raise ValueError("unsupported principal config format or missing source")
    if document.get("config_sha256") != FROZEN_PRINCIPAL_CONFIG_SHA256:
        raise ValueError("principal config differs from the frozen G008 baseline digest")
    values = document["config"]
    config_fields = {field.name: field for field in fields(TrainConfig)}
    expected_fields = set(config_fields)
    if not isinstance(values, dict):
        raise ValueError("principal config fields differ from TrainConfig; config must be an object")
    missing, extra = expected_fields - set(values or {}), set(values or {}) - expected_fields
    required_missing = {name for name in missing if config_fields[name].default is MISSING
                        and config_fields[name].default_factory is MISSING}
    if extra or required_missing:
        raise ValueError(f"principal config fields differ from TrainConfig; missing={sorted(missing)}, extra={sorted(extra)}")
    # The frozen config's digest contract is sorted-key, compact JSON for the nested object.
    import hashlib
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    actual_digest = hashlib.sha256(canonical).hexdigest()
    if document["config_sha256"] != actual_digest:
        raise ValueError("principal config SHA-256 mismatch")
    if values["device"] != "cuda":
        raise ValueError("G008 principal benchmark requires device=cuda")
    # Validate every static TrainConfig constraint without probing the local
    # accelerator; the actual train() runs must then prove CUDA execution.
    effective_values = dict(values)
    for name in missing:
        field = config_fields[name]
        effective_values[name] = (field.default_factory() if field.default_factory is not MISSING
                                  else field.default)
    config = TrainConfig(**{**effective_values, "device": "auto"})
    config.validate()
    config.device = "cuda"
    return config, document


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    # The five-run IQR uses the conventional median-of-halves definition.
    q1 = median(ordered[:len(ordered) // 2])
    q3 = median(ordered[(len(ordered) + 1) // 2:])
    return {"median": median(ordered), "min": min(ordered), "max": max(ordered), "iqr": q3 - q1}


def _validate_cuda_run(metrics: dict[str, Any], expected_updates: int, label: str) -> dict[str, Any]:
    device = metrics.get("device")
    if not isinstance(device, dict) or device.get("device") != "cuda" or device.get("cuda_available") is not True:
        raise RuntimeError(f"{label}: run did not prove CUDA execution")
    if not isinstance(device.get("cuda_name"), str) or not device["cuda_name"].strip():
        raise RuntimeError(f"{label}: CUDA device name is missing")
    updates = device.get("optimizer_updates")
    cuda_updates = device.get("cuda_optimizer_updates")
    if type(updates) is not int or updates <= 0 or cuda_updates != updates:
        raise RuntimeError(f"{label}: optimizer updates are not fully evidenced on CUDA")
    training = metrics.get("training")
    logs = training.get("updates") if isinstance(training, dict) else None
    if not isinstance(logs, list) or len(logs) != expected_updates:
        raise RuntimeError(f"{label}: expected exactly {expected_updates} completed PPO updates")
    if sum(item.get("optimizer_updates", 0) for item in logs) != updates:
        raise RuntimeError(f"{label}: optimizer update count disagrees with update logs")
    if any(item.get("device") != "cuda" or item.get("cuda_updates", 0) != item.get("optimizer_updates") for item in logs):
        raise RuntimeError(f"{label}: an optimizer update lacks CUDA evidence")
    return device


def _remove_checkpoint_outputs(run_dir: Path, metrics: dict[str, Any]) -> None:
    """Persist only runtime/performance and CUDA execution evidence."""
    device = metrics.get("device", {})
    training = metrics.get("training", {})
    sanitized = {
        "format": "circuit-ai-g008-benchmark-run-metrics/1",
        "device": {key: device[key] for key in
                   ("device", "cuda_available", "cuda_name", "optimizer_updates", "cuda_optimizer_updates")
                   if key in device},
        "training": {
            key: training[key] for key in ("elapsed_seconds", "update_elapsed_seconds", "mean_rollout_seconds")
            if key in training
        },
    }
    sanitized["training"]["updates"] = [
        {key: update[key] for key in ("device", "cuda_updates", "optimizer_updates", "rollout_seconds",
                                      "optimizer_seconds", "checkpoint_seconds", "update_latency_seconds")
         if key in update}
        for update in training.get("updates", [])
    ]
    metrics.clear()
    metrics.update(sanitized)
    metrics_path = run_dir / "training-metrics.json"
    if metrics_path.exists():
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    checkpoints = run_dir / "checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints)


def prepare_benchmark_schedules(run_ids: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Build and fully validate every frozen root/reset slot before workload."""
    import re
    from training.eval.provenance import ProvenanceError

    keys = benchmark_run_keys()
    if not isinstance(run_ids, dict) or set(run_ids) != set(keys):
        raise ProvenanceError("benchmark requires the exact frozen measured/warmup run-key roster")
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    if any(not isinstance(value, str) or not re.fullmatch(uuid_pattern, value) for value in run_ids.values()):
        raise ProvenanceError("benchmark run IDs must be canonical lowercase RFC 4122 UUIDs")
    if len(set(run_ids.values())) != len(run_ids):
        raise ProvenanceError("benchmark schedule run IDs must be unique")
    # JSON canonicalization sorts object keys, so do not treat mapping insertion
    # order as part of the roster contract. Validate the exact key set above,
    # then pass IDs to the schedule builder in the frozen protocol order.
    schedules = _cached_benchmark_schedules(tuple((key, run_ids[key]) for key in keys))
    if tuple(schedules) != benchmark_run_keys():
        raise ProvenanceError("expanded benchmark schedules differ from the frozen run-key roster")
    return schedules


@lru_cache(maxsize=4)
def _cached_benchmark_schedules(items: tuple[tuple[str, str], ...]) -> dict[str, dict[str, Any]]:
    run_ids = dict(items)
    schedules = build_benchmark_run_schedules(suite_id="g008-benchmark", run_ids=run_ids)
    # Validate each side's disjoint collector coordinates before any workload;
    # measured pair tuples are shared in the full roster and must be resolved
    # explicitly by role, while every warmup root is preallocated and checked.
    from training.ppo.train import G008EnvironmentSeedSchedule
    for key, schedule in schedules.items():
        if key.startswith("pair-"):
            for role in ("incumbent", "candidate"):
                G008EnvironmentSeedSchedule(schedule, run_id=run_ids[key], arena_count=128,
                                            total_updates=3, rollout_steps=32,
                                            benchmark_role=role)
        else:
            G008EnvironmentSeedSchedule(schedule, run_id=run_ids[key], arena_count=128,
                                        total_updates=1, rollout_steps=32)
    return schedules


def run_principal_benchmark(output: str | Path, config_path: str | Path = CONFIG_PATH,
                            run_ids: dict[str, str] | None = None) -> dict[str, Any]:
    """Run 2 discarded one-update rollouts followed by five frozen 3-update runs."""
    config, source_document = load_principal_config(config_path)
    if run_ids is None:
        raise ValueError("G008 benchmark requires preregistered UUID run_ids for every frozen schedule")
    schedules = prepare_benchmark_schedules(run_ids)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("G008 benchmark output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    warmup_config = TrainConfig(**{**config.__dict__, "total_updates": WARMUP_UPDATES})
    device_metadata: dict[str, Any] | None = None
    warmups = []
    for index in range(WARMUP_REPEATS):
        run_key = f"warmup-python-oracle-incumbent-{index:02d}"
        run_id = run_ids[run_key]
        warmup = train(warmup_config, output / "warmups" / f"warmup-{index:02d}",
                       export_onnx=False, run_evaluation=False,
                       g008_schedule=schedules[run_key], run_id=run_id)
        _remove_checkpoint_outputs(output / "warmups" / f"warmup-{index:02d}", warmup)
        device = _validate_cuda_run(warmup, WARMUP_UPDATES, f"warmup-{index:02d}")
        if device_metadata is not None and (device_metadata["cuda_name"] != device["cuda_name"] or device_metadata["device"] != device["device"]):
            raise RuntimeError("CUDA device changed during warmups")
        device_metadata = device
        warmups.append({"run": index, "discarded": True,
                        "schedule_digest": schedules[run_key]["expanded_schedule_digest"],
                        "completed_joint_arena_transitions": config.arena_count * config.rollout_steps})

    measured = []
    transitions = config.arena_count * config.rollout_steps * config.total_updates
    for index in range(MEASURED_REPEATS):
        run_dir = output / "measured" / f"repeat-{index:02d}"
        run_key = f"pair-{index:02d}"
        metrics = train(config, run_dir, export_onnx=False, run_evaluation=False,
                        g008_schedule=schedules[run_key], run_id=run_ids[run_key],
                        g008_backend_role="incumbent")
        _remove_checkpoint_outputs(run_dir, metrics)
        device = _validate_cuda_run(metrics, config.total_updates, f"repeat-{index:02d}")
        if device_metadata is None or device["device"] != device_metadata["device"] or device["cuda_name"] != device_metadata["cuda_name"]:
            raise RuntimeError("CUDA device changed between warmups and measured repeats")
        training = metrics["training"]
        update_only_seconds = sum(float(update["update_latency_seconds"]) - float(update["checkpoint_seconds"])
                                  for update in training["updates"])
        durations = {"rollout_seconds": float(training["mean_rollout_seconds"]) * config.total_updates,
                     "optimizer_seconds": sum(float(update["optimizer_seconds"]) for update in training["updates"]),
                     "update_only_seconds": update_only_seconds,
                     "checkpoint_inclusive_update_elapsed_seconds": float(training["update_elapsed_seconds"]),
                     "checkpoint_seconds": sum(float(update["checkpoint_seconds"]) for update in training["updates"]),
                     "wall_seconds": float(training["elapsed_seconds"])}
        if (any(value <= 0 for key, value in durations.items() if key != "checkpoint_seconds")
                or durations["checkpoint_seconds"] < 0
                or durations["checkpoint_inclusive_update_elapsed_seconds"] < update_only_seconds):
            raise RuntimeError(f"repeat-{index:02d}: invalid or incomplete duration evidence")
        update_seconds = durations["update_only_seconds"]
        wall_seconds = durations["wall_seconds"]
        measured.append({"repeat": index, "completed_joint_arena_transitions": transitions,
                         "agent_decisions": transitions * 2,
                         "update_only_joint_transitions_per_second": transitions / update_seconds,
                         "wall_inclusive_joint_transitions_per_second": transitions / wall_seconds,
                         "update_only_agent_decisions_per_second": transitions * 2 / update_seconds,
                         "wall_inclusive_agent_decisions_per_second": transitions * 2 / wall_seconds,
                         **durations,
                         "optimizer_updates": device["optimizer_updates"],
                         "cuda_optimizer_updates": device["cuda_optimizer_updates"],
                         "schedule_digest": schedules[run_key]["expanded_schedule_digest"],
                         "training_metrics_sha256": file_sha256(run_dir / "training-metrics.json")})
    report = {"format": "circuit-ai-g008-principal-benchmark/1",
              "purpose": "single-baseline measurement; current trainer implementation; no candidate comparison or gate claim",
              "baseline_note": "This measures the current Python trainer as-is. No speedup, optimization, or 10,000 transitions/s gate is implied.",
              "source": source_document["source"], "config": config.__dict__,
              "config_sha256": source_document["config_sha256"], "config_artifact_sha256": file_sha256(config_path),
              "schedule_spec_digest": schedules[benchmark_run_keys()[0]]["schedule_spec_digest"],
              "warmups": {"count": WARMUP_REPEATS, "updates_each": WARMUP_UPDATES, "discarded": True, "runs": warmups},
              "measured_repeats": MEASURED_REPEATS, "updates_per_repeat": config.total_updates,
              "numerator_definition": "completed joint arena transitions = arena_count × rollout_steps × PPO updates; both agents act per joint transition",
              "agent_decisions_definition": "2 × completed joint arena transitions",
              "device": device_metadata,
              "summary": {field: _percentiles([r[field] for r in measured]) for field in
                          ("update_only_joint_transitions_per_second", "wall_inclusive_joint_transitions_per_second",
                           "update_only_agent_decisions_per_second", "wall_inclusive_agent_decisions_per_second",
                           "rollout_seconds", "optimizer_seconds", "update_only_seconds",
                           "checkpoint_inclusive_update_elapsed_seconds", "checkpoint_seconds", "wall_seconds")},
              "optimizer_updates_total": sum(r["optimizer_updates"] for r in measured),
              "cuda_optimizer_updates_total": sum(r["cuda_optimizer_updates"] for r in measured),
              "repeats": measured}
    (output / "g008-principal-benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--run-ids", type=Path, required=True,
                        help="JSON object of every preregistered frozen benchmark run_key to run_id")
    args = parser.parse_args(argv)
    run_ids = json.loads(args.run_ids.read_text(encoding="utf-8"))
    print(json.dumps(run_principal_benchmark(args.output, args.config, run_ids), indent=2))


if __name__ == "__main__":
    main()
