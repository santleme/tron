"""Frozen paired G008 principal throughput measurement.

This harness compares the exact pre-optimization incumbent against the frozen
current candidate on matched schedules; the report applies the approved parity
and paired-gain selection policy without recording strategic outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import MISSING, fields
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
from statistics import median
import sys
from typing import Any

from training.eval.provenance import (ProvenanceError, benchmark_run_keys,
                                      build_benchmark_run_schedules,
                                      schedule_spec_digest, seed_plan_sha256,
                                      validate_expanded_schedule)
from training.ppo.protocol import file_sha256
import training.ppo.train as candidate_train_module
from training.ppo.train import TrainConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "g008-principal-benchmark.json"
CONFIG_FORMAT = "circuit-ai-principal-benchmark/1"
MEASURED_REPEATS = 5
WARMUP_REPEATS = 2
WARMUP_UPDATES = 1
WARMUP_BACKENDS = ("python-oracle", "low-risk", "numpy-control", "numba-candidate")
FROZEN_PRINCIPAL_CONFIG_SHA256 = "0ec65b7978e26ecb13b893e76ec358d48c41428db96b0a60dd810f9f24f0e484"
REPORT_FORMAT = "circuit-ai-g008-principal-benchmark/3"
INCUMBENT_DIR = Path(__file__).resolve().parent / "g008_incumbent"
INCUMBENT_MANIFEST = INCUMBENT_DIR / "manifest.json"
INCUMBENT_SOURCE_COMMIT = "3ee627a95a584f0556454fcd18910c2fb0d77c17"
INCUMBENT_LOCAL_PATCH_ID = "g008-incumbent-seat-observation-expansion-v1"
CANDIDATE_SOURCE_COMMIT = "4849bb95d255af186225c1006f29bc66fb832427"
INCUMBENT_TRAIN_SHA256 = "18eca4fe98d635646e51dd0e25d229aa64d2563ff7ad177881b89e7459b3f4e1"
INCUMBENT_ENGINE_SHA256 = "a93dc0c8001325dbb55bb90da5cd9ae4e262948de9b27be01a195299fd8be25b"
CANDIDATE_TRAIN_SHA256 = "9991dc8e0c45cf586d2597f82f1cc30ceaf62c6754b976bd39eca70a3f86c38c"
CANDIDATE_ENGINE_SHA256 = "cfd6fe1817e06e5e892306f1fac2b5eef3cdeb4e68da2e83b26c8f4f23a6c7d0"
SELECTION_MIN_MEDIAN_GAIN = 0.10
SELECTION_MIN_PAIR_GAINS = 4
IMPLEMENTATION_VARIANT_IDS = {
    "incumbent": "incumbent",
    "candidate": "candidate",
}


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


def implementation_hashes(role: str) -> dict[str, str]:
    """Return the exact trainer/engine bytes selected for one benchmark arm."""
    if role == "candidate":
        actual = {
            "trainer_sha256": file_sha256(Path(candidate_train_module.__file__)),
            "engine_sha256": file_sha256(Path(__import__("training.arena.engine", fromlist=["step"]).__file__)),
        }
        expected = {"trainer_sha256": CANDIDATE_TRAIN_SHA256,
                    "engine_sha256": CANDIDATE_ENGINE_SHA256}
        if actual != expected:
            raise ValueError("candidate trainer/engine source digest differs from the frozen optimization candidate")
        return actual
    if role != "incumbent":
        raise ValueError("implementation role must be incumbent or candidate")
    manifest = json.loads(INCUMBENT_MANIFEST.read_text(encoding="utf-8"))
    if manifest != {"format": "circuit-ai-g008-incumbent-snapshot/1",
                    "source_commit": INCUMBENT_SOURCE_COMMIT,
                    "local_patch_id": INCUMBENT_LOCAL_PATCH_ID,
                    "files": {"train.py": INCUMBENT_TRAIN_SHA256,
                              "engine.py": INCUMBENT_ENGINE_SHA256}}:
        raise ValueError("incumbent snapshot manifest differs from the frozen source identity")
    actual = {"trainer_sha256": file_sha256(INCUMBENT_DIR / "train.py"),
              "engine_sha256": file_sha256(INCUMBENT_DIR / "engine.py")}
    expected = {"trainer_sha256": INCUMBENT_TRAIN_SHA256,
                "engine_sha256": INCUMBENT_ENGINE_SHA256}
    if actual != expected:
        raise ValueError("incumbent source snapshot digest mismatch")
    return actual


@lru_cache(maxsize=1)
def _load_incumbent_train_module():
    """Load the checked-in historical trainer with its matching historical engine."""
    implementation_hashes("incumbent")
    # The frozen engine's relative `.seeds` import is intentionally resolved
    # against the current arena package without changing the archived bytes.
    engine_name = "training.arena._g008_incumbent_engine"
    train_name = "training.ppo._g008_incumbent_train"
    engine_spec = importlib.util.spec_from_file_location(engine_name, INCUMBENT_DIR / "engine.py")
    if engine_spec is None or engine_spec.loader is None:
        raise RuntimeError("cannot load frozen incumbent engine source")
    engine_module = importlib.util.module_from_spec(engine_spec)
    sys.modules[engine_name] = engine_module
    engine_spec.loader.exec_module(engine_module)
    trainer_spec = importlib.util.spec_from_file_location(train_name, INCUMBENT_DIR / "train.py")
    if trainer_spec is None or trainer_spec.loader is None:
        raise RuntimeError("cannot load frozen incumbent trainer source")
    trainer_module = importlib.util.module_from_spec(trainer_spec)
    sys.modules[train_name] = trainer_module
    trainer_spec.loader.exec_module(trainer_module)
    # The snapshot's only altered production imports are the two engine entry
    # points. Bind them directly to the separately hash-verified engine bytes.
    trainer_module.initialize_episode = engine_module.initialize_episode
    trainer_module.step = engine_module.step
    return trainer_module


def select_implementation(role: str):
    """Select a known implementation by role and bind its byte-level identity."""
    hashes = implementation_hashes(role)
    module = candidate_train_module if role == "candidate" else _load_incumbent_train_module()
    source_commit = INCUMBENT_SOURCE_COMMIT if role == "incumbent" else CANDIDATE_SOURCE_COMMIT
    return module.train, {"role": role, "variant_id": IMPLEMENTATION_VARIANT_IDS[role],
                          "source_commit": source_commit, **hashes}


def select_winning_implementation(report: dict[str, Any]) -> str:
    """Apply the frozen parity and paired-throughput selection rule."""
    repeats = report.get("pairs")
    parity = report.get("parity")
    if not isinstance(repeats, list) or len(repeats) != MEASURED_REPEATS:
        raise ValueError("selection requires exactly five measured pairs")
    if not isinstance(parity, dict) or parity.get("passed") is not True:
        return "incumbent"
    ratios = [entry.get("candidate_speedup_ratio") for entry in repeats]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("selection requires positive numeric candidate/incumbent throughput ratios")
    gain_count = sum(value >= 1 + SELECTION_MIN_MEDIAN_GAIN for value in ratios)
    return ("candidate" if median(ratios) >= 1 + SELECTION_MIN_MEDIAN_GAIN
            and gain_count >= SELECTION_MIN_PAIR_GAINS else "incumbent")


def _reset_arm_rngs(seed: int) -> None:
    """Put Python, NumPy, Torch CPU, and all CUDA generators at one arm origin."""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed % (1 << 32))
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_document(state) -> dict[str, Any]:
    return {"width": state.width, "height": state.height, "tick": state.tick,
            "result": state.result,
            "agents": [{"position": list(agent.position), "heading": agent.heading,
                        "trail": [list(cell) for cell in agent.trail], "alive": agent.alive}
                       for agent in state.agents]}


def _stream_seed_document(seed) -> dict[str, Any]:
    """Convert the immutable seed record to the JSON form used in parity traces."""
    return {"label": seed.label, "initstate": seed.initstate, "initseq": seed.initseq,
            "digest_hex": seed.digest_hex, "preimage_hex": seed.preimage_hex}


def _transition_parity_digest(module, config: TrainConfig, schedule: dict[str, Any],
                              run_id: str, role: str) -> dict[str, Any]:
    """Hash a real model-sampled collector rollout before any timing run."""
    torch = module._torch()
    device = torch.device(config.device)
    _reset_arm_rngs(config.seed)
    model = module._model_class(torch)(config.width, config.height, config.hidden_size).to(device)
    seeded = module.G008EnvironmentSeedSchedule(
        schedule, run_id=run_id, arena_count=config.arena_count,
        total_updates=config.total_updates, rollout_steps=config.rollout_steps,
        benchmark_role=role)
    events = []
    initialized = 0
    transition_index = 0
    original_initialize = module.initialize_episode
    original_step = module.step

    def capture_initialize(width, height, master_seed, arena_index=0):
        nonlocal initialized
        state, seeds = original_initialize(width, height, master_seed, arena_index)
        event = "initial" if initialized < config.arena_count else "reset"
        events.append({"event": event, "seed": master_seed, "arena_index": arena_index,
                       "state": _state_document(state),
                       "stream_seeds": [_stream_seed_document(seed) for seed in seeds]})
        initialized += 1
        return state, seeds

    def capture_step(state, actions, **kwargs):
        nonlocal transition_index
        transition = original_step(state, actions, **kwargs)
        arena = transition_index % config.arena_count
        step_no = transition_index // config.arena_count
        learner_seat = arena % 2
        reward = 0.001 if state.agents[learner_seat].alive else 0.0
        done = transition.state.result is not None
        if done:
            reward = module.terminal_rewards(transition.state.result)[learner_seat]
        events.append({"event": "transition", "index": transition_index,
                       "step": step_no, "arena": arena,
                       "initial_state": _state_document(state), "actions": list(actions),
                       "transition_state": _state_document(transition.state),
                       "causes": transition.causes, "destinations": transition.destinations,
                       "reward": reward, "done": done})
        transition_index += 1
        return transition

    module.initialize_episode = capture_initialize
    module.step = capture_step
    try:
        module._collect(torch, model, None, config, config.seed, device, 0,
                        seeded, None)
    finally:
        module.initialize_episode = original_initialize
        module.step = original_step
    return {"sha256": _canonical_sha256(events), "transition_count": transition_index,
            "reset_count": sum(item["event"] == "reset" for item in events),
            "action_source": "g008-trainer-model-sampled-collector/1"}


def compute_transition_parity(config: TrainConfig, schedule: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Prove each trainer arm emits the same model-sampled collector trace."""
    incumbent_module = _load_incumbent_train_module()
    candidate_module = candidate_train_module
    # Validate identity before running either arm's parity workload.
    hashes = {"incumbent": implementation_hashes("incumbent"),
              "candidate": implementation_hashes("candidate")}
    incumbent = _transition_parity_digest(incumbent_module, config, schedule, run_id, "incumbent")
    candidate = _transition_parity_digest(candidate_module, config, schedule, run_id, "candidate")
    return {"passed": incumbent["sha256"] == candidate["sha256"],
            "incumbent": incumbent, "candidate": candidate,
            "implementation_hashes": hashes}


def _assert_paired_environment_seeds(config: TrainConfig, schedule: dict[str, Any], run_id: str) -> None:
    from training.ppo.train import G008EnvironmentSeedSchedule

    adapters = {role: G008EnvironmentSeedSchedule(
        schedule, run_id=run_id, arena_count=config.arena_count,
        total_updates=config.total_updates, rollout_steps=config.rollout_steps,
        benchmark_role=role) for role in ("incumbent", "candidate")}
    for arena in range(config.arena_count):
        if adapters["incumbent"].benchmark_initial_seed(arena) != adapters["candidate"].benchmark_initial_seed(arena):
            raise ProvenanceError("paired backend roles do not share identical initial environment seeds")
        for update in range(config.total_updates):
            for ordinal in range(1, config.rollout_steps + 1):
                incumbent = adapters["incumbent"].benchmark_reset_seed(
                    update=update, arena=arena, reset_ordinal=ordinal)
                candidate = adapters["candidate"].benchmark_reset_seed(
                    update=update, arena=arena, reset_ordinal=ordinal)
                if incumbent != candidate:
                    raise ProvenanceError("paired backend roles do not share identical reset environment seeds")


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
    expected_spec_digest = schedule_spec_digest()
    for key in keys:
        schedule = schedules[key]
        validate_expanded_schedule(schedule)
        if (schedule["run_key"] != key or schedule["run_id"] != run_ids[key]
                or schedule["schedule_spec_digest"] != expected_spec_digest):
            raise ProvenanceError("expanded benchmark schedule has a missing or stale run/spec binding")
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
    """Run role-specific warmups and five matched incumbent/candidate pairs."""
    config, source_document = load_principal_config(config_path)
    if run_ids is None:
        raise ValueError("G008 benchmark requires preregistered UUID run_ids for every frozen schedule")
    schedules = prepare_benchmark_schedules(run_ids)
    # Freeze report ancestry from the validated provenance loader and the exact
    # expanded schedule objects consumed by train(); no report-side digest constants.
    keys = benchmark_run_keys()
    frozen_seed_plan_sha = seed_plan_sha256()
    frozen_spec_digest = schedules[keys[0]]["schedule_spec_digest"]
    if frozen_spec_digest != schedule_spec_digest(frozen_seed_plan_sha):
        raise ProvenanceError("benchmark schedules are not bound to the validated seed plan")
    schedule_roster = [{"run_key": key, "run_id": schedules[key]["run_id"],
                        "expanded_schedule_digest": schedules[key]["expanded_schedule_digest"]}
                       for key in keys]
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("G008 benchmark output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    from training.eval.provenance import SEED_PLAN_PATH
    seed_plan = json.loads(SEED_PLAN_PATH.read_text(encoding="utf-8"))
    benchmark_plan = seed_plan["partitions"]["benchmark"]
    measured_plan = benchmark_plan["measured_roots"]
    expected_order = [["incumbent", "candidate"] if index % 2 == 0
                      else ["candidate", "incumbent"] for index in range(MEASURED_REPEATS)]
    if [entry.get("order") for entry in measured_plan] != expected_order:
        raise ProvenanceError("frozen measured pair order differs from the approved alternating schedule")
    warmup_assignments = benchmark_plan["warmup_assignments"]
    frozen_warmup_roles = [(assignment.get("backend_id"), assignment.get("role"))
                           for assignment in warmup_assignments]
    expected_warmup_roles = [(backend, role) for backend in WARMUP_BACKENDS
                             for role in ("incumbent", "candidate")]
    if (frozen_warmup_roles != expected_warmup_roles
            or any(len(assignment.get("root_seeds", [])) != WARMUP_REPEATS
                   for assignment in warmup_assignments)):
        raise ProvenanceError("frozen warmup assignments differ from the exact backend/role roster")
    pair_parity = {}
    for index in range(MEASURED_REPEATS):
        run_key = f"pair-{index:02d}"
        _assert_paired_environment_seeds(config, schedules[run_key], run_ids[run_key])
        pair_parity[run_key] = compute_transition_parity(config, schedules[run_key], run_ids[run_key])
    parity = {"passed": all(item["passed"] for item in pair_parity.values()),
              "pairs": pair_parity,
              "implementation_hashes": pair_parity["pair-00"]["implementation_hashes"]}
    warmup_config = TrainConfig(**{**config.__dict__, "total_updates": WARMUP_UPDATES})
    device_metadata: dict[str, Any] | None = None
    warmups = []
    for backend in WARMUP_BACKENDS:
        for role in ("incumbent", "candidate"):
            for index in range(WARMUP_REPEATS):
                run_key = f"warmup-{backend}-{role}-{index:02d}"
                run_id = run_ids[run_key]
                train_impl, identity = select_implementation(role)
                _reset_arm_rngs(config.seed)
                run_dir = output / "warmups" / backend / role / f"warmup-{index:02d}"
                warmup = train_impl(warmup_config, run_dir, export_onnx=False,
                                    run_evaluation=False, g008_schedule=schedules[run_key],
                                    run_id=run_id)
                _remove_checkpoint_outputs(run_dir, warmup)
                device = _validate_cuda_run(warmup, WARMUP_UPDATES, f"warmup-{role}-{index:02d}")
                if device_metadata is not None and (device_metadata["cuda_name"] != device["cuda_name"]
                                                    or device_metadata["device"] != device["device"]):
                    raise RuntimeError("CUDA device changed during warmups")
                device_metadata = device
                warmups.append({"backend_id": backend, "role": role, "run": index, "run_key": run_key,
                                "run_id": run_id, "implementation": identity, "discarded": True,
                                "expanded_schedule_digest": schedules[run_key]["expanded_schedule_digest"],
                                "training_metrics_file": (Path("warmups") / backend / role /
                                                          f"warmup-{index:02d}" / "training-metrics.json").as_posix(),
                                "training_metrics_sha256": file_sha256(run_dir / "training-metrics.json"),
                                "completed_joint_arena_transitions": config.arena_count * config.rollout_steps})

    pairs = []
    transitions = config.arena_count * config.rollout_steps * config.total_updates
    for index in range(MEASURED_REPEATS):
        run_key = f"pair-{index:02d}"
        arm_metrics = {}
        for role in measured_plan[index]["order"]:
            train_impl, identity = select_implementation(role)
            _reset_arm_rngs(config.seed)
            run_dir = output / "measured" / run_key / role
            metrics = train_impl(config, run_dir, export_onnx=False, run_evaluation=False,
                                 g008_schedule=schedules[run_key], run_id=run_ids[run_key],
                                 g008_backend_role=role)
            _remove_checkpoint_outputs(run_dir, metrics)
            device = _validate_cuda_run(metrics, config.total_updates, f"{run_key}-{role}")
            if device_metadata is None or device["device"] != device_metadata["device"] or device["cuda_name"] != device_metadata["cuda_name"]:
                raise RuntimeError("CUDA device changed between warmups and measured pairs")
            training = metrics["training"]
            update_only_seconds = sum(float(update["update_latency_seconds"]) - float(update["checkpoint_seconds"])
                                      for update in training["updates"])
            durations = {"rollout_seconds": float(training["mean_rollout_seconds"]) * config.total_updates,
                         "optimizer_seconds": sum(float(update["optimizer_seconds"]) for update in training["updates"]),
                         "update_only_seconds": update_only_seconds,
                         "checkpoint_inclusive_update_elapsed_seconds": float(training["update_elapsed_seconds"]),
                         "checkpoint_seconds": sum(float(update["checkpoint_seconds"]) for update in training["updates"]),
                         "wall_seconds": float(training["elapsed_seconds"])}
            if (any(not math.isfinite(value) for value in durations.values())
                    or any(value <= 0 for key, value in durations.items() if key != "checkpoint_seconds")
                    or durations["checkpoint_seconds"] < 0
                    or durations["checkpoint_inclusive_update_elapsed_seconds"] < update_only_seconds):
                raise RuntimeError(f"{run_key}-{role}: invalid or incomplete duration evidence")
            speed = transitions / update_only_seconds
            arm_metrics[role] = {"implementation": identity, "run_key": run_key,
                                 "run_id": run_ids[run_key], "seed": config.seed,
                                 "config_sha256": source_document["config_sha256"],
                                 "expanded_schedule_digest": schedules[run_key]["expanded_schedule_digest"],
                                 "completed_joint_arena_transitions": transitions,
                                 "agent_decisions": transitions * 2,
                                 "update_only_joint_transitions_per_second": speed,
                                 "wall_inclusive_joint_transitions_per_second": transitions / durations["wall_seconds"],
                                 "update_only_agent_decisions_per_second": transitions * 2 / update_only_seconds,
                                 "wall_inclusive_agent_decisions_per_second": transitions * 2 / durations["wall_seconds"],
                                 **durations,
                                 "optimizer_updates": device["optimizer_updates"],
                                 "cuda_optimizer_updates": device["cuda_optimizer_updates"],
                                 "training_metrics_file": (Path("measured") / run_key / role /
                                                           "training-metrics.json").as_posix(),
                                 "training_metrics_sha256": file_sha256(run_dir / "training-metrics.json")}
        if set(arm_metrics) != {"incumbent", "candidate"}:
            raise ProvenanceError(f"{run_key}: measured schedule did not run both frozen implementation roles")
        if (arm_metrics["incumbent"]["run_id"] != arm_metrics["candidate"]["run_id"]
                or arm_metrics["incumbent"]["seed"] != arm_metrics["candidate"]["seed"]
                or arm_metrics["incumbent"]["expanded_schedule_digest"] != arm_metrics["candidate"]["expanded_schedule_digest"]):
            raise ProvenanceError(f"{run_key}: paired roles differ in run ID, config seed, or environment schedule")
        incumbent_rate = arm_metrics["incumbent"]["update_only_joint_transitions_per_second"]
        candidate_rate = arm_metrics["candidate"]["update_only_joint_transitions_per_second"]
        pairs.append({"repeat": index, "run_key": run_key, "run_id": run_ids[run_key],
                      "order": measured_plan[index]["order"], "seed": config.seed,
                      "config_sha256": source_document["config_sha256"],
                      "expanded_schedule_digest": schedules[run_key]["expanded_schedule_digest"],
                      "parity_sha256": pair_parity[run_key]["incumbent"]["sha256"],
                      "candidate_speedup_ratio": candidate_rate / incumbent_rate,
                      "arms": arm_metrics})
    candidate_rates = [pair["arms"]["candidate"]["update_only_joint_transitions_per_second"] for pair in pairs]
    incumbent_rates = [pair["arms"]["incumbent"]["update_only_joint_transitions_per_second"] for pair in pairs]
    ratios = [pair["candidate_speedup_ratio"] for pair in pairs]
    selection = select_winning_implementation({"pairs": pairs, "parity": parity})
    report = {"format": REPORT_FORMAT,
              "purpose": "matched five-pair incumbent/candidate trainer and engine throughput comparison",
              "selection_rule": {"parity_required": True, "median_gain_fraction": SELECTION_MIN_MEDIAN_GAIN,
                                 "minimum_pairs_at_gain": SELECTION_MIN_PAIR_GAINS,
                                 "pair_gain_fraction": SELECTION_MIN_MEDIAN_GAIN},
              "rng_reset_policy": "fresh-model-and-optimizer; reset Python, NumPy, Torch CPU, and all CUDA RNGs to TrainConfig.seed before each arm",
              "selection": selection, "selected_variant_id": IMPLEMENTATION_VARIANT_IDS[selection],
              "implementations": parity["implementation_hashes"],
              "implementation_source_commits": {"incumbent": INCUMBENT_SOURCE_COMMIT,
                                                "candidate": CANDIDATE_SOURCE_COMMIT},
              "source_sha256": _canonical_sha256(parity["implementation_hashes"]),
              "source": source_document["source"], "config": config.__dict__,
              "config_sha256": source_document["config_sha256"], "config_artifact_sha256": file_sha256(config_path),
              "seed_plan_sha256": frozen_seed_plan_sha,
              "schedule_spec_digest": frozen_spec_digest,
              "schedule_roster": schedule_roster,
              "warmups": {"count_per_role": WARMUP_REPEATS, "updates_each": WARMUP_UPDATES,
                          "discarded": True, "runs": warmups},
              "measured_repeats": MEASURED_REPEATS, "updates_per_repeat": config.total_updates,
              "numerator_definition": "completed joint arena transitions = arena_count × rollout_steps × PPO updates; both agents act per joint transition",
              "agent_decisions_definition": "2 × completed joint arena transitions",
              "parity": parity,
              "device": device_metadata,
              "summary": {"candidate_update_only_joint_transitions_per_second": _percentiles(candidate_rates),
                          "incumbent_update_only_joint_transitions_per_second": _percentiles(incumbent_rates),
                          "candidate_speedup_ratio": _percentiles(ratios),
                          **{field: _percentiles([pair["arms"][role][field] for pair in pairs])
                             for role in ("incumbent", "candidate")
                             for field in ("wall_inclusive_joint_transitions_per_second", "update_only_seconds",
                                           "wall_seconds", "optimizer_seconds", "rollout_seconds")}},
              "optimizer_updates_total": sum(pair["arms"][role]["optimizer_updates"]
                                             for pair in pairs for role in ("incumbent", "candidate")),
              "cuda_optimizer_updates_total": sum(pair["arms"][role]["cuda_optimizer_updates"]
                                                  for pair in pairs for role in ("incumbent", "candidate")),
              "pairs": pairs}
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
