"""Append-only, hash-chained G008 experiment registry.

This is a local integrity mechanism, not a signature. It detects inconsistent
partial edits while independently captured head hashes remain available.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import statistics
from decimal import Decimal, InvalidOperation
import unicodedata
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
_HASH_CHARS = frozenset("0123456789abcdef")
_REGISTRATION_KINDS = frozenset({"train", "imitation", "ablation", "validation", "final", "benchmark"})
_RECOVERY_EVENT_TYPES = frozenset({
    "recovery_amendment", "workload_preflight", "workload_push_dispatched",
    "workload_push_accepted", "workload_attempt",
})
_LEGACY_HEAD_INDEX = 12
_LEGACY_HEAD_HASH = "0208abba4954ffd08b01d5069c674e8bc3f5d1817c95ddb0b0bb4eedd41f3fd7"
_RECOVERY_PLAN_SHA256 = "0358c3c00ffa438a3bd033939687827774da5f04ba54746c075eb4680b384d72"
_CYCLE2_PROPOSAL_SHA256 = "efd2ea6e9525b2d59b3143fae97110f1eb3a76707ae2e19a7512bc84b94cf776"
_CYCLE1_FINAL_EVENT_INDEX = 19
_CYCLE1_FINAL_EVENT_HASH = "7280fb7471484e856b48d1a931cb18d97d5bd2f94949790c41defef9154c4213"
_CYCLE2_EVENT_INDEX = 20
_CYCLE2_BENCHMARK_GPU_SECONDS = 5600
_CYCLE2_TRAIN_GPU_SECONDS = 7200
_CYCLE2_OUTPUT_CATEGORY_CAP = 1_000_000_000
_CYCLE2_BENCHMARK_ATTEMPT_OUTPUT_CAP = 100_000_000
_CYCLE2_CAMPAIGN_GPU_BEFORE = 8549
_CYCLE2_CAMPAIGN_OUTPUT_BEFORE = 30922
_SUPERSEDED_PLAN_SHA256 = "0a9e1ebccae2a793fadf531fb99013a9d19ff832a5880de9ee15204b477ca81a"
_SUPERSEDED_TEST_SPEC_SHA256 = "e3b8efef4b441d694883a856164ed3d25f2b278f2bd7ca9ee30dfb8097b3ba6c"
_RECOVERY_SEED_PLAN_SHA256 = "c6b16733a045e94e7310183e8be8a62cf4327b5d2bdc3eed40bb9d552ce17593"
_RECOVERY_SCHEDULE_SPEC_DIGEST = "ca13f87ee6b45f7b4d340f6c62715d4980f9966468e5511f7a06953f056c8ec5"
_VALIDATION_SUITE_SHA256 = "ee9db0e48dc055ed3e744a92166f274003f0d435cb58014acbd3e8d250d151b0"
_FINAL_SUITE_SHA256 = "1178831d3373f596affe032b88fc8a0768eec406d038e6c35bd9adfb84afefa3"
_VALIDATION_OPPONENT_ROSTER_SHA256 = "974adf788d5519633ad1fe7130a6deb634a0cc55d6af2d549e06d64c0f5496c0"
_FINAL_OPPONENT_ROSTER_SHA256 = "0fa20d714149054782dd5a83ab9ae99ebca6582b6c5702bdcc206bf7e3b61c27"
_VALIDATION_OPPONENT_SHA256 = "f79ed460afc6d4419e2e1b0bd084a4f1cf77058fcb8cdcebb7122a2864304412"
_FINAL_OPPONENT_CODE_SHA256 = "9e5103176218ca092206ba50899a35c43ad488ce825a95af487e76798ce60123"
_FINAL_OPPONENT_CONFIG_SHA256 = "4043dda35bb8cc2adc43112cb5b00eff50ac721eb062b15d0d83b7ae41510b7f"
_RECOVERY_BENCHMARK_GPU_SECONDS = 3600
_RECOVERY_TRAIN_GPU_SECONDS = 7200
_RECOVERY_MAX_OUTPUT_BYTES = 5_000_000_000
_RECOVERY_MAX_RUN_OUTPUT_BYTES = 1_000_000_000
_GPU_HOUR = 3600


class RegistryError(ValueError):
    """Registry is malformed or an operation violates its lifecycle."""


def _normal(value: Any) -> Any:
    if isinstance(value, str):
        result = unicodedata.normalize("NFC", value)
        if result != value:
            raise RegistryError("registry strings must already be NFC")
        return value
    if isinstance(value, float):
        raise RegistryError("registry JSON forbids floats")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, list):
        return [_normal(v) for v in value]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise RegistryError("registry object keys must be strings")
        return {_normal(k): _normal(v) for k, v in value.items()}
    raise RegistryError(f"unsupported registry value: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Serialize canonical UTF-8 JSON (NFC, sorted keys, compact, no floats)."""
    try:
        return json.dumps(_normal(value), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, RegistryError):
            raise
        raise RegistryError(f"not canonical-JSON serializable: {exc}") from exc


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HASH_CHARS for c in value)


def _validate_attempt_identity(d: dict[str, Any]) -> None:
    if not isinstance(d.get("attempt_id"), str) or not d["attempt_id"]:
        raise RegistryError("attempt_id must be a non-empty string")
    index = d.get("registration_event_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise RegistryError("invalid registration_event_index")
    if not _is_hash(d.get("registration_event_hash")):
        raise RegistryError("invalid registration_event_hash")


def _validate_attempt_hashes(d: dict[str, Any]) -> None:
    for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256"):
        if not _is_hash(d.get(key)):
            raise RegistryError(f"invalid {key}")
    for key in ("trainer_sha256", "engine_sha256"):
        if not _is_hash(d.get(key)):
            raise RegistryError(f"invalid {key}")
    timeout = d.get("session_timeout_seconds")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise RegistryError("session_timeout_seconds must be a positive integer")
    if not isinstance(d.get("kernel_id"), str) or not d["kernel_id"]:
        raise RegistryError("kernel_id must be non-empty")
    if not isinstance(d.get("implementation_variant_id"), str) or not d["implementation_variant_id"]:
        raise RegistryError("implementation_variant_id is required")


def _keys(data: Any, expected: set[str], event_type: str) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != expected:
        raise RegistryError(f"{event_type} event_data must contain exactly {sorted(expected)}")
    _normal(data)
    return data


def _validate_data(event_type: str, data: Any) -> None:
    if event_type == "recovery_cycle_amendment":
        expected = {"schema_id", "schema_version", "cycle_id", "created_at_utc", "prior_head_hash",
                    "prior_final_event_index", "proposal_sha256", "seed_plan_sha256",
                    "retired_registration_event_indices", "recovery_registrations", "compute_caps_gpu_seconds",
                    "project_gpu_seconds_before", "project_output_bytes_before", "quota_gpu_seconds_remaining_before",
                    "protected_final_reserve_gpu_seconds", "output_caps_bytes", "output_category_reservations_bytes",
                    "kernel_id", "current_version_number", "current_terminal_status", "current_status_evidence",
                    "quota_evidence", "user_authorization"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-kaggle-recovery-cycle-amendment" or d["schema_version"] != 1
                or d["cycle_id"] != "g008-recovery-cycle-2"):
            raise RegistryError("unsupported recovery cycle amendment schema")
        if (d["prior_final_event_index"] != _CYCLE1_FINAL_EVENT_INDEX
                or d["prior_head_hash"] != _CYCLE1_FINAL_EVENT_HASH):
            raise RegistryError("cycle-2 amendment must bind exact terminal cycle-1 head")
        if d["proposal_sha256"] != _CYCLE2_PROPOSAL_SHA256 or d["seed_plan_sha256"] != _RECOVERY_SEED_PLAN_SHA256:
            raise RegistryError("cycle-2 amendment does not bind approved proposal and seed plan")
        authorization = d["user_authorization"]
        if (not isinstance(authorization, dict)
                or set(authorization) != {"proposal_sha256", "statement", "confirmed_at_utc"}
                or authorization.get("proposal_sha256") != _CYCLE2_PROPOSAL_SHA256
                or not isinstance(authorization.get("statement"), str) or not authorization["statement"]
                or not isinstance(authorization.get("confirmed_at_utc"), str)
                or not authorization["confirmed_at_utc"].endswith("Z")):
            raise RegistryError("cycle-2 amendment requires durable user authorization bound to proposal")
        if d["retired_registration_event_indices"] != [14, 15]:
            raise RegistryError("cycle-2 amendment must retire both cycle-1 registrations")
        if d["compute_caps_gpu_seconds"] != {"benchmark": _CYCLE2_BENCHMARK_GPU_SECONDS,
                                             "train": _CYCLE2_TRAIN_GPU_SECONDS,
                                             "project_experiment": 21600, "project_total": 28800}:
            raise RegistryError("cycle-2 compute caps differ from approved proposal")
        if (d["project_gpu_seconds_before"] != _CYCLE2_CAMPAIGN_GPU_BEFORE
                or d["project_output_bytes_before"] != _CYCLE2_CAMPAIGN_OUTPUT_BEFORE):
            raise RegistryError("cycle-2 project baseline differs from terminal cycle-1 accounting")
        if not isinstance(d["kernel_id"], str) or not d["kernel_id"]:
            raise RegistryError("cycle-2 amendment requires exact Kaggle kernel identity")
        version = d["current_version_number"]
        if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
            raise RegistryError("cycle-2 amendment requires exact positive current version")
        if d["current_terminal_status"] not in {"COMPLETE", "ERROR", "CANCELLED", "TIMEOUT"}:
            raise RegistryError("cycle-2 amendment requires terminal current kernel status")
        status_evidence = d["current_status_evidence"]
        if (not isinstance(status_evidence, dict)
                or set(status_evidence) != {"status", "current_version_number", "kernel_id"}
                or status_evidence != {"status": d["current_terminal_status"],
                                       "current_version_number": version, "kernel_id": d["kernel_id"]}):
            raise RegistryError("cycle-2 amendment current status evidence is not exact/sanitized")
        quota_evidence = d["quota_evidence"]
        if (not isinstance(quota_evidence, dict)
                or set(quota_evidence) != {"remaining_gpu_seconds", "used_gpu_seconds", "source"}
                or quota_evidence.get("remaining_gpu_seconds") != d["quota_gpu_seconds_remaining_before"]
                or not isinstance(quota_evidence.get("remaining_gpu_seconds"), int)
                or isinstance(quota_evidence.get("remaining_gpu_seconds"), bool)
                or quota_evidence["remaining_gpu_seconds"] < 0
                or not isinstance(quota_evidence.get("used_gpu_seconds"), int)
                or isinstance(quota_evidence.get("used_gpu_seconds"), bool)
                or quota_evidence["used_gpu_seconds"] < 0
                or not isinstance(quota_evidence.get("source"), str) or not quota_evidence["source"]):
            raise RegistryError("cycle-2 amendment quota evidence must bind exact live account totals")
        quota = d["quota_gpu_seconds_remaining_before"]
        if not isinstance(quota, int) or isinstance(quota, bool) or quota < 0:
            raise RegistryError("cycle-2 amendment requires current quota evidence")
        if quota < _CYCLE2_BENCHMARK_GPU_SECONDS + _CYCLE2_TRAIN_GPU_SECONDS + 7200:
            raise RegistryError("cycle-2 quota cannot cover workloads and protected final reserve")
        if d["protected_final_reserve_gpu_seconds"] != 7200:
            raise RegistryError("cycle-2 must retain protected final reserve")
        if d["output_caps_bytes"] != {"aggregate": _RECOVERY_MAX_OUTPUT_BYTES, "per_run": _RECOVERY_MAX_RUN_OUTPUT_BYTES}:
            raise RegistryError("cycle-2 output caps differ from approved proposal")
        categories = {"benchmark": _CYCLE2_OUTPUT_CATEGORY_CAP, "training": _CYCLE2_OUTPUT_CATEGORY_CAP,
                      "validation": _CYCLE2_OUTPUT_CATEGORY_CAP, "final": _CYCLE2_OUTPUT_CATEGORY_CAP}
        if d["output_category_reservations_bytes"] != categories:
            raise RegistryError("cycle-2 output category reservations differ from approved proposal")
        if d["project_gpu_seconds_before"] + _CYCLE2_BENCHMARK_GPU_SECONDS + _CYCLE2_TRAIN_GPU_SECONDS > 21600:
            raise RegistryError("cycle-2 GPU projection exceeds experiment cap")
        if d["project_gpu_seconds_before"] + _CYCLE2_BENCHMARK_GPU_SECONDS + _CYCLE2_TRAIN_GPU_SECONDS + 7200 > 28800:
            raise RegistryError("cycle-2 GPU projection exceeds project cap with final reserve")
        if d["project_output_bytes_before"] + sum(categories.values()) > _RECOVERY_MAX_OUTPUT_BYTES:
            raise RegistryError("cycle-2 output reservations exceed aggregate cap")
        regs = d["recovery_registrations"]
        if (not isinstance(regs, list) or len(regs) != 2
                or sorted(r.get("registration_kind") for r in regs if isinstance(r, dict)) != ["benchmark", "train"]):
            raise RegistryError("cycle-2 amendment must name exactly one benchmark and one train")
        if any(set(r) != {"registration_id", "registration_kind", "run_uuid"}
               or not all(isinstance(r[k], str) and r[k] for k in r) for r in regs):
            raise RegistryError("cycle-2 registration identities are malformed")
        if len({r["registration_id"] for r in regs}) != 2 or len({r["run_uuid"] for r in regs}) != 2:
            raise RegistryError("cycle-2 registration IDs and UUIDs must be unique")
        if not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z"):
            raise RegistryError("cycle-2 amendment timestamp must be UTC")
    elif event_type == "recovery_amendment":
        expected = {"schema_id", "schema_version", "created_at_utc", "legacy_head_hash",
                    "legacy_final_event_index", "recovery_plan_sha256", "superseded_plan_sha256",
                    "superseded_test_spec_sha256", "historical_overrun_evidence", "user_request_basis",
                    "recovery_registrations", "submission_attempt_limit", "compute_caps_gpu_seconds",
                    "output_caps_bytes", "protected_final_reserve_gpu_seconds", "required_preflight",
                    "reservation_metadata"}
        d = _keys(data, expected, event_type)
        if d["schema_id"] != "g008-kaggle-recovery-amendment" or d["schema_version"] != 1:
            raise RegistryError("unsupported recovery amendment schema")
        if d["legacy_head_hash"] != _LEGACY_HEAD_HASH or d["legacy_final_event_index"] != _LEGACY_HEAD_INDEX:
            raise RegistryError("recovery amendment does not bind the exact immutable legacy head")
        for key in ("recovery_plan_sha256", "superseded_plan_sha256", "superseded_test_spec_sha256"):
            if not _is_hash(d[key]):
                raise RegistryError(f"invalid {key}")
        if d["recovery_plan_sha256"] != _RECOVERY_PLAN_SHA256:
            raise RegistryError("recovery amendment does not bind the approved plan bytes")
        if (d["superseded_plan_sha256"] != _SUPERSEDED_PLAN_SHA256
                or d["superseded_test_spec_sha256"] != _SUPERSEDED_TEST_SPEC_SHA256):
            raise RegistryError("recovery amendment changed superseded plan/test-spec anchors")
        if d["submission_attempt_limit"] is not None:
            raise RegistryError("recovery amendment must remove only the fixed attempt-count limit")
        if d["compute_caps_gpu_seconds"] != {"benchmark": _RECOVERY_BENCHMARK_GPU_SECONDS,
                                             "training": _RECOVERY_TRAIN_GPU_SECONDS,
                                             "project_experiment": 6 * _GPU_HOUR,
                                             "project_total": 8 * _GPU_HOUR}:
            raise RegistryError("recovery compute caps differ from approved plan")
        if d["protected_final_reserve_gpu_seconds"] != 2 * _GPU_HOUR:
            raise RegistryError("recovery amendment must retain the protected final reserve")
        if d["output_caps_bytes"] != {"aggregate": _RECOVERY_MAX_OUTPUT_BYTES,
                                      "per_run": _RECOVERY_MAX_RUN_OUTPUT_BYTES}:
            raise RegistryError("recovery output caps differ from approved plan")
        if not isinstance(d["recovery_registrations"], list) or len(d["recovery_registrations"]) != 2:
            raise RegistryError("recovery amendment must name exactly two workloads")
        kinds = [r.get("registration_kind") for r in d["recovery_registrations"] if isinstance(r, dict)]
        if sorted(kinds) != ["benchmark", "train"]:
            raise RegistryError("recovery amendment must name one benchmark and one train workload")
        if any(set(r) != {"registration_id", "registration_kind", "run_uuid"}
               or not all(isinstance(r[k], str) and r[k] for k in r)
               for r in d["recovery_registrations"]):
            raise RegistryError("recovery registration identities are malformed")
        ids = [r["registration_id"] for r in d["recovery_registrations"]]
        uuids = [r["run_uuid"] for r in d["recovery_registrations"]]
        if len(set(ids)) != 2 or len(set(uuids)) != 2:
            raise RegistryError("recovery registration IDs and run UUIDs must be unique")
        if not isinstance(d["historical_overrun_evidence"], dict) or not isinstance(d["user_request_basis"], dict):
            raise RegistryError("recovery amendment evidence must be objects")
        historical = d["historical_overrun_evidence"]
        terminal_history = historical.get("prior_attempts_terminal_evidence")
        if historical.get("all_prior_attempts_terminal") is not True or not isinstance(terminal_history, list) or not terminal_history:
            raise RegistryError("recovery amendment must reconcile every historical attempt to terminal status")
        if not isinstance(historical.get("kernel_id"), str) or not historical["kernel_id"]:
            raise RegistryError("historical reconciliation must bind exact kernel identity")
        if not isinstance(historical.get("current_version_number"), int) or isinstance(historical.get("current_version_number"), bool) or historical["current_version_number"] <= 0:
            raise RegistryError("historical reconciliation must bind the exact current kernel version")
        historical_versions: set[int] = set()
        for prior in terminal_history:
            if (not isinstance(prior, dict) or prior.get("kernel_id") != historical["kernel_id"]
                    or not isinstance(prior.get("version"), int) or isinstance(prior.get("version"), bool) or prior["version"] <= 0
                    or prior.get("status") not in {"COMPLETE", "ERROR", "CANCELLED", "TIMEOUT"}
                    or not isinstance(prior.get("evidence"), dict)):
                raise RegistryError("historical attempt is missing exact terminal status evidence")
            historical_versions.add(prior["version"])
        if historical["current_version_number"] not in historical_versions:
            raise RegistryError("current kernel version lacks bound historical terminal evidence")
        if not isinstance(d["required_preflight"], dict) or not isinstance(d["reservation_metadata"], dict):
            raise RegistryError("recovery preflight and reservation metadata must be objects")
        if d["reservation_metadata"].get("approved_corrected_cycle5_seed_plan_sha256") != _RECOVERY_SEED_PLAN_SHA256:
            raise RegistryError("recovery reservation does not bind the approved Cycle-5 seed plan")
        exact_reservation_hashes = {
            "schedule_spec_digest": _RECOVERY_SCHEDULE_SPEC_DIGEST,
            "validation_suite_sha256": _VALIDATION_SUITE_SHA256,
            "final_suite_sha256": _FINAL_SUITE_SHA256,
            "validation_opponent_roster_sha256": _VALIDATION_OPPONENT_ROSTER_SHA256,
            "final_opponent_roster_sha256": _FINAL_OPPONENT_ROSTER_SHA256,
            "validation_opponent_checkpoint_sha256": _VALIDATION_OPPONENT_SHA256,
            "final_opponent_code_sha256": _FINAL_OPPONENT_CODE_SHA256,
            "final_opponent_config_sha256": _FINAL_OPPONENT_CONFIG_SHA256,
        }
        if any(d["reservation_metadata"].get(key) != expected for key, expected in exact_reservation_hashes.items()):
            raise RegistryError("recovery reservation hashes differ from the frozen seed-plan /3 suite")
        if (d["reservation_metadata"].get("validation_opponent_id") != "g007-step-300"
                or d["reservation_metadata"].get("final_opponent_id") != "open-space"):
            raise RegistryError("recovery reservation changed frozen validation/final opponent identities")
        if d["reservation_metadata"].get("reservation_only") is not True or d["reservation_metadata"].get("pre_lock_execution") is not False:
            raise RegistryError("validation/final reservations must be metadata-only")
        for key in ("campaign_gpu_seconds_before", "campaign_output_bytes_before", "quota_gpu_seconds_remaining_before"):
            value = d["reservation_metadata"].get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RegistryError(f"recovery reservation requires non-negative {key}")
        if d["reservation_metadata"]["campaign_gpu_seconds_before"] < 8424:
            raise RegistryError("recovery amendment must account for at least the recorded 2.34 G008 GPU-hours")
        if d["reservation_metadata"]["campaign_gpu_seconds_before"] + _RECOVERY_BENCHMARK_GPU_SECONDS + _RECOVERY_TRAIN_GPU_SECONDS > 6 * _GPU_HOUR:
            raise RegistryError("recovery reservations exceed remaining experiment allocation")
        if d["reservation_metadata"]["campaign_output_bytes_before"] > _RECOVERY_MAX_OUTPUT_BYTES:
            raise RegistryError("prior campaign output already exceeds aggregate cap")
        if d["reservation_metadata"]["quota_gpu_seconds_remaining_before"] - _RECOVERY_BENCHMARK_GPU_SECONDS - _RECOVERY_TRAIN_GPU_SECONDS < 2 * _GPU_HOUR:
            raise RegistryError("recovery would consume the protected two-hour account reserve")
        reservation = d["reservation_metadata"]
        for key in ("validation_game_count", "final_game_count"):
            value = reservation.get(key)
            if value != 512 or isinstance(value, bool):
                raise RegistryError(f"reservation metadata changed frozen {key}")
        seats = reservation.get("target_seat_order")
        if seats != [0, 1]:
            raise RegistryError("reservation metadata changed frozen target seat order")
        for name, first in (("validation", 1_000_000), ("final", 4_000_000)):
            roots = reservation.get(f"{name}_roots")
            bounds = reservation.get(f"{name}_root_range")
            if roots != list(range(first, first + 256)) or bounds != [first, first + 255]:
                raise RegistryError(f"reservation metadata changed frozen {name} root roster/range")
        policy = reservation.get("protocol_selection_policy")
        if policy != {"protocol": "G008 frozen principal protocol / schedule seed plan /3",
                      "selection": "candidate requires parity and median paired improvement >=10% with at least 4/5 pairs >=10%; otherwise use incumbent"}:
            raise RegistryError("reservation metadata changed frozen protocol or implementation selection policy")
        gpu = reservation.get("reserved_gpu_seconds")
        runtime = reservation.get("reserved_runtime_seconds")
        output = reservation.get("reserved_output_bytes")
        if not all(isinstance(obj, dict) for obj in (gpu, runtime, output)):
            raise RegistryError("reservation metadata must bind GPU, runtime, and output resources")
        resource_keys = {"benchmark", "training", "validation", "final"}
        if any(set(obj) != resource_keys for obj in (gpu, runtime, output)):
            raise RegistryError("reservation resource objects must bind benchmark, training, validation, and final")
        if gpu.get("benchmark") != _RECOVERY_BENCHMARK_GPU_SECONDS or gpu.get("training") != _RECOVERY_TRAIN_GPU_SECONDS:
            raise RegistryError("reservation metadata workload GPU bounds differ from approved caps")
        if any(not isinstance(gpu.get(k), int) or isinstance(gpu[k], bool) or gpu[k] < 0 for k in ("validation", "final")):
            raise RegistryError("validation/final GPU reservations must be non-negative seconds")
        if gpu["validation"] + gpu["final"] > 2 * _GPU_HOUR:
            raise RegistryError("validation/final GPU reservations exceed protected final reserve")
        for key, maximum in (("benchmark", 3600), ("training", 7200), ("validation", 7200), ("final", 7200)):
            if not isinstance(runtime.get(key), int) or isinstance(runtime[key], bool) or not 0 <= runtime[key] <= maximum:
                raise RegistryError(f"invalid reserved runtime bound for {key}")
            if not isinstance(output.get(key), int) or isinstance(output[key], bool) or not 0 <= output[key] <= _RECOVERY_MAX_OUTPUT_BYTES:
                raise RegistryError(f"invalid reserved output bound for {key}")
        if runtime["benchmark"] + runtime["training"] > _RECOVERY_BENCHMARK_GPU_SECONDS + _RECOVERY_TRAIN_GPU_SECONDS or runtime["validation"] + runtime["final"] > 2 * _GPU_HOUR:
            raise RegistryError("reserved runtime exceeds workload/final-evaluation budgets")
        if sum(output.values()) > _RECOVERY_MAX_OUTPUT_BYTES or any(value > _RECOVERY_MAX_RUN_OUTPUT_BYTES for value in output.values()):
            raise RegistryError("validation/final and workload outputs exceed approved output caps")
        if not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z"):
            raise RegistryError("recovery amendment timestamp must be UTC")
    elif event_type == "registration":
        required = {"registration_id", "registration_kind", "record"}
        cycle2 = required | {"cycle_id"}
        d = _keys(data, cycle2 if isinstance(data, dict) and set(data) == cycle2 else required, event_type)
        if "cycle_id" in d and d["cycle_id"] != "g008-recovery-cycle-2":
            raise RegistryError("unsupported registration cycle_id")
        if not isinstance(d["registration_id"], str) or not d["registration_id"]:
            raise RegistryError("registration_id must be a non-empty string")
        if d["registration_kind"] not in _REGISTRATION_KINDS:
            raise RegistryError("unsupported registration_kind")
        if not isinstance(d["record"], dict):
            raise RegistryError("registration record must be an object")
    elif event_type == "registration_closed":
        d = _keys(data, {"pre_lock_head_hash", "pre_lock_final_event_index"}, event_type)
        idx, head = d["pre_lock_final_event_index"], d["pre_lock_head_hash"]
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < -1:
            raise RegistryError("invalid pre-lock final event index")
        if idx == -1:
            if head is not None:
                raise RegistryError("empty prefix must have null head")
        elif not _is_hash(head):
            raise RegistryError("non-empty prefix must have a SHA-256 head")
    elif event_type == "benchmark_artifact":
        old = {"artifact_sha256", "pre_lock_head_hash", "pre_lock_final_event_index",
               "registration_event_index", "registration_event_hash"}
        recovery = old | {"attempt_event_index", "attempt_event_hash", "report_sha256", "selection_evidence"}
        d = _keys(data, recovery if isinstance(data, dict) and set(data) == recovery else old, event_type)
        _validate_benchmark_binding(d)
        if "attempt_event_index" in d:
            if not isinstance(d["attempt_event_index"], int) or isinstance(d["attempt_event_index"], bool) or d["attempt_event_index"] < 0:
                raise RegistryError("invalid benchmark attempt event index")
            if not _is_hash(d["attempt_event_hash"]) or not _is_hash(d["report_sha256"]):
                raise RegistryError("invalid recovery benchmark attempt/report hash")
            _validate_selection_evidence(d["selection_evidence"])
    elif event_type == "benchmark_attestation":
        old = {"artifact_sha256", "pre_lock_head_hash", "pre_lock_final_event_index",
               "registration_event_index", "registration_event_hash", "lock_head_hash"}
        recovery = old | {"attempt_event_index", "attempt_event_hash", "report_sha256", "selection_evidence"}
        d = _keys(data, recovery if isinstance(data, dict) and set(data) == recovery else old, event_type)
        _validate_benchmark_binding(d)
        if not _is_hash(d["lock_head_hash"]):
            raise RegistryError("invalid lock_head_hash")
            if "selection_evidence" in d:
                _validate_selection_evidence(d["selection_evidence"])
    elif event_type in {"workload_completion", "evidence"}:
        d = _keys(data, {"registration_event_index", "registration_event_hash", "lock_head_hash", "artifact_sha256"}, event_type)
        if (not isinstance(d["registration_event_index"], int) or isinstance(d["registration_event_index"], bool)
                or d["registration_event_index"] < 0):
            raise RegistryError("invalid registration_event_index")
        if not all(_is_hash(d[k]) for k in ("registration_event_hash", "lock_head_hash", "artifact_sha256")):
            raise RegistryError("invalid hash in post-lock evidence")
    elif event_type == "workload_preflight":
        required = {"attempt_id", "registration_event_index", "registration_event_hash", "registration_id",
                         "kernel_id", "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                         "backend_sha256", "trainer_sha256", "engine_sha256", "session_timeout_seconds", "accelerator_count", "reserved_gpu_seconds",
                         "expected_current_version", "quota_evidence", "policy_evidence", "prior_attempts",
                         "implementation_variant_id", "no_outside_writer_confirmed", "exclusive_lock_id", "reconciliation_slack_seconds",
                         "lock_head_hash"}
        cycle2 = required | {"cycle_id"}
        d = _keys(data, cycle2 if isinstance(data, dict) and set(data) == cycle2 else required, event_type)
        _validate_attempt_identity(d)
        _validate_attempt_hashes(d)
        if d["reserved_gpu_seconds"] != d["session_timeout_seconds"] or d["accelerator_count"] != 1:
            raise RegistryError("preflight reservation must equal timeout on one accelerator")
        if not isinstance(d["quota_evidence"], dict) or not isinstance(d["policy_evidence"], dict):
            raise RegistryError("preflight quota/policy evidence must be objects")
        if not isinstance(d["expected_current_version"], int) or isinstance(d["expected_current_version"], bool) or d["expected_current_version"] <= 0:
            raise RegistryError("preflight requires exact positive current kernel version metadata")
        remaining = d["quota_evidence"].get("remaining_gpu_seconds")
        if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0:
            raise RegistryError("preflight must bind current account GPU quota remaining")
        if d["policy_evidence"].get("paid_services") is not False or d["policy_evidence"].get("billing_changes") is not False:
            raise RegistryError("recovery preflight forbids paid services and billing changes")
        if d["policy_evidence"].get("machine_shape") != "NvidiaTeslaT4":
            raise RegistryError("recovery preflight must select the approved single-T4 machine shape")
        if d["policy_evidence"].get("prior_runs_terminal") is not True or d["policy_evidence"].get("no_queued_or_running") is not True:
            raise RegistryError("preflight must prove prior attempts terminal and no kernel run active")
        if not isinstance(d["prior_attempts"], list):
            raise RegistryError("preflight prior_attempts must be a list")
        if d["no_outside_writer_confirmed"] is not True or not isinstance(d["exclusive_lock_id"], str) or not d["exclusive_lock_id"]:
            raise RegistryError("latest-status preflight requires outside-writer confirmation and exclusive lock")
        if not isinstance(d["reconciliation_slack_seconds"], int) or isinstance(d["reconciliation_slack_seconds"], bool) or not 0 <= d["reconciliation_slack_seconds"] <= 600:
            raise RegistryError("reconciliation slack exceeds approved maximum")
        if d["lock_head_hash"] is not None and not _is_hash(d["lock_head_hash"]):
            raise RegistryError("invalid training lock_head_hash")
    elif event_type == "workload_push_dispatched":
        required = {"attempt_id", "registration_event_index", "registration_event_hash", "preflight_event_index",
                         "preflight_event_hash", "kernel_id", "package_sha256", "source_sha256", "config_sha256",
                         "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256", "session_timeout_seconds", "accelerator_count",
                         "reserved_gpu_seconds", "implementation_variant_id", "dispatch_timestamp_utc", "request_identity", "lock_head_hash"}
        cycle2 = required | {"cycle_id"}
        d = _keys(data, cycle2 if isinstance(data, dict) and set(data) == cycle2 else required, event_type)
        _validate_attempt_identity(d)
        _validate_attempt_hashes(d)
        if not _is_hash(d["preflight_event_hash"]) or d["reserved_gpu_seconds"] != d["session_timeout_seconds"] or d["accelerator_count"] != 1:
            raise RegistryError("dispatch must link a valid preflight and reserve one T4 for the full timeout")
        if not isinstance(d["dispatch_timestamp_utc"], str) or not d["dispatch_timestamp_utc"].endswith("Z"):
            raise RegistryError("dispatch timestamp must be UTC")
        if not isinstance(d["request_identity"], str) or not d["request_identity"]:
            raise RegistryError("dispatch request identity is required")
        if d["lock_head_hash"] is not None and not _is_hash(d["lock_head_hash"]):
            raise RegistryError("invalid training lock_head_hash")
    elif event_type == "workload_push_accepted":
        required = {"attempt_id", "registration_event_index", "registration_event_hash", "preflight_event_index",
                         "preflight_event_hash", "dispatch_event_index", "dispatch_event_hash", "assigned_version",
                         "implementation_variant_id", "push_response", "confirming_metadata", "lock_head_hash"}
        cycle2 = required | {"cycle_id"}
        d = _keys(data, cycle2 if isinstance(data, dict) and set(data) == cycle2 else required, event_type)
        _validate_attempt_identity(d)
        if not isinstance(d["implementation_variant_id"], str) or not d["implementation_variant_id"]:
            raise RegistryError("accepted push must preserve implementation variant")
        for key in ("preflight_event_hash", "dispatch_event_hash"):
            if not _is_hash(d[key]):
                raise RegistryError(f"invalid {key}")
        if not isinstance(d["assigned_version"], int) or isinstance(d["assigned_version"], bool) or d["assigned_version"] <= 0:
            raise RegistryError("assigned Kaggle version must be a positive integer")
        if not isinstance(d["push_response"], dict) or not isinstance(d["confirming_metadata"], dict):
            raise RegistryError("accepted push evidence must be objects")
        if d["confirming_metadata"].get("current_version_number") != d["assigned_version"]:
            raise RegistryError("acceptance requires metadata confirming the assigned current version")
        if not isinstance(d["confirming_metadata"].get("kernel_id"), str) or not d["confirming_metadata"]["kernel_id"]:
            raise RegistryError("acceptance requires exact kernel identity metadata")
        if d["lock_head_hash"] is not None and not _is_hash(d["lock_head_hash"]):
            raise RegistryError("invalid training lock_head_hash")
    elif event_type == "workload_attempt":
        required = {"attempt_id", "outcome", "registration_event_index", "registration_event_hash",
                         "preflight_event_index", "preflight_event_hash", "dispatch_event_index", "dispatch_event_hash",
                         "accepted_event_index", "accepted_event_hash", "assigned_version", "terminal_status",
                         "terminal_metadata", "status_evidence", "actual_gpu_seconds", "output_bytes", "result",
                         "artifact_sha256", "rejection_evidence", "no_assignment_metadata", "implementation_variant_id",
                         "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256",
                         "trainer_sha256", "engine_sha256", "lock_head_hash"}
        cycle2 = required | {"cycle_id"}
        d = _keys(data, cycle2 if isinstance(data, dict) and set(data) == cycle2 else required, event_type)
        _validate_attempt_identity(d)
        if not isinstance(d["implementation_variant_id"], str) or not d["implementation_variant_id"]:
            raise RegistryError("terminal attempt must preserve implementation variant")
        for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256"):
            if not _is_hash(d[key]):
                raise RegistryError(f"invalid {key}")
        if d["outcome"] not in {"terminal", "pre_assignment_rejected"}:
            raise RegistryError("workload_attempt must be a terminal resolution")
        for key in ("preflight_event_hash", "dispatch_event_hash"):
            if not _is_hash(d[key]):
                raise RegistryError(f"invalid {key}")
        if d["outcome"] == "pre_assignment_rejected":
            if any(d[k] is not None for k in ("accepted_event_index", "accepted_event_hash", "assigned_version")):
                raise RegistryError("pre-assignment rejection must not claim an accepted version")
            if d["actual_gpu_seconds"] != 0 or not isinstance(d["rejection_evidence"], dict) or not isinstance(d["no_assignment_metadata"], dict):
                raise RegistryError("pre-assignment rejection requires proof and zero remote GPU use")
        else:
            if not isinstance(d["accepted_event_index"], int) or not _is_hash(d["accepted_event_hash"]):
                raise RegistryError("terminal attempt must link its accepted push receipt")
            if not isinstance(d["assigned_version"], int) or isinstance(d["assigned_version"], bool) or d["assigned_version"] <= 0:
                raise RegistryError("terminal attempt must bind its assigned version")
            if d["terminal_status"] not in {"COMPLETE", "ERROR", "CANCELLED", "TIMEOUT"}:
                raise RegistryError("terminal attempt status is not recognized")
            if not isinstance(d["terminal_metadata"], dict) or not isinstance(d["status_evidence"], list):
                raise RegistryError("terminal attempt requires metadata and poll evidence")
            if not d["status_evidence"] or d["terminal_metadata"].get("status") != d["terminal_status"]:
                raise RegistryError("terminal attempt must bind matching observed status evidence")
            if d["terminal_metadata"].get("current_version_number") != d["assigned_version"]:
                raise RegistryError("terminal metadata must identify the accepted current version")
            if not isinstance(d["terminal_metadata"].get("kernel_id"), str) or not d["terminal_metadata"]["kernel_id"]:
                raise RegistryError("terminal metadata must identify exact kernel")
            last_observation = d["status_evidence"][-1]
            if not isinstance(last_observation, dict):
                raise RegistryError("terminal status evidence must contain observed response objects")
            if (last_observation.get("status") != d["terminal_status"]
                    or last_observation.get("current_version_number") != d["assigned_version"]
                    or last_observation.get("kernel_id") != d["terminal_metadata"]["kernel_id"]):
                raise RegistryError("last observed status response must confirm exact terminal status, version, and kernel")
            for observation in d["status_evidence"]:
                if (not isinstance(observation, dict) or not isinstance(observation.get("status"), str)
                        or observation.get("current_version_number") != d["assigned_version"]
                        or observation.get("kernel_id") != d["terminal_metadata"]["kernel_id"]):
                    raise RegistryError("status observations must bind exact assigned version and kernel")
        for key in ("actual_gpu_seconds", "output_bytes"):
            if not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < 0:
                raise RegistryError(f"{key} must be a non-negative integer")
        if d["artifact_sha256"] is not None and not _is_hash(d["artifact_sha256"]):
            raise RegistryError("invalid attempt artifact SHA-256")
        if not isinstance(d["result"], dict):
            raise RegistryError("attempt result must be an object")
        if d["lock_head_hash"] is not None and not _is_hash(d["lock_head_hash"]):
            raise RegistryError("invalid training lock_head_hash")
    else:
        raise RegistryError(f"unknown event_type: {event_type!r}")


def _validate_benchmark_binding(d: dict[str, Any]) -> None:
    if not _is_hash(d["artifact_sha256"]) or not _is_hash(d["registration_event_hash"]):
        raise RegistryError("invalid benchmark artifact/registration hash")
    if d["pre_lock_head_hash"] is not None and not _is_hash(d["pre_lock_head_hash"]):
        raise RegistryError("invalid pre-lock head hash")
    for key in ("pre_lock_final_event_index", "registration_event_index"):
        if not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < -1:
            raise RegistryError(f"invalid {key}")


def _validate_selection_evidence(evidence: Any) -> None:
    """Validate the exact decision fields copied from the paired benchmark /3 report."""
    if not isinstance(evidence, dict) or set(evidence) != {
            "parity", "pairs", "summary", "selection", "selected_variant_id"}:
        raise RegistryError("benchmark selection evidence must preserve the paired report decision fields")
    parity = evidence["parity"]
    pairs = evidence["pairs"]
    summary = evidence["summary"]
    if not isinstance(parity, dict) or set(parity) != {"passed"} or type(parity["passed"]) is not bool:
        raise RegistryError("benchmark selection evidence requires exact parity.passed boolean")
    if not isinstance(pairs, list) or len(pairs) != 5:
        raise RegistryError("benchmark selection evidence requires exactly five paired speedup ratios")
    ratios: list[Decimal] = []
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"candidate_speedup_ratio"}:
            raise RegistryError("benchmark pair selection evidence must contain only candidate_speedup_ratio")
        ratio = _selection_decimal(pair["candidate_speedup_ratio"], positive=True)
        ratios.append(ratio)
    median_node = summary.get("candidate_speedup_ratio") if isinstance(summary, dict) else None
    if (not isinstance(summary, dict) or set(summary) != {"candidate_speedup_ratio"}
            or not isinstance(median_node, dict) or set(median_node) != {"median"}):
        raise RegistryError("benchmark selection evidence requires summary.candidate_speedup_ratio.median")
    median = _selection_decimal(median_node["median"], positive=False)
    if median != statistics.median(ratios):
        raise RegistryError("benchmark median candidate speedup differs from the five paired ratios")
    gain_threshold = Decimal("1.10")
    selected = ("candidate" if parity["passed"] and median >= gain_threshold
                and sum(ratio >= gain_threshold for ratio in ratios) >= 4 else "incumbent")
    if evidence["selection"] != selected or evidence["selected_variant_id"] != selected:
        raise RegistryError("benchmark selection does not follow the frozen parity and paired-gain criteria")


def _selection_decimal(value: Any, *, positive: bool) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise RegistryError("benchmark speedup evidence must be finite numeric data")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise RegistryError("benchmark speedup evidence must be finite numeric data") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise RegistryError("benchmark speedup evidence must be finite positive data")
    if isinstance(value, str) and format(parsed.normalize(), "f") != value:
        raise RegistryError("benchmark speedup evidence string is not canonical")
    return parsed


def _canonical_selection_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Store report numeric values as canonical decimal strings (registry JSON forbids floats)."""
    _validate_selection_evidence(evidence)
    return {
        "parity": {"passed": evidence["parity"]["passed"]},
        "pairs": [{"candidate_speedup_ratio": format(
            _selection_decimal(pair["candidate_speedup_ratio"], positive=True).normalize(), "f")}
            for pair in evidence["pairs"]],
        "summary": {"candidate_speedup_ratio": {"median": format(
            _selection_decimal(evidence["summary"]["candidate_speedup_ratio"]["median"], positive=False).normalize(), "f")}},
        "selection": evidence["selection"],
        "selected_variant_id": evidence["selected_variant_id"],
    }


def _payload(index: int, event_type: str, event_data: dict[str, Any], prev: str) -> dict[str, Any]:
    return {"event_index": index, "event_type": event_type, "event_data": event_data,
            "prev_event_hash": prev}


def _event_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _decode_line(raw: bytes, expected_index: int, expected_prev: str) -> dict[str, Any]:
    try:
        event = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"invalid registry JSON at event {expected_index}") from exc
    if not isinstance(event, dict) or set(event) != {"event_index", "event_type", "event_data", "prev_event_hash", "event_hash"}:
        raise RegistryError(f"invalid event fields at event {expected_index}")
    payload = {k: event[k] for k in ("event_index", "event_type", "event_data", "prev_event_hash")}
    _validate_data(event["event_type"], event["event_data"])
    if payload["event_index"] != expected_index or payload["prev_event_hash"] != expected_prev:
        raise RegistryError(f"broken index/hash chain at event {expected_index}")
    if _event_hash(payload) != event["event_hash"]:
        raise RegistryError(f"event hash mismatch at event {expected_index}")
    if canonical_bytes(event) != raw.rstrip(b"\n"):
        raise RegistryError(f"non-canonical event encoding at event {expected_index}")
    return event


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RegistryError("registry has an incomplete trailing event")
    lines = raw.splitlines()
    events: list[dict[str, Any]] = []
    prev = GENESIS
    registrations: dict[str, int] = {}
    lock_hash: str | None = None
    benchmarks: dict[int, dict[str, Any]] = {}
    attestations: set[int] = set()
    for i, line in enumerate(lines):
        event = _decode_line(line, i, prev)
        typ, data = event["event_type"], event["event_data"]
        if typ == "registration":
            if lock_hash is not None:
                raise RegistryError("registration after registration_closed")
            rid = data["registration_id"]
            if rid in registrations:
                raise RegistryError("duplicate registration_id")
            registrations[rid] = i
        elif typ == "registration_closed":
            if lock_hash is not None:
                raise RegistryError("duplicate registration_closed")
            pre_head = events[-1]["event_hash"] if events else None
            pre_index = len(events) - 1
            if data != {"pre_lock_head_hash": pre_head, "pre_lock_final_event_index": pre_index}:
                raise RegistryError("registration_closed does not pin exact preceding prefix")
            lock_hash = event["event_hash"]
        elif typ == "benchmark_artifact":
            if lock_hash is not None:
                raise RegistryError("benchmark artifact must be recorded pre-lock")
            _verify_benchmark_registration(data, events, registrations)
            if data["pre_lock_head_hash"] != (events[-1]["event_hash"] if events else None) or data["pre_lock_final_event_index"] != len(events)-1:
                raise RegistryError("benchmark artifact has stale prefix anchor")
            benchmarks[data["registration_event_index"]] = data
        elif typ == "benchmark_attestation":
            if lock_hash is None:
                raise RegistryError("benchmark attestation requires closed registrations")
            reg_i = data["registration_event_index"]
            artifact = benchmarks.get(reg_i)
            if artifact is None or reg_i in attestations:
                raise RegistryError("missing/duplicate benchmark artifact attestation")
            expected = {**artifact, "lock_head_hash": lock_hash}
            if data != expected:
                raise RegistryError("benchmark attestation does not match immutable artifact/lock")
            attestations.add(reg_i)
        elif typ in {"workload_completion", "evidence"}:
            if lock_hash is None or data["lock_head_hash"] != lock_hash:
                raise RegistryError("post-lock evidence has absent or mismatched lock")
            reg_i = data["registration_event_index"]
            if reg_i >= i or reg_i < 0 or events[reg_i]["event_type"] != "registration":
                raise RegistryError("post-lock evidence references no prior registration")
            reg = events[reg_i]
            if data["registration_event_hash"] != reg["event_hash"]:
                raise RegistryError("post-lock evidence registration hash mismatch")
        prev = event["event_hash"]
        events.append(event)
    cycle2_indices = [i for i, event in enumerate(events) if event["event_type"] == "recovery_cycle_amendment"]
    if cycle2_indices:
        _validate_recovery_suffix(events[:_CYCLE2_EVENT_INDEX])
        _validate_recovery_cycle2_suffix(events)
    else:
        _validate_recovery_suffix(events)
    return events


def _verify_benchmark_registration(data: dict[str, Any], events: list[dict[str, Any]], registrations: dict[str, int]) -> None:
    i = data["registration_event_index"]
    if i < 0 or i >= len(events):
        raise RegistryError("benchmark registration reference is not prior")
    event = events[i]
    if (event["event_type"] != "registration" or event["event_data"]["registration_kind"] != "benchmark"
            or event["event_hash"] != data["registration_event_hash"]):
        raise RegistryError("benchmark artifact registration reference mismatch")


def _validate_recovery_suffix(events: list[dict[str, Any]]) -> None:
    """Validate the one-time recovery suffix while leaving the legacy prefix untouched."""
    amendments = [i for i, event in enumerate(events) if event["event_type"] == "recovery_amendment"]
    recovery_types = _RECOVERY_EVENT_TYPES | {"workload_attempt"}
    if not amendments:
        if any(e["event_type"] in recovery_types for e in events):
            raise RegistryError("recovery lifecycle event has no amendment")
        return
    if amendments != [_LEGACY_HEAD_INDEX + 1]:
        raise RegistryError("recovery amendment must be the sole event at immutable legacy index 13")
    if len(events) <= _LEGACY_HEAD_INDEX or events[_LEGACY_HEAD_INDEX]["event_hash"] != _LEGACY_HEAD_HASH:
        raise RegistryError("recovery amendment requires the exact immutable legacy head")
    amendment = events[amendments[0]]["event_data"]
    named = {r["registration_id"]: r for r in amendment["recovery_registrations"]}
    registrations: dict[int, dict[str, Any]] = {}
    id_to_index: dict[str, int] = {}
    variant_sets: dict[str, dict[str, dict[str, str]]] = {}
    attempt_state: dict[str, dict[str, Any]] = {}
    attempt_order: list[str] = []
    seen_run_ids: set[str] = set()
    recovery_start = amendments[0] + 1
    recovery_events = events[recovery_start:]
    lock_index = next((i for i, e in enumerate(recovery_events, recovery_start)
                       if e["event_type"] == "registration_closed"), None)
    recovery_artifact_index = None
    recovery_artifact_data = None
    recovery_attestation_index = None
    current_kernel_version = amendment["historical_overrun_evidence"]["current_version_number"]

    # The two authorized workload registrations are the only registration events after the amendment.
    for i, event in enumerate(recovery_events, recovery_start):
        typ, data = event["event_type"], event["event_data"]
        if (typ in {"workload_preflight", "registration_closed", "benchmark_artifact"}
                and set(id_to_index) != set(named)):
            raise RegistryError("both recovery registrations must precede execution and lock")
        if recovery_artifact_index is not None and (lock_index is None or i < lock_index) and typ != "registration_closed":
            raise RegistryError("recovery benchmark artifact must immediately precede registration lock")
        if lock_index is not None and i > lock_index:
            required_post_lock = "benchmark_attestation" if i == lock_index + 1 else None
            if required_post_lock and typ != required_post_lock:
                raise RegistryError("benchmark attestation must immediately follow recovery lock")
            if i == lock_index + 1 and typ == "benchmark_attestation":
                pass
            elif i > lock_index + 1 and typ == "benchmark_attestation":
                raise RegistryError("duplicate or delayed recovery benchmark attestation")
        if typ == "registration":
            rid = data["registration_id"]
            if rid not in named or rid in id_to_index:
                raise RegistryError("amendment authorizes exactly its two named registrations")
            rule = named[rid]
            rec = data["record"]
            if data["registration_kind"] != rule["registration_kind"] or rec.get("run_uuid") != rule["run_uuid"]:
                raise RegistryError("recovery registration differs from amendment identity")
            if rec.get("recovery_amendment_event_index") != amendments[0] or rec.get("recovery_amendment_event_hash") != events[amendments[0]]["event_hash"]:
                raise RegistryError("recovery registration does not pin amendment")
            if rec.get("submission_policy") != "cumulative_compute_only":
                raise RegistryError("recovery registration lacks cumulative-compute-only policy")
            kind = data["registration_kind"]
            expected_cap = _RECOVERY_BENCHMARK_GPU_SECONDS if kind == "benchmark" else _RECOVERY_TRAIN_GPU_SECONDS
            if rec.get("workload_gpu_seconds_cap") != expected_cap:
                raise RegistryError("recovery workload compute cap differs from approved plan")
            timeout = rec.get("session_timeout_seconds")
            if not isinstance(timeout, int) or isinstance(timeout, bool) or not 0 < timeout <= expected_cap:
                raise RegistryError("recovery workload timeout must be finite and within its cap")
            if rec.get("accelerator_count") != 1 or rec.get("machine_shape") != "NvidiaTeslaT4" or not isinstance(rec.get("kernel_id"), str) or not rec["kernel_id"]:
                raise RegistryError("recovery workload must bind one accelerator and exact kernel")
            if rec["kernel_id"] != amendment["historical_overrun_evidence"]["kernel_id"]:
                raise RegistryError("recovery workload changed the reconciled Kaggle kernel")
            if not isinstance(rec.get("workload_output_bytes_cap"), int) or isinstance(rec["workload_output_bytes_cap"], bool) or not 0 < rec["workload_output_bytes_cap"] <= _RECOVERY_MAX_RUN_OUTPUT_BYTES:
                raise RegistryError("recovery per-run output cap exceeds approved maximum")
            for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256"):
                if not _is_hash(rec.get(key)):
                    raise RegistryError(f"recovery registration has invalid {key}")
            if rec["seed_plan_sha256"] != _RECOVERY_SEED_PLAN_SHA256:
                raise RegistryError("recovery registration uses an unapproved seed-plan hash")
            variants = rec.get("allowed_implementation_variants" if kind == "train" else "implementation_variants")
            if not isinstance(variants, list) or len(variants) != 2 or any(not isinstance(v, dict) for v in variants):
                raise RegistryError("recovery registration must freeze exactly two implementation variants")
            parsed_variants: dict[str, dict[str, str]] = {}
            for variant in variants:
                if set(variant) != {"variant_id", "package_sha256", "source_sha256", "trainer_sha256",
                                    "backend_sha256", "engine_sha256", "config_sha256", "seed_plan_sha256"}:
                    raise RegistryError("implementation variant must bind all exact source and runtime hashes")
                variant_id = variant["variant_id"]
                if variant_id not in {"incumbent", "candidate"} or variant_id in parsed_variants:
                    raise RegistryError("implementation variants must be incumbent and candidate exactly once")
                if any(not _is_hash(variant[k]) for k in set(variant) - {"variant_id"}):
                    raise RegistryError("implementation variant contains an invalid SHA-256")
                if variant["seed_plan_sha256"] != _RECOVERY_SEED_PLAN_SHA256 or variant["config_sha256"] != rec["config_sha256"]:
                    raise RegistryError("implementation variant changes frozen seed plan or workload config")
                parsed_variants[variant_id] = {k: v for k, v in variant.items() if k != "variant_id"}
            variant_sets[kind] = parsed_variants
            id_to_index[rid] = i
            registrations[i] = {"event": event, "record": rec, "kind": kind}
            seen_run_ids.add(rule["run_uuid"])
            if set(variant_sets) == {"benchmark", "train"}:
                shared_hashes = ("source_sha256", "trainer_sha256", "backend_sha256", "engine_sha256",
                                 "config_sha256", "seed_plan_sha256")
                for variant_id in ("incumbent", "candidate"):
                    if any(variant_sets["benchmark"][variant_id][key] != variant_sets["train"][variant_id][key]
                           for key in shared_hashes):
                        raise RegistryError("benchmark and training registrations differ in shared implementation hashes")
        elif typ in {"workload_preflight", "workload_push_dispatched", "workload_push_accepted", "workload_attempt"}:
            attempt_id = data["attempt_id"]
            if typ == "workload_preflight":
                if attempt_id in attempt_state:
                    raise RegistryError("attempt_id cannot be reused")
                ri = data["registration_event_index"]
                if ri not in registrations or registrations[ri]["event"]["event_hash"] != data["registration_event_hash"]:
                    raise RegistryError("preflight does not reference an authorized prior workload")
                registration = registrations[ri]
                if data["expected_current_version"] != current_kernel_version:
                    raise RegistryError("preflight current version differs from exact reconciled kernel head")
                if registration["kind"] == "benchmark" and lock_index is not None and i > lock_index:
                    raise RegistryError("benchmark attempts must be pre-lock")
                if registration["kind"] == "train" and lock_index is None:
                    raise RegistryError("training attempts require registration lock")
                if registration["kind"] == "train" and (
                        recovery_artifact_data is None or recovery_attestation_index is None
                        or recovery_attestation_index >= i):
                    raise RegistryError("training attempts require an attested benchmark selection decision")
                if registration["kind"] == "train" and data["lock_head_hash"] != events[lock_index]["event_hash"]:
                    raise RegistryError("training preflight must pin the exact registration lock")
                if registration["kind"] == "benchmark" and data["lock_head_hash"] is not None:
                    raise RegistryError("benchmark preflight cannot pin a future lock")
                for key in ("kernel_id", "session_timeout_seconds"):
                    expected = registration["record"].get(key)
                    if data[key] != expected:
                        raise RegistryError(f"preflight {key} differs from immutable registration")
                if registration["kind"] == "train":
                    variant_id = data["implementation_variant_id"]
                    selected_variant = variant_sets["train"].get(variant_id)
                    if selected_variant is None:
                        raise RegistryError("training attempt selected an unregistered implementation variant")
                    if variant_id != recovery_artifact_data["selection_evidence"]["selected_variant_id"]:
                        raise RegistryError("training preflight must bind the benchmark-selected implementation variant")
                    for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256"):
                        if data[key] != selected_variant[key]:
                            raise RegistryError(f"training preflight {key} differs from selected frozen variant")
                else:
                    if data["implementation_variant_id"] != "paired":
                        raise RegistryError("benchmark lifecycle must identify paired implementation execution")
                    for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256"):
                        if data[key] != registration["record"][key]:
                            raise RegistryError(f"benchmark preflight {key} differs from immutable registration")
                if data["expected_current_version"] is not None and (not isinstance(data["expected_current_version"], int) or isinstance(data["expected_current_version"], bool) or data["expected_current_version"] < 0):
                    raise RegistryError("invalid exact current version in preflight")
                if data["reconciliation_slack_seconds"] > 600:
                    raise RegistryError("preflight reconciliation slack exceeds 10 minutes")
                if data["reserved_gpu_seconds"] != data["session_timeout_seconds"]:
                    raise RegistryError("preflight does not reserve full timeout")
                remaining_quota = data["quota_evidence"]["remaining_gpu_seconds"]
                future_train_budget = _RECOVERY_TRAIN_GPU_SECONDS if registration["kind"] == "benchmark" else 0
                if remaining_quota < data["reserved_gpu_seconds"] + future_train_budget + 2 * _GPU_HOUR:
                    raise RegistryError("preflight quota cannot cover reservation and protected future/final reserve")
                expected_prior = []
                for old_attempt_id in attempt_order:
                    old = attempt_state[old_attempt_id]
                    if old["registration_index"] == ri:
                        if old["phase"] != "terminal":
                            raise RegistryError("prior attempt has not reached terminal resolution")
                        expected_prior.append({"attempt_id": old_attempt_id,
                                               "terminal_event_index": old["terminal_index"],
                                               "terminal_event_hash": old["terminal_hash"]})
                if data["prior_attempts"] != expected_prior:
                    raise RegistryError("preflight does not bind every prior terminal attempt")
                if any(s["phase"] != "terminal" for s in attempt_state.values()):
                    raise RegistryError("pending attempt blocks a new preflight")
                state = {"phase": "preflight", "preflight_index": i, "preflight_hash": event["event_hash"],
                         "registration_index": ri, "registration_hash": data["registration_event_hash"], "data": data}
                attempt_state[attempt_id] = state
                attempt_order.append(attempt_id)
            else:
                state = attempt_state.get(attempt_id)
                if state is None:
                    raise RegistryError("attempt lifecycle event has no preflight")
                if state["phase"] == "preflight":
                    required = "workload_push_dispatched"
                elif state["phase"] == "dispatched":
                    required = "workload_push_accepted" if typ == "workload_push_accepted" else "workload_attempt"
                elif state["phase"] == "accepted":
                    required = "workload_attempt"
                else:
                    raise RegistryError("attempt already has a terminal resolution")
                predecessor_index = state.get("last_index", state["preflight_index"])
                if typ != required or i != predecessor_index + 1:
                    raise RegistryError("attempt lifecycle events must be contiguous and correctly ordered")
                for key in ("registration_event_index", "registration_event_hash"):
                    expected = state["data"][key]
                    if data[key] != expected:
                        raise RegistryError("attempt lifecycle event changed registration binding")
                if data["lock_head_hash"] != state["data"]["lock_head_hash"]:
                    raise RegistryError("attempt lifecycle event changed lock binding")
                if typ == "workload_push_dispatched":
                    for key in ("preflight_event_index", "preflight_event_hash"):
                        expected = state["preflight_index"] if key.endswith("index") else state["preflight_hash"]
                        if data[key] != expected:
                            raise RegistryError("dispatch does not link exact preflight")
                    for key in ("kernel_id", "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256", "session_timeout_seconds", "accelerator_count", "reserved_gpu_seconds", "implementation_variant_id"):
                        if data[key] != state["data"][key]:
                            raise RegistryError("dispatch changed frozen preflight workload")
                    state.update(phase="dispatched", dispatch_index=i, dispatch_hash=event["event_hash"], last_index=i)
                elif typ == "workload_push_accepted":
                    links = {"preflight_event_index": state["preflight_index"], "preflight_event_hash": state["preflight_hash"],
                             "dispatch_event_index": state["dispatch_index"], "dispatch_event_hash": state["dispatch_hash"]}
                    if any(data[k] != v for k, v in links.items()):
                        raise RegistryError("accepted receipt does not link exact dispatch/preflight")
                    if data["implementation_variant_id"] != state["data"]["implementation_variant_id"]:
                        raise RegistryError("accepted receipt changed implementation variant")
                    if data["confirming_metadata"]["kernel_id"] != state["data"]["kernel_id"]:
                        raise RegistryError("accepted metadata identifies a different kernel")
                    if data["assigned_version"] <= state["data"]["expected_current_version"]:
                        raise RegistryError("accepted push must assign a new version beyond preflight metadata")
                    state.update(phase="accepted", accepted_index=i, accepted_hash=event["event_hash"],
                                 assigned_version=data["assigned_version"], last_index=i)
                else:
                    links = {"preflight_event_index": state["preflight_index"], "preflight_event_hash": state["preflight_hash"],
                             "dispatch_event_index": state["dispatch_index"], "dispatch_event_hash": state["dispatch_hash"]}
                    if any(data[k] != v for k, v in links.items()):
                        raise RegistryError("terminal resolution does not link exact dispatch/preflight")
                    if data["implementation_variant_id"] != state["data"]["implementation_variant_id"]:
                        raise RegistryError("terminal resolution changed implementation variant")
                    if data["outcome"] == "terminal" and data["terminal_metadata"]["kernel_id"] != state["data"]["kernel_id"]:
                        raise RegistryError("terminal metadata identifies a different kernel")
                    for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256"):
                        if data[key] != state["data"][key]:
                            raise RegistryError(f"terminal resolution changed frozen {key}")
                    reg = registrations[state["registration_index"]]
                    if reg["kind"] == "train":
                        variant = variant_sets["train"][state["data"]["implementation_variant_id"]]
                        if data["trainer_sha256"] != variant["trainer_sha256"] or data["engine_sha256"] != variant["engine_sha256"]:
                            raise RegistryError("terminal resolution changed trainer/engine hashes")
                    if state["phase"] == "dispatched":
                        if data["outcome"] != "pre_assignment_rejected":
                            raise RegistryError("dispatched push requires accepted receipt or verified pre-assignment rejection")
                        no_assignment = data["no_assignment_metadata"]
                        if (data["rejection_evidence"].get("server_rejected_before_assignment") is not True
                                or no_assignment.get("no_version_assigned") is not True
                                or no_assignment.get("pre_push_current_version") != state["data"]["expected_current_version"]
                                or no_assignment.get("post_push_current_version") != state["data"]["expected_current_version"]):
                            raise RegistryError("pre-assignment rejection lacks exact no-version-assigned proof")
                    elif state["phase"] == "accepted":
                        if data["outcome"] != "terminal" or data["accepted_event_index"] != state["accepted_index"] or data["accepted_event_hash"] != state["accepted_hash"] or data["assigned_version"] != state["assigned_version"]:
                            raise RegistryError("terminal resolution differs from accepted receipt")
                    if reg["kind"] == "benchmark" and lock_index is not None and i > lock_index:
                        raise RegistryError("benchmark attempt resolution must precede lock")
                    if reg["kind"] == "train" and data["lock_head_hash"] != events[lock_index]["event_hash"]:
                        raise RegistryError("training attempt must bind exact lock")
                    if data["actual_gpu_seconds"] > reg["record"]["session_timeout_seconds"]:
                        raise RegistryError("actual GPU use exceeds frozen attempt timeout")
                    if data["output_bytes"] > reg["record"]["workload_output_bytes_cap"]:
                        raise RegistryError("attempt output exceeds frozen per-run cap")
                    state.update(phase="terminal", terminal_index=i, terminal_hash=event["event_hash"],
                                 actual_gpu_seconds=data["actual_gpu_seconds"], output_bytes=data["output_bytes"])
                    if data["outcome"] == "terminal":
                        current_kernel_version = data["assigned_version"]
                    else:
                        current_kernel_version = data["no_assignment_metadata"]["post_push_current_version"]
        elif typ == "benchmark_artifact":
            if lock_index is not None and i >= lock_index:
                raise RegistryError("recovery benchmark artifact must precede lock")
            # The recovery artifact must identify the exact successful benchmark attempt.
            if "attempt_event_index" not in data or "attempt_event_hash" not in data:
                raise RegistryError("recovery benchmark artifact must bind a terminal attempt")
            ai = data["attempt_event_index"]
            if not isinstance(ai, int) or isinstance(ai, bool) or ai < 0 or ai >= i:
                raise RegistryError("invalid benchmark attempt reference")
            attempt = events[ai]
            if attempt["event_type"] != "workload_attempt" or attempt["event_data"]["outcome"] != "terminal" or attempt["event_data"]["terminal_status"] != "COMPLETE" or attempt["event_hash"] != data["attempt_event_hash"]:
                raise RegistryError("benchmark artifact must bind an exact successful terminal attempt")
            ri = attempt["event_data"]["registration_event_index"]
            if ri not in registrations or registrations[ri]["kind"] != "benchmark":
                raise RegistryError("benchmark artifact attempt is not from registered benchmark workload")
            if data["registration_event_index"] != ri or data["registration_event_hash"] != registrations[ri]["event"]["event_hash"]:
                raise RegistryError("benchmark artifact registration binding differs from attempt")
            if "selection_evidence" not in data:
                raise RegistryError("recovery benchmark artifact must bind the paired selection decision")
            if "train" not in variant_sets or data["selection_evidence"]["selected_variant_id"] not in variant_sets["train"]:
                raise RegistryError("benchmark selected variant is not in the frozen training registration")
                if any(state["phase"] != "terminal" for state in attempt_state.values()):
                    raise RegistryError("benchmark artifact cannot be recorded while an attempt is pending")
                if actual_benchmark := [state for state in attempt_state.values()
                                        if registrations[state["registration_index"]]["kind"] == "benchmark"]:
                    if not any(state.get("terminal_index") == ai for state in actual_benchmark):
                        raise RegistryError("benchmark artifact attempt is not in the recovery lifecycle")
            recovery_artifact_index = i
            recovery_artifact_data = data
        elif typ == "registration_closed":
            if recovery_artifact_index is None or i != recovery_artifact_index + 1:
                raise RegistryError("recovery benchmark artifact must immediately precede lock")
            if data != {"pre_lock_head_hash": events[i-1]["event_hash"], "pre_lock_final_event_index": i-1}:
                raise RegistryError("recovery lock must pin the exact artifact prefix")
        elif typ == "benchmark_attestation":
            if recovery_artifact_index is None or lock_index is None:
                raise RegistryError("recovery attestation requires the recovery artifact and lock")
            if (i != recovery_artifact_index + 2
                    or data != {**recovery_artifact_data, "lock_head_hash": events[lock_index]["event_hash"]}):
                raise RegistryError("recovery attestation must copy the exact artifact decision and lock binding")
            recovery_attestation_index = i
        else:
            raise RegistryError(f"event {typ!r} is not allowed after the recovery amendment")

    if len(id_to_index) > 2:
        raise RegistryError("recovery permits exactly two workload registrations")
    unresolved = [state for state in attempt_state.values() if state["phase"] != "terminal"]
    if unresolved:
        latest_unresolved = max(state.get("last_index", state["preflight_index"]) for state in unresolved)
        if latest_unresolved != len(events) - 1:
            raise RegistryError("pending recovery attempt blocks all later events")
        # At most one unresolved lifecycle can exist; it holds its entire reservation.
        if len(unresolved) != 1:
            raise RegistryError("pending recovery attempt blocks a second preflight")
    # Enforce cumulative actual + outstanding reservations within the frozen recovery caps.
    actual = {"benchmark": 0, "train": 0}
    reserved = {"benchmark": 0, "train": 0}
    output_total = 0
    reserved_output_total = 0
    for attempt_id, state in attempt_state.items():
        record = registrations[state["registration_index"]]["record"]
        kind = registrations[state["registration_index"]]["kind"]
        if state["phase"] == "terminal":
            actual[kind] += state["actual_gpu_seconds"]
            output_total += state["output_bytes"]
        else:
            reserved[kind] += state["data"]["session_timeout_seconds"]
            reserved_output_total += record["workload_output_bytes_cap"]
    for kind, cap in (("benchmark", _RECOVERY_BENCHMARK_GPU_SECONDS), ("train", _RECOVERY_TRAIN_GPU_SECONDS)):
        if actual[kind] + reserved[kind] > cap:
            raise RegistryError(f"cumulative {kind} GPU use/reservation exceeds approved cap")
    amendment_reservation = amendment["reservation_metadata"]
    if amendment_reservation["campaign_gpu_seconds_before"] + sum(actual.values()) + sum(reserved.values()) > 6 * _GPU_HOUR:
        raise RegistryError("campaign GPU usage and pending reservation exceed experiment allocation")
    if amendment_reservation["campaign_output_bytes_before"] + output_total + reserved_output_total > _RECOVERY_MAX_OUTPUT_BYTES:
        raise RegistryError("recovery output plus pending output reservation exceeds aggregate cap")


def _validate_recovery_cycle2_suffix(events: list[dict[str, Any]]) -> None:
    """Validate the separately authorized cycle-2 suffix without reinterpreting cycle 1."""
    cycle_events = [i for i, event in enumerate(events)
                    if event["event_type"] == "recovery_cycle_amendment"]
    if cycle_events != [_CYCLE2_EVENT_INDEX]:
        raise RegistryError("cycle-2 amendment must be the sole amendment at immutable index 20")
    if len(events) < _CYCLE2_EVENT_INDEX + 1:
        raise RegistryError("cycle-2 amendment is missing its immutable cycle-1 parent")
    if events[19]["event_hash"] != _CYCLE1_FINAL_EVENT_HASH:
        raise RegistryError("cycle-2 amendment requires exact terminal cycle-1 head")
    if events[19]["event_type"] != "workload_attempt":
        raise RegistryError("cycle-2 parent event 19 must be the terminal cycle-1 attempt")
    first_amendment = events[13]["event_data"]
    terminal_attempt = events[19]["event_data"]
    if (terminal_attempt["actual_gpu_seconds"] != 125 or terminal_attempt["output_bytes"] != 4519
            or first_amendment["reservation_metadata"]["campaign_gpu_seconds_before"] != 8424
            or first_amendment["reservation_metadata"]["campaign_output_bytes_before"] != 26403):
        raise RegistryError("cycle-1 terminal resource use differs from frozen historical evidence")
    amendment_event = events[_CYCLE2_EVENT_INDEX]
    amendment = amendment_event["event_data"]
    if (amendment["prior_head_hash"] != events[19]["event_hash"]
            or amendment["prior_final_event_index"] != 19):
        raise RegistryError("cycle-2 amendment does not bind exact event-19 parent")
    if (amendment["project_gpu_seconds_before"] != first_amendment["reservation_metadata"]["campaign_gpu_seconds_before"] + terminal_attempt["actual_gpu_seconds"]
            or amendment["project_output_bytes_before"] != first_amendment["reservation_metadata"]["campaign_output_bytes_before"] + terminal_attempt["output_bytes"]):
        raise RegistryError("cycle-2 baseline must count event-19 GPU/output use exactly once")
    if (amendment["kernel_id"] != terminal_attempt["terminal_metadata"]["kernel_id"]
            or amendment["current_version_number"] != terminal_attempt["assigned_version"]
            or amendment["current_terminal_status"] != terminal_attempt["terminal_status"]
            or amendment["current_status_evidence"] != {"status": terminal_attempt["terminal_status"],
                                                         "current_version_number": terminal_attempt["assigned_version"],
                                                         "kernel_id": terminal_attempt["terminal_metadata"]["kernel_id"]}):
        raise RegistryError("cycle-2 amendment does not bind exact terminal event-19 status")

    named = {row["registration_id"]: row for row in amendment["recovery_registrations"]}
    registrations: dict[int, dict[str, Any]] = {}
    by_id: dict[str, int] = {}
    variants_by_kind: dict[str, dict[str, dict[str, str]]] = {}
    attempts: dict[str, dict[str, Any]] = {}
    attempt_order: list[str] = []
    actual_gpu = {"benchmark": 0, "train": 0}
    reserved_gpu = {"benchmark": 0, "train": 0}
    actual_output = {"benchmark": 0, "training": 0, "validation": 0, "final": 0}
    reserved_output = dict(actual_output)
    artifact_index: int | None = None
    artifact_data: dict[str, Any] | None = None
    lock_index: int | None = None
    attestation_index: int | None = None
    attested = False
    current_version = amendment["current_version_number"]

    for i, event in enumerate(events[_CYCLE2_EVENT_INDEX + 1:], _CYCLE2_EVENT_INDEX + 1):
        typ, data = event["event_type"], event["event_data"]
        if typ == "recovery_cycle_amendment":
            raise RegistryError("a second cycle amendment is forbidden")
        if typ == "registration":
            if i > _CYCLE2_EVENT_INDEX + 2 or len(registrations) >= 2:
                raise RegistryError("cycle 2 permits exactly two consecutive workload registrations")
            expected_kind = "benchmark" if i == _CYCLE2_EVENT_INDEX + 1 else "train"
            if data["registration_kind"] != expected_kind:
                raise RegistryError("cycle-2 registration order must be benchmark then train")
            rid = data["registration_id"]
            if rid not in named or rid in by_id:
                raise RegistryError("cycle-2 amendment authorizes exactly its two named registrations")
            if data.get("cycle_id") != "g008-recovery-cycle-2":
                raise RegistryError("cycle-2 registration must identify its cycle")
            rule = named[rid]
            record = data["record"]
            kind = data["registration_kind"]
            if kind != rule["registration_kind"] or record.get("run_uuid") != rule["run_uuid"]:
                raise RegistryError("cycle-2 registration differs from amended identity")
            old_ids = {e["event_data"]["registration_id"] for e in events[:20]
                       if e["event_type"] == "registration"}
            old_uuids = {e["event_data"]["record"].get("run_uuid") for e in events[:20]
                         if e["event_type"] == "registration"}
            if rid in old_ids or rule["run_uuid"] in old_uuids:
                raise RegistryError("cycle-2 registration reuses prior registration identity")
            if (record.get("cycle_id") != "g008-recovery-cycle-2"
                    or record.get("recovery_cycle_amendment_event_index") != _CYCLE2_EVENT_INDEX
                    or record.get("recovery_cycle_amendment_event_hash") != amendment_event["event_hash"]):
                raise RegistryError("cycle-2 registration does not pin its exact amendment")
            cap = _CYCLE2_BENCHMARK_GPU_SECONDS if kind == "benchmark" else _CYCLE2_TRAIN_GPU_SECONDS
            if record.get("workload_gpu_seconds_cap") != cap:
                raise RegistryError("cycle-2 workload GPU cap differs from approved proposal")
            if (not isinstance(record.get("session_timeout_seconds"), int)
                    or isinstance(record.get("session_timeout_seconds"), bool)
                    or not 0 < record["session_timeout_seconds"] <= cap):
                raise RegistryError("cycle-2 timeout must be positive and within its per-cycle cap")
            expected_output_cap = (_CYCLE2_BENCHMARK_ATTEMPT_OUTPUT_CAP if kind == "benchmark"
                                   else _CYCLE2_OUTPUT_CATEGORY_CAP)
            if record.get("workload_output_bytes_cap") != expected_output_cap:
                raise RegistryError("cycle-2 workload output cap differs from reserved category")
            if (record.get("accelerator_count") != 1 or record.get("machine_shape") != "NvidiaTeslaT4"
                    or not isinstance(record.get("kernel_id"), str) or not record["kernel_id"]):
                raise RegistryError("cycle-2 registration must bind one T4 and exact kernel")
            if record["kernel_id"] != amendment["kernel_id"]:
                raise RegistryError("cycle-2 registration changed the reconciled Kaggle kernel")
            expected_timeout = 2800 if kind == "benchmark" else 7200
            if record["session_timeout_seconds"] != expected_timeout:
                raise RegistryError("cycle-2 session timeout differs from approved one-T4 bound")
            for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                        "backend_sha256", "trainer_sha256", "engine_sha256"):
                if not _is_hash(record.get(key)):
                    raise RegistryError(f"cycle-2 registration has invalid {key}")
            if record["seed_plan_sha256"] != amendment["seed_plan_sha256"]:
                raise RegistryError("cycle-2 registration changed the approved seed plan")
            variant_rows = record.get("allowed_implementation_variants" if kind == "train" else "implementation_variants")
            if not isinstance(variant_rows, list) or len(variant_rows) != 2:
                raise RegistryError("cycle-2 registration must freeze incumbent and candidate variants")
            parsed: dict[str, dict[str, str]] = {}
            variant_keys = {"variant_id", "package_sha256", "source_sha256", "trainer_sha256",
                            "backend_sha256", "engine_sha256", "config_sha256", "seed_plan_sha256"}
            for variant in variant_rows:
                if not isinstance(variant, dict) or set(variant) != variant_keys:
                    raise RegistryError("cycle-2 variant must bind all implementation and runtime hashes")
                variant_id = variant["variant_id"]
                if variant_id not in {"incumbent", "candidate"} or variant_id in parsed:
                    raise RegistryError("cycle-2 variants must be incumbent and candidate exactly once")
                if any(not _is_hash(variant[key]) for key in variant_keys - {"variant_id"}):
                    raise RegistryError("cycle-2 variant contains an invalid SHA-256")
                if variant["seed_plan_sha256"] != amendment["seed_plan_sha256"] or variant["config_sha256"] != record["config_sha256"]:
                    raise RegistryError("cycle-2 variant changed frozen seed plan or workload config")
                parsed[variant_id] = {k: v for k, v in variant.items() if k != "variant_id"}
            variants_by_kind[kind] = parsed
            if set(variants_by_kind) == {"benchmark", "train"}:
                shared = ("source_sha256", "trainer_sha256", "backend_sha256", "engine_sha256",
                          "config_sha256", "seed_plan_sha256")
                for variant_id in ("incumbent", "candidate"):
                    if any(variants_by_kind["benchmark"][variant_id][key] != variants_by_kind["train"][variant_id][key]
                           for key in shared):
                        raise RegistryError("cycle-2 benchmark/train registrations differ in shared variant hashes")
            index = i
            registrations[index] = {"event": event, "record": record, "kind": kind}
            by_id[rid] = index
            continue

        if typ in {"workload_preflight", "workload_push_dispatched", "workload_push_accepted", "workload_attempt"}:
            if len(registrations) != 2:
                raise RegistryError("both cycle-2 registrations must precede workload execution")
            if data.get("cycle_id") != "g008-recovery-cycle-2":
                raise RegistryError("cycle-2 attempts must identify their cycle")
            attempt_id = data["attempt_id"]
            if typ == "workload_preflight":
                if attempt_id in attempts:
                    raise RegistryError("cycle-2 attempt_id cannot be reused")
                if any(state["phase"] != "terminal" for state in attempts.values()):
                    raise RegistryError("unresolved cycle-2 attempt blocks a new preflight")
                reg_index = data["registration_event_index"]
                registration = registrations.get(reg_index)
                if registration is None or registration["event"]["event_hash"] != data["registration_event_hash"]:
                    raise RegistryError("cycle-2 preflight references no current-cycle registration")
                kind = registration["kind"]
                if reg_index in amendment["retired_registration_event_indices"]:
                    raise RegistryError("cycle-1 registrations are retired for future execution")
                if data["expected_current_version"] != current_version:
                    raise RegistryError("cycle-2 preflight does not bind current Kaggle version")
                if kind == "train" and (not attested or artifact_index is None or lock_index is None):
                    raise RegistryError("cycle-2 training requires benchmark artifact, lock, and attestation")
                if kind == "benchmark" and (artifact_index is not None or lock_index is not None):
                    raise RegistryError("cycle-2 benchmark attempts must precede artifact and lock")
                if kind == "train" and data["lock_head_hash"] != events[lock_index]["event_hash"]:
                    raise RegistryError("cycle-2 training preflight must bind exact lock")
                if kind == "benchmark" and data["lock_head_hash"] is not None:
                    raise RegistryError("cycle-2 benchmark preflight cannot bind future lock")
                for key in ("kernel_id", "session_timeout_seconds"):
                    if data[key] != registration["record"][key]:
                        raise RegistryError(f"cycle-2 preflight {key} differs from immutable registration")
                if kind == "benchmark":
                    if data["implementation_variant_id"] != "paired":
                        raise RegistryError("cycle-2 benchmark must use paired implementation mode")
                    for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                                "backend_sha256", "trainer_sha256", "engine_sha256"):
                        if data[key] != registration["record"][key]:
                            raise RegistryError(f"cycle-2 benchmark preflight changed registered {key}")
                else:
                    variant = variants_by_kind["train"].get(data["implementation_variant_id"])
                    if variant is None:
                        raise RegistryError("cycle-2 training selected an unregistered implementation variant")
                    if variant_id := (artifact_data or {}).get("selection_evidence", {}).get("selected_variant_id"):
                        if data["implementation_variant_id"] != variant_id:
                            raise RegistryError("cycle-2 training differs from benchmark-selected variant")
                    for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                                "backend_sha256", "trainer_sha256", "engine_sha256"):
                        if data[key] != variant[key]:
                            raise RegistryError(f"cycle-2 training preflight changed frozen {key}")
                if data["reserved_gpu_seconds"] != registration["record"]["session_timeout_seconds"]:
                    raise RegistryError("cycle-2 preflight must reserve full timeout")
                if data["implementation_variant_id"] not in {"paired", "incumbent", "candidate"}:
                    raise RegistryError("cycle-2 preflight has invalid implementation variant")
                state = {"phase": "preflight", "registration_index": reg_index, "kind": kind,
                         "data": data, "preflight_index": i, "preflight_hash": event["event_hash"],
                         "last_index": i}
                attempts[attempt_id] = state
                attempt_order.append(attempt_id)
                reserved_gpu[kind] += data["reserved_gpu_seconds"]
                category = "benchmark" if kind == "benchmark" else "training"
                reserved_output[category] += registration["record"]["workload_output_bytes_cap"]
                if actual_output[category] + reserved_output[category] > amendment["output_category_reservations_bytes"][category]:
                    raise RegistryError(f"cycle-2 {category} actual plus retry output reservation exceeds category cap")
                remaining_quota = data["quota_evidence"]["remaining_gpu_seconds"]
                planned_benchmark = max(0, _CYCLE2_BENCHMARK_GPU_SECONDS - actual_gpu["benchmark"]
                                        - reserved_gpu["benchmark"])
                planned_train = max(0, _CYCLE2_TRAIN_GPU_SECONDS - actual_gpu["train"] - reserved_gpu["train"])
                if remaining_quota < planned_benchmark + planned_train + 7200:
                    raise RegistryError("cycle-2 live quota cannot cover remaining cycle caps and protected reserve")
            else:
                state = attempts.get(attempt_id)
                if state is None or state["phase"] == "terminal" or i != state["last_index"] + 1:
                    raise RegistryError("cycle-2 attempt lifecycle must be contiguous and unreused")
                if data["registration_event_index"] != state["registration_index"] or data["registration_event_hash"] != registrations[state["registration_index"]]["event"]["event_hash"]:
                    raise RegistryError("cycle-2 lifecycle changed registration binding")
                if data.get("cycle_id") != "g008-recovery-cycle-2":
                    raise RegistryError("cycle-2 lifecycle changed cycle identity")
                state["last_index"] = i
                if typ == "workload_push_dispatched":
                    if state["phase"] != "preflight" or data["preflight_event_index"] != state["preflight_index"] or data["preflight_event_hash"] != state["preflight_hash"]:
                        raise RegistryError("cycle-2 dispatch must bind exact preflight")
                    for key in ("kernel_id", "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                                "backend_sha256", "trainer_sha256", "engine_sha256", "session_timeout_seconds",
                                "accelerator_count", "reserved_gpu_seconds", "implementation_variant_id", "lock_head_hash"):
                        if data[key] != state["data"][key]:
                            raise RegistryError("cycle-2 dispatch changed preflight workload bytes/resources")
                    state.update(phase="dispatched", dispatch_index=i, dispatch_hash=event["event_hash"])
                elif typ == "workload_push_accepted":
                    if state["phase"] != "dispatched" or data["dispatch_event_index"] != state["dispatch_index"] or data["dispatch_event_hash"] != state["dispatch_hash"]:
                        raise RegistryError("cycle-2 acceptance must bind exact dispatch")
                    if (data["preflight_event_index"] != state["preflight_index"]
                            or data["preflight_event_hash"] != state["preflight_hash"]
                            or data["implementation_variant_id"] != state["data"]["implementation_variant_id"]
                            or data["lock_head_hash"] != state["data"]["lock_head_hash"]):
                        raise RegistryError("cycle-2 acceptance changed preflight/variant binding")
                    if data["assigned_version"] <= state["data"]["expected_current_version"]:
                        raise RegistryError("cycle-2 acceptance must assign a new kernel version")
                    state.update(phase="accepted", accepted_index=i, accepted_hash=event["event_hash"],
                                 assigned_version=data["assigned_version"])
                else:
                    if state["phase"] not in {"dispatched", "accepted"}:
                        raise RegistryError("cycle-2 terminal attempt lacks dispatched lifecycle")
                    if (state["phase"] == "accepted" and
                            (data["outcome"] != "terminal"
                             or data["accepted_event_index"] != state["accepted_index"]
                             or data["accepted_event_hash"] != state["accepted_hash"]
                             or data["assigned_version"] != state["assigned_version"])):
                        raise RegistryError("cycle-2 terminal attempt changed accepted receipt")
                    if state["phase"] == "dispatched" and data["outcome"] != "pre_assignment_rejected":
                        raise RegistryError("cycle-2 dispatched attempt needs accepted receipt or proven rejection")
                    if state["phase"] == "dispatched":
                        no_assignment = data["no_assignment_metadata"]
                        if (data["rejection_evidence"].get("server_rejected_before_assignment") is not True
                                or no_assignment.get("no_version_assigned") is not True
                                or no_assignment.get("pre_push_current_version") != state["data"]["expected_current_version"]
                                or no_assignment.get("post_push_current_version") != state["data"]["expected_current_version"]):
                            raise RegistryError("cycle-2 pre-assignment rejection lacks exact no-version-assigned proof")
                    if (data["preflight_event_index"] != state["preflight_index"]
                            or data["preflight_event_hash"] != state["preflight_hash"]
                            or data["dispatch_event_index"] != state["dispatch_index"]
                            or data["dispatch_event_hash"] != state["dispatch_hash"]):
                        raise RegistryError("cycle-2 terminal attempt changed preflight/dispatch binding")
                    for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                                "backend_sha256", "trainer_sha256", "engine_sha256", "implementation_variant_id",
                                "lock_head_hash"):
                        if data[key] != state["data"][key]:
                            raise RegistryError("cycle-2 terminal attempt changed frozen registered workload")
                    if data["actual_gpu_seconds"] > registrations[state["registration_index"]]["record"]["session_timeout_seconds"]:
                        raise RegistryError("cycle-2 actual GPU use exceeds attempt timeout")
                    category = "benchmark" if state["kind"] == "benchmark" else "training"
                    reserved_gpu[state["kind"]] -= state["data"]["reserved_gpu_seconds"]
                    reserved_output[category] -= registrations[state["registration_index"]]["record"]["workload_output_bytes_cap"]
                    actual_gpu[state["kind"]] += data["actual_gpu_seconds"]
                    actual_output[category] += data["output_bytes"]
                    if data["output_bytes"] > registrations[state["registration_index"]]["record"]["workload_output_bytes_cap"]:
                        raise RegistryError("cycle-2 output exceeds immutable per-run cap")
                    if data["outcome"] == "terminal":
                        current_version = data["assigned_version"]
                    state.update(phase="terminal", terminal_index=i, actual_gpu_seconds=data["actual_gpu_seconds"],
                                 output_bytes=data["output_bytes"], event_data=data)
            continue

        if typ == "benchmark_artifact":
            if artifact_index is not None or lock_index is not None:
                raise RegistryError("cycle-2 benchmark artifact/lock cannot be reused")
            if not any(state["kind"] == "benchmark" and state["phase"] == "terminal"
                       and state.get("terminal_index") == data.get("attempt_event_index")
                       and state["event_data"].get("terminal_status") == "COMPLETE"
                       and events[state["terminal_index"]]["event_hash"] == data.get("attempt_event_hash")
                       for state in attempts.values()):
                raise RegistryError("cycle-2 artifact requires successful cycle-2 benchmark attempt")
            if any(state["phase"] != "terminal" for state in attempts.values()):
                raise RegistryError("cycle-2 artifact cannot be recorded while an attempt is unresolved")
            artifact_index, artifact_data = i, data
            continue
        if typ == "registration_closed":
            if artifact_index is None or i != artifact_index + 1:
                raise RegistryError("cycle-2 benchmark artifact must immediately precede lock")
            lock_index = i
            continue
        if typ == "benchmark_attestation":
            if lock_index is None or i != lock_index + 1 or artifact_data is None:
                raise RegistryError("cycle-2 attestation must immediately follow lock")
            attested = True
            attestation_index = i
            continue
        raise RegistryError(f"event {typ!r} is not allowed in cycle-2 suffix")

    has_work_started = any(e["event_type"] in {"workload_preflight", "workload_push_dispatched",
                                                "workload_push_accepted", "workload_attempt",
                                                "benchmark_artifact", "registration_closed", "benchmark_attestation"}
                           for e in events[_CYCLE2_EVENT_INDEX + 1:])
    if has_work_started and set(by_id) != set(named):
        raise RegistryError("both cycle-2 registrations must precede execution")
    if len(by_id) > 2:
        raise RegistryError("cycle-2 permits exactly two registrations")
    if any(state["phase"] != "terminal" for state in attempts.values()):
        if not attempts or max(state["last_index"] for state in attempts.values() if state["phase"] != "terminal") != len(events) - 1:
            raise RegistryError("pending cycle-2 attempt blocks all later events")
    for kind, cap in (("benchmark", _CYCLE2_BENCHMARK_GPU_SECONDS), ("train", _CYCLE2_TRAIN_GPU_SECONDS)):
        if actual_gpu[kind] + reserved_gpu[kind] > cap:
            raise RegistryError(f"cycle-2 cumulative {kind} GPU use/reservation exceeds cap")
    for category, cap in amendment["output_category_reservations_bytes"].items():
        if actual_output[category] + reserved_output[category] > cap:
            raise RegistryError(f"cycle-2 {category} output use/reservation exceeds category cap")
    project_gpu = amendment["project_gpu_seconds_before"] + sum(actual_gpu.values()) + sum(reserved_gpu.values())
    if project_gpu > 21600 or project_gpu + 7200 > 28800:
        raise RegistryError("cycle-2 actual/reserved GPU use exceeds project budget")
    # Four category pools are committed exactly once at the cycle boundary. Attempt
    # output replaces usage inside a pool; terminal reconciliation does not add a
    # second copy of event 19 or charge a pool twice.
    project_output = amendment["project_output_bytes_before"] + sum(amendment["output_category_reservations_bytes"].values())
    if project_output > _RECOVERY_MAX_OUTPUT_BYTES:
        raise RegistryError("cycle-2 actual/reserved output exceeds aggregate cap")


class Registry:
    """Safe registry operations over a JSON-lines file (never creates one implicitly)."""
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def events(self, *, expected_head_hash: str | None = None,
               expected_final_event_index: int | None = None) -> list[dict[str, Any]]:
        events = _read(self.path)
        if expected_head_hash is not None and (not events or events[-1]["event_hash"] != expected_head_hash):
            raise RegistryError("registry head differs from independently captured anchor")
        if expected_final_event_index is not None and len(events) - 1 != expected_final_event_index:
            raise RegistryError("registry final event index differs from independently captured anchor")
        return events

    @property
    def head(self) -> tuple[str | None, int]:
        events = self.events()
        return (events[-1]["event_hash"] if events else None, len(events) - 1)

    @property
    def lock_head_hash(self) -> str | None:
        return next((e["event_hash"] for e in self.events() if e["event_type"] == "registration_closed"), None)

    def _append(self, event_type: str, event_data: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            events = _read(self.path)
            _validate_data(event_type, event_data)
            locked = any(e["event_type"] == "registration_closed" for e in events)
            if event_type == "registration" and locked:
                raise RegistryError("cannot register after lock")
            if event_type == "registration" and any(
                    e["event_type"] == "registration" and
                    e["event_data"]["registration_id"] == event_data["registration_id"] for e in events):
                raise RegistryError("duplicate registration_id")
            if event_type == "registration_closed" and locked:
                raise RegistryError("registration lock is irreversible")
            if event_type == "benchmark_artifact" and locked:
                raise RegistryError("benchmark artifact must be recorded pre-lock")
            if event_type in {"workload_completion", "evidence", "benchmark_attestation"} and not locked:
                raise RegistryError(f"{event_type} requires registration_closed")
            if event_type == "benchmark_artifact":
                _verify_benchmark_registration(event_data, events, {})
                if event_data["pre_lock_head_hash"] != (events[-1]["event_hash"] if events else None) or event_data["pre_lock_final_event_index"] != len(events)-1:
                    raise RegistryError("benchmark artifact prefix anchor is stale")
            if event_type == "benchmark_attestation":
                reg_i = event_data["registration_event_index"]
                artifacts = [e["event_data"] for e in events if e["event_type"] == "benchmark_artifact"
                             and e["event_data"]["registration_event_index"] == reg_i]
                locks = [e["event_hash"] for e in events if e["event_type"] == "registration_closed"]
                if not artifacts or event_data != {**artifacts[0], "lock_head_hash": locks[0]}:
                    raise RegistryError("benchmark attestation mismatch")
                if any(e["event_type"] == "benchmark_attestation" and
                       e["event_data"]["registration_event_index"] == reg_i for e in events):
                    raise RegistryError("duplicate benchmark attestation")
            if event_type in {"workload_completion", "evidence"}:
                reg_i = event_data["registration_event_index"]
                if reg_i >= len(events) or reg_i < 0 or events[reg_i]["event_type"] != "registration":
                    raise RegistryError("post-lock evidence must reference a prior registration")
                if event_data["registration_event_hash"] != events[reg_i]["event_hash"]:
                    raise RegistryError("post-lock evidence registration hash mismatch")
                lock_hash = next(e["event_hash"] for e in events if e["event_type"] == "registration_closed")
                if event_data["lock_head_hash"] != lock_hash:
                    raise RegistryError("post-lock evidence lock hash mismatch")
            # Validate proposed event by replaying lifecycle rules after append.
            prev = events[-1]["event_hash"] if events else GENESIS
            payload = _payload(len(events), event_type, event_data, prev)
            event = {**payload, "event_hash": _event_hash(payload)}
            candidate_events = events + [event]
            if any(e["event_type"] == "recovery_cycle_amendment" for e in candidate_events):
                _validate_recovery_suffix(candidate_events[:_CYCLE2_EVENT_INDEX])
                _validate_recovery_cycle2_suffix(candidate_events)
            else:
                _validate_recovery_suffix(candidate_events)
            encoded = canonical_bytes(event) + b"\n"
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                written = os.write(fd, encoded)
                if written != len(encoded):
                    raise RegistryError("short append; registry may require recovery")
                os.fsync(fd)
            finally:
                os.close(fd)
            _read(self.path)
            return event

    def register(self, registration_id: str, registration_kind: str, record: dict[str, Any]) -> dict[str, Any]:
        if not registration_id or not isinstance(registration_id, str):
            raise RegistryError("registration_id must be non-empty")
        if not isinstance(record, dict):
            raise RegistryError("record must be an object")
        events = self.events()
        if any(e["event_type"] == "registration" and e["event_data"]["registration_id"] == registration_id for e in events):
            raise RegistryError("duplicate registration_id")
        if any(e["event_type"] == "registration_closed" for e in events):
            raise RegistryError("cannot register after lock")
        return self._append("registration", {"registration_id": registration_id,
                                              "registration_kind": registration_kind, "record": record})

    def register_recovery_amendment(self, amendment: dict[str, Any]) -> dict[str, Any]:
        """Append the single approved amendment after independently checking the exact legacy head."""
        events = self.events(expected_head_hash=_LEGACY_HEAD_HASH,
                             expected_final_event_index=_LEGACY_HEAD_INDEX)
        if events[-1]["event_index"] != _LEGACY_HEAD_INDEX:
            raise RegistryError("recovery amendment requires immutable legacy indices 0 through 12")
        return self._append("recovery_amendment", amendment)

    def register_recovery_cycle_amendment(self, amendment: dict[str, Any]) -> dict[str, Any]:
        """Append the reviewed cycle-2 amendment only after the exact terminal cycle-1 head."""
        events = self.events(expected_head_hash=_CYCLE1_FINAL_EVENT_HASH,
                             expected_final_event_index=_CYCLE1_FINAL_EVENT_INDEX)
        if events[-1]["event_type"] != "workload_attempt":
            raise RegistryError("cycle-2 amendment requires terminal cycle-1 event 19")
        return self._append("recovery_cycle_amendment", amendment)

    def register_recovery_workload(self, registration_id: str, registration_kind: str,
                                   record: dict[str, Any]) -> dict[str, Any]:
        """Append one of the two exact workload identities named by the amendment."""
        if registration_kind not in {"benchmark", "train"}:
            raise RegistryError("recovery only registers benchmark and train workloads")
        return self.register(registration_id, registration_kind, record)

    def register_recovery_cycle_workload(self, registration_id: str, registration_kind: str,
                                         record: dict[str, Any]) -> dict[str, Any]:
        """Register one named cycle-2 workload with its explicit cycle marker."""
        if registration_kind not in {"benchmark", "train"}:
            raise RegistryError("cycle 2 only registers benchmark and train workloads")
        if record.get("cycle_id") != "g008-recovery-cycle-2":
            raise RegistryError("cycle-2 workload record must identify its cycle")
        return self._append("registration", {"registration_id": registration_id,
                                              "registration_kind": registration_kind,
                                              "record": record,
                                              "cycle_id": "g008-recovery-cycle-2"})

    def preflight_workload(self, data: dict[str, Any]) -> dict[str, Any]:
        """Reserve the full registered timeout before a single SDK push attempt."""
        return self._append("workload_preflight", data)

    def dispatch_workload_push(self, data: dict[str, Any]) -> dict[str, Any]:
        """Durably mark the one allowed push call immediately before invoking the SDK."""
        return self._append("workload_push_dispatched", data)

    def accept_workload_push(self, data: dict[str, Any]) -> dict[str, Any]:
        """Record exact assigned-version and metadata evidence before status polling."""
        return self._append("workload_push_accepted", data)

    def resolve_workload_attempt(self, data: dict[str, Any]) -> dict[str, Any]:
        """Resolve an attempt only with terminal status or proven pre-assignment rejection."""
        return self._append("workload_attempt", data)

    def register_benchmark_artifact(self, artifact_sha256: str, registration_event_index: int,
                                    registration_event_hash: str, *, attempt_event_index: int | None = None,
                                    attempt_event_hash: str | None = None,
                                    report_sha256: str | None = None,
                                    selection_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        events = self.events()
        head = events[-1]["event_hash"] if events else None
        data = {"artifact_sha256": artifact_sha256, "pre_lock_head_hash": head,
                "pre_lock_final_event_index": len(events)-1,
                "registration_event_index": registration_event_index,
                "registration_event_hash": registration_event_hash}
        recovery_started = any(e["event_type"] == "recovery_amendment" for e in events)
        supplied_attempt = (attempt_event_index is not None, attempt_event_hash is not None,
                            report_sha256 is not None, selection_evidence is not None)
        if recovery_started:
            if not all(supplied_attempt):
                raise RegistryError("recovery benchmark artifact must bind terminal attempt and report")
            normalized_selection = _canonical_selection_evidence(selection_evidence)
            data.update(attempt_event_index=attempt_event_index, attempt_event_hash=attempt_event_hash,
                        report_sha256=report_sha256, selection_evidence=normalized_selection)
        elif any(supplied_attempt):
            raise RegistryError("attempt-bound benchmark artifact is reserved for recovery lifecycle")
        _verify_benchmark_registration(data, events, {})
        return self._append("benchmark_artifact", data)

    def close_registrations(self) -> dict[str, Any]:
        head, index = self.head
        return self._append("registration_closed", {"pre_lock_head_hash": head,
                                                       "pre_lock_final_event_index": index})

    def attest_benchmark(self, artifact_event_index: int) -> dict[str, Any]:
        events = self.events()
        lock_hash = self.lock_head_hash
        if lock_hash is None:
            raise RegistryError("registration lock must exist before attestation")
        if not isinstance(artifact_event_index, int) or artifact_event_index < 0 or artifact_event_index >= len(events):
            raise RegistryError("invalid benchmark artifact event index")
        artifact_event = events[artifact_event_index]
        if artifact_event["event_type"] != "benchmark_artifact":
            raise RegistryError("event is not a benchmark artifact")
        data = {**artifact_event["event_data"], "lock_head_hash": lock_hash}
        return self._append("benchmark_attestation", data)

    def benchmark_is_attested(self, artifact_event_index: int, artifact_sha256: str) -> bool:
        events = self.events()
        if self.lock_head_hash is None or artifact_event_index < 0 or artifact_event_index >= len(events):
            return False
        artifact = events[artifact_event_index]
        if artifact["event_type"] != "benchmark_artifact" or artifact["event_data"]["artifact_sha256"] != artifact_sha256:
            return False
        registration_index = artifact["event_data"]["registration_event_index"]
        return any(e["event_type"] == "benchmark_attestation"
                   and e["event_data"]["registration_event_index"] == registration_index
                   and e["event_data"]["artifact_sha256"] == artifact_sha256
                   and e["event_data"]["lock_head_hash"] == self.lock_head_hash for e in events)
