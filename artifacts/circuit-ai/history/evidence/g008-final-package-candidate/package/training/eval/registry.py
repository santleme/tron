"""Append-only, hash-chained G008 experiment registry.

This is a local integrity mechanism, not a signature. It detects inconsistent
partial edits while independently captured head hashes remain available.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import statistics
import tempfile
from decimal import Decimal, InvalidOperation
import unicodedata
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
_HASH_CHARS = frozenset("0123456789abcdef")
_REGISTRATION_KINDS = frozenset({"train", "imitation", "ablation", "validation", "final", "benchmark"})
_RECOVERY_EVENT_TYPES = frozenset({
    "recovery_amendment", "workload_preflight", "workload_push_dispatched",
    "workload_push_accepted", "workload_attempt", "recovery_cycle_amendment_v2",
    "recovery_cycle_retry_amendment_v1", "cycle3_retry_policy_amendment_v1",
    "exploratory_kernel_observation_v1", "cycle3_retry_rebased_authorization_v1",
    "benchmark_artifact_validation_rejected",
})
_LEGACY_HEAD_INDEX = 12
_LEGACY_HEAD_HASH = "0208abba4954ffd08b01d5069c674e8bc3f5d1817c95ddb0b0bb4eedd41f3fd7"
_RECOVERY_PLAN_SHA256 = "0358c3c00ffa438a3bd033939687827774da5f04ba54746c075eb4680b384d72"
_CYCLE2_PROPOSAL_SHA256 = "efd2ea6e9525b2d59b3143fae97110f1eb3a76707ae2e19a7512bc84b94cf776"
_CYCLE1_FINAL_EVENT_INDEX = 19
_CYCLE1_FINAL_EVENT_HASH = "7280fb7471484e856b48d1a931cb18d97d5bd2f94949790c41defef9154c4213"
_CYCLE2_EVENT_INDEX = 20
_CYCLE3_EVENT_INDEX = 27
_CYCLE3_PARENT_EVENT_HASH = "850b7fe3f0c2afbdf229cda83a3d2dba51bc607fa3a6efa337d7a6d70de309f7"
# Frozen digest of the reviewed Cycle-3 retry proposal.
_CYCLE3_RETRY_PROPOSAL_SHA256 = "40bb92c5901c040e9efe4ac7d62cf71a124c18451eb504b53d6c62f44e391f89"
# Policy digest for append-only continuations of the already registered Cycle-3
# benchmark. The original registration, resources, and cumulative caps remain fixed.
_CYCLE3_RETRY_POLICY_SHA256 = hashlib.sha256(
    b"g008-cycle3-retry-policy-v1:append-only;exact-head;same-registration;budget-bounded;"
    b"terminal-error-or-complete-with-explicit-validation-rejection"
).hexdigest()
_CYCLE3_RETRY_PARENT_EVENT_INDEX = 33
_CYCLE3_RETRY_PARENT_EVENT_HASH = "4f7c8d909349b3a95c91758a2fe6c7490f241ed5db9df9925ce4f600163db966"
# Frozen Cycle-3 proposal digest; changing the proposal requires a reviewed registry update.
_CYCLE3_PROPOSAL_SHA256 = "731ad7a7b477bc18e87572f6827fa7f18f38bbdedf130765d600fe5558250527"
# Provisional Cycle-3 limits copied from the approved Cycle-2 envelope; update
# together with the reviewed Cycle-3 proposal if its frozen limits differ.
_CYCLE3_BENCHMARK_GPU_SECONDS = 5600
_CYCLE3_TRAIN_GPU_SECONDS = 7200
_CYCLE3_PROJECT_EXPERIMENT_GPU_SECONDS = 21600
_CYCLE3_PROJECT_TOTAL_GPU_SECONDS = 28800
_CYCLE3_FINAL_RESERVE_GPU_SECONDS = 7200
_CYCLE3_OUTPUT_CATEGORY_CAP_BYTES = 1_000_000_000
_CYCLE2_BENCHMARK_GPU_SECONDS = 5600
_CYCLE2_TRAIN_GPU_SECONDS = 7200
_CYCLE2_OUTPUT_CATEGORY_CAP = 1_000_000_000
_CYCLE2_BENCHMARK_ATTEMPT_OUTPUT_CAP = 100_000_000
_CYCLE2_CAMPAIGN_GPU_BEFORE = 8549
_CYCLE2_CAMPAIGN_OUTPUT_BEFORE = 30922
_SUPERSEDED_PLAN_SHA256 = "0a9e1ebccae2a793fadf531fb99013a9d19ff832a5880de9ee15204b477ca81a"
_SUPERSEDED_TEST_SPEC_SHA256 = "e3b8efef4b441d694883a856164ed3d25f2b278f2bd7ca9ee30dfb8097b3ba6c"
_RECOVERY_SEED_PLAN_SHA256 = "c6b16733a045e94e7310183e8be8a62cf4327b5d2bdc3eed40bb9d552ce17593"
_G008_VALIDATION_REFERENCE_SHA256 = "22f7e0fca065eba24ab57c5fae4ea78223fa6d01153a7a8d90e83ac8269228d7"
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


_CYCLE3_DERIVED_RECORD_FIELDS = frozenset({
    "recovery_cycle_amendment_event_index", "recovery_cycle_amendment_event_hash",
})


def cycle3_registration_record_sha256(record: dict[str, Any]) -> str:
    """Hash Cycle-3 registration bytes before adding the derived amendment pointer."""
    if not isinstance(record, dict):
        raise RegistryError("cycle-3 registration record must be an object")
    projection = {key: value for key, value in record.items() if key not in _CYCLE3_DERIVED_RECORD_FIELDS}
    return hashlib.sha256(canonical_bytes(projection)).hexdigest()


def cycle3_registration_bindings_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash the ordered, amendment-bound benchmark/train identity and record rows."""
    if not isinstance(rows, list) or len(rows) != 2:
        raise RegistryError("cycle-3 registration bindings require benchmark and train rows")
    if any(not isinstance(row, dict) or set(row) != {
            "registration_id", "registration_kind", "run_uuid", "record_sha256"} for row in rows):
        raise RegistryError("cycle-3 registration binding row shape is invalid")
    if [row["registration_kind"] for row in rows] != ["benchmark", "train"]:
        raise RegistryError("cycle-3 registration bindings must be ordered benchmark then train")
    if any(not all(isinstance(row[key], str) and row[key] for key in ("registration_id", "run_uuid"))
           or not _is_hash(row["record_sha256"]) for row in rows):
        raise RegistryError("cycle-3 registration binding row contains invalid identity or record hash")
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


_CYCLE3_PREFLIGHT_CONTEXT_FIELDS = (
    "kernel_id", "current_version_number", "current_terminal_status", "current_status_evidence",
    "quota_gpu_seconds_remaining_before", "quota_evidence",
)


def cycle3_preflight_context_sha256(context: dict[str, Any]) -> str:
    """Hash the exact live-kernel and quota context authorized before Cycle 3."""
    if not isinstance(context, dict) or any(key not in context for key in _CYCLE3_PREFLIGHT_CONTEXT_FIELDS):
        raise RegistryError("cycle-3 preflight context is incomplete")
    projection = {key: context[key] for key in _CYCLE3_PREFLIGHT_CONTEXT_FIELDS}
    return hashlib.sha256(canonical_bytes(projection)).hexdigest()


def cycle3_authorization_statement(proposal_sha256: str, registration_bindings_sha256: str,
                                   preflight_context_sha256: str) -> str:
    """Return the only accepted authorization statement for all frozen hashes."""
    if not all(_is_hash(value) for value in
               (proposal_sha256, registration_bindings_sha256, preflight_context_sha256)):
        raise RegistryError("cycle-3 authorization statement requires SHA-256 bindings")
    return (f"Authorize G008 Cycle 3 proposal sha256:{proposal_sha256}. "
            f"Registration bindings sha256:{registration_bindings_sha256}. "
            f"Preflight context sha256:{preflight_context_sha256}. "
            "Exclusive writer: no other writer updates pascalcoimbra/circuit-ai until the benchmark is terminal and reconciled.")


_CYCLE3_RETRY_RECORD_POINTER_FIELDS = frozenset({
    "cycle3_retry_amendment_event_index", "cycle3_retry_amendment_event_hash",
    "recovery_cycle_amendment_event_index", "recovery_cycle_amendment_event_hash",
})
_CYCLE3_RETRY_CONTEXT_FIELDS = (
    "kernel_id", "current_version_number", "current_terminal_status", "current_status_evidence",
    "quota_gpu_seconds_remaining_before", "quota_evidence",
)


def cycle3_retry_registration_record_sha256(record: dict[str, Any]) -> str:
    """Hash the authorized retry benchmark record before adding derived event pointers."""
    if not isinstance(record, dict):
        raise RegistryError("cycle-3 retry registration record must be an object")
    projection = {key: value for key, value in record.items()
                  if key not in _CYCLE3_RETRY_RECORD_POINTER_FIELDS}
    return hashlib.sha256(canonical_bytes(projection)).hexdigest()


def cycle3_retry_preflight_context_sha256(context: dict[str, Any]) -> str:
    """Hash exact live Kaggle status and quota context for the retry authorization."""
    if not isinstance(context, dict) or any(key not in context for key in _CYCLE3_RETRY_CONTEXT_FIELDS):
        raise RegistryError("cycle-3 retry preflight context is incomplete")
    projection = {key: context[key] for key in _CYCLE3_RETRY_CONTEXT_FIELDS}
    return hashlib.sha256(canonical_bytes(projection)).hexdigest()


def cycle3_retry_context_sha256(context: dict[str, Any]) -> str:
    """Public short name for the retry's frozen live-context digest."""
    return cycle3_retry_preflight_context_sha256(context)


def cycle3_retry_binding_sha256(registration_id: str, registration_kind: str,
                                run_uuid: str, record_sha256: str) -> str:
    """Hash the exact single-registration identity and authorized record digest."""
    row = {"registration_id": registration_id, "registration_kind": registration_kind,
           "run_uuid": run_uuid, "record_sha256": record_sha256}
    if (registration_kind != "benchmark" or not isinstance(registration_id, str) or not registration_id
            or not isinstance(run_uuid, str) or not run_uuid or not _is_hash(record_sha256)):
        raise RegistryError("cycle-3 retry binding must identify one benchmark registration")
    return hashlib.sha256(canonical_bytes(row)).hexdigest()


def cycle3_retry_authorization_statement(proposal_sha256: str, record_sha256: str,
                                         context_sha256: str) -> str:
    """Return the canonical authorization statement binding the retry proposal and package."""
    if not all(_is_hash(value) for value in (proposal_sha256, record_sha256, context_sha256)):
        raise RegistryError("cycle-3 retry authorization requires SHA-256 bindings")
    return (f"Authorize G008 Cycle 3 benchmark retry proposal sha256:{proposal_sha256}. "
            f"Benchmark record sha256:{record_sha256}. Preflight context sha256:{context_sha256}. "
            "Exclusive writer: no other writer updates pascalcoimbra/circuit-ai until the retry benchmark is terminal and reconciled.")


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HASH_CHARS for c in value)


_G008_V2_TRAIN_CONFIG_FLOAT_FIELDS = frozenset({
    "learning_rate", "gamma", "gae_lambda", "clip_ratio", "entropy_coef", "value_coef",
    "snapshot_opponent_probability", "heuristic_opponent_probability",
})
_G008_V2_TRAIN_CONFIG_INT_FIELDS = frozenset({
    "width", "height", "arena_count", "rollout_steps", "total_updates", "update_epochs",
    "minibatch_size", "hidden_size", "seed", "checkpoint_every", "onnx_opset",
})
_G008_V2_TRAIN_CONFIG_FIELDS = (_G008_V2_TRAIN_CONFIG_FLOAT_FIELDS
                                | _G008_V2_TRAIN_CONFIG_INT_FIELDS | {"device"})


def _g008_v2_numeric_train_config(train_config: Any) -> dict[str, Any]:
    """Decode the float-as-decimal-string v2 config wire format without changing registry JSON."""
    if not isinstance(train_config, dict) or set(train_config) != _G008_V2_TRAIN_CONFIG_FIELDS:
        raise RegistryError("campaign v2 train_config must contain the exact TrainConfig fields")
    numeric = dict(train_config)
    for key in _G008_V2_TRAIN_CONFIG_FLOAT_FIELDS:
        value = train_config[key]
        if not isinstance(value, str):
            raise RegistryError(f"campaign v2 train_config {key} must use canonical decimal-string encoding")
        try:
            decimal = Decimal(value)
            converted = float(decimal)
        except (InvalidOperation, ValueError, OverflowError) as exc:
            raise RegistryError(f"campaign v2 train_config {key} is not a finite canonical decimal") from exc
        if not math.isfinite(converted) or repr(converted) != value:
            raise RegistryError(f"campaign v2 train_config {key} must round-trip through repr(float)")
        numeric[key] = converted
    if (any(not isinstance(train_config[key], int) or isinstance(train_config[key], bool)
            for key in _G008_V2_TRAIN_CONFIG_INT_FIELDS)
            or train_config["device"] != "auto"):
        raise RegistryError("campaign v2 train_config integer fields or device differ from frozen config wire format")
    return numeric


def _g008_v2_train_config_sha256(train_config: Any) -> str:
    numeric = _g008_v2_numeric_train_config(train_config)
    from training.ppo.protocol import canonical_json
    return hashlib.sha256(canonical_json(numeric)).hexdigest()


def _g008_v2_shared_config_sha256(train_config: Any) -> str:
    numeric = _g008_v2_numeric_train_config(train_config)
    shared = {key: value for key, value in numeric.items()
              if key not in {"device", "seed", "snapshot_opponent_probability",
                             "heuristic_opponent_probability"}}
    from training.ppo.protocol import canonical_json
    return hashlib.sha256(canonical_json(shared)).hexdigest()


def competitive_final_statistics(wins: int, draws: int, losses: int) -> dict[str, Any]:
    """Return the frozen 512-game win-rate/Wilson report fields for final evaluation."""
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (wins, draws, losses)) or wins + draws + losses != 512):
        raise RegistryError("final evaluation statistics require nonnegative W/D/L totaling 512")
    rate = wins / 512
    z = 1.959963984540054
    denominator = 1 + z * z / 512
    center = (rate + z * z / (2 * 512)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / 512 + z * z / (4 * 512 * 512)) / denominator
    return {"game_count": 512, "wins": wins, "draws": draws, "losses": losses,
            "win_rate": f"{rate:.6f}", "wilson_95_lower": f"{center - margin:.6f}",
            "wilson_95_upper": f"{center + margin:.6f}", "passed": rate >= 0.60}


def competitive_noninferiority_statistics(wins: int, draws: int, losses: int) -> dict[str, Any]:
    """Compare the frozen candidate's win rate with G008 v15 using Newcombe's 95% CI."""
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (wins, draws, losses)) or wins + draws + losses != 512):
        raise RegistryError("validation statistics require nonnegative W/D/L totaling 512")
    reference_wins, sample_n = 217, 512
    candidate_rate, reference_rate = wins / sample_n, reference_wins / sample_n
    z = 1.959963984540054

    def bounds(rate: float) -> tuple[float, float]:
        denominator = 1 + z * z / sample_n
        center = (rate + z * z / (2 * sample_n)) / denominator
        margin = z * math.sqrt(rate * (1 - rate) / sample_n + z * z / (4 * sample_n * sample_n)) / denominator
        return center - margin, center + margin

    candidate_lower, candidate_upper = bounds(candidate_rate)
    reference_lower, reference_upper = bounds(reference_rate)
    difference = candidate_rate - reference_rate
    difference_lower = difference - math.sqrt((candidate_rate - candidate_lower) ** 2
                                               + (reference_upper - reference_rate) ** 2)
    difference_upper = difference + math.sqrt((candidate_upper - candidate_rate) ** 2
                                               + (reference_rate - reference_lower) ** 2)
    return {"reference_report_sha256": _G008_VALIDATION_REFERENCE_SHA256,
            "reference_game_count": 512, "reference_wins": 217, "reference_draws": 51,
            "reference_losses": 244, "noninferiority_margin": "0.065000",
            "game_count": 512, "wins": wins, "draws": draws, "losses": losses,
            "win_rate": f"{candidate_rate:.6f}",
            "wilson_95_lower": f"{candidate_lower:.6f}",
            "wilson_95_upper": f"{candidate_upper:.6f}",
            "win_rate_difference_95_lower": f"{difference_lower:.6f}",
            "win_rate_difference_95_upper": f"{difference_upper:.6f}",
            "passed": difference_lower >= -0.065}


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
    if event_type == "g008_competitive_campaign_amendment_v1":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "prior_final_event_index", "rejection_event_index",
                    "rejection_event_hash", "kernel_id", "current_version_number",
                    "benchmark_gpu_seconds_cap", "training_gpu_seconds_cap", "total_gpu_seconds_cap",
                    "evaluation_gpu_seconds_cap", "protected_final_reserve_gpu_seconds",
                    "evaluation_rosters", "validation_reference", "records"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-competitive-campaign" or d["schema_version"] != 1
                or d["campaign_id"] != "g008-competitive-20261020-v1"):
            raise RegistryError("unsupported G008 competitive campaign amendment")
        if (not _is_hash(d["prior_head_hash"]) or not isinstance(d["prior_final_event_index"], int)
                or isinstance(d["prior_final_event_index"], bool) or d["prior_final_event_index"] < 0
                or not isinstance(d["rejection_event_index"], int) or isinstance(d["rejection_event_index"], bool)
                or d["rejection_event_index"] < 0 or not _is_hash(d["rejection_event_hash"])
                or not isinstance(d["kernel_id"], str) or not d["kernel_id"]
                or not isinstance(d["current_version_number"], int) or isinstance(d["current_version_number"], bool)
                or d["current_version_number"] < 14
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")):
            raise RegistryError("competitive campaign must bind exact head, rejection, and Kaggle version")
        budget_keys = ("benchmark_gpu_seconds_cap", "training_gpu_seconds_cap", "total_gpu_seconds_cap",
                       "evaluation_gpu_seconds_cap",
                       "protected_final_reserve_gpu_seconds")
        if any(not isinstance(d[k], int) or isinstance(d[k], bool) or d[k] <= 0 for k in budget_keys):
            raise RegistryError("competitive campaign budgets must be positive integer seconds")
        if (d["total_gpu_seconds_cap"] > 6 * _GPU_HOUR
                or d["benchmark_gpu_seconds_cap"] + d["training_gpu_seconds_cap"] > d["total_gpu_seconds_cap"]
                or d["benchmark_gpu_seconds_cap"] + d["training_gpu_seconds_cap"] + d["evaluation_gpu_seconds_cap"] > d["total_gpu_seconds_cap"]
                or d["protected_final_reserve_gpu_seconds"] != 2 * _GPU_HOUR):
            raise RegistryError("competitive campaign exceeds experiment allocation or protected reserve")
        rosters = d["evaluation_rosters"]
        if (not isinstance(rosters, dict) or set(rosters) != {"validation", "final"}
                or any(not isinstance(rosters[name], dict) or set(rosters[name]) != {
                    "roster_sha256", "opponent_code_sha256", "game_count", "seat_order"}
                       or not _is_hash(rosters[name]["roster_sha256"])
                       or not _is_hash(rosters[name]["opponent_code_sha256"])
                       or rosters[name]["game_count"] != 512 or rosters[name]["seat_order"] != [0, 1]
                       for name in ("validation", "final"))):
            raise RegistryError("campaign must freeze hash-only 512-game validation and final rosters")
        expected_rosters = {
            "validation": (_VALIDATION_OPPONENT_ROSTER_SHA256, _VALIDATION_OPPONENT_SHA256),
            "final": (_FINAL_OPPONENT_ROSTER_SHA256, _FINAL_OPPONENT_CODE_SHA256),
        }
        if any((rosters[name]["roster_sha256"], rosters[name]["opponent_code_sha256"]) != expected
               for name, expected in expected_rosters.items()):
            raise RegistryError("campaign evaluation roster hashes differ from frozen seed plan")
        reference = d["validation_reference"]
        if (not isinstance(reference, dict) or set(reference) != {
                "report_sha256", "game_count", "wins", "draws", "losses", "noninferiority_margin"}
                or reference.get("report_sha256") != _G008_VALIDATION_REFERENCE_SHA256
                or (reference.get("game_count"), reference.get("wins"), reference.get("draws"),
                    reference.get("losses"), reference.get("noninferiority_margin"))
                != (512, 217, 51, 244, "0.065000")):
            raise RegistryError("campaign must bind the exact historical validation reference and 6.5-point margin")
        rows = d["records"]
        if not isinstance(rows, list) or len(rows) != 9:
            raise RegistryError("competitive campaign must freeze benchmark, six training, validation, and final records")
        if any(not isinstance(row, dict) or set(row) != {"registration_id", "registration_kind", "record_sha256", "record"}
               or not isinstance(row["registration_id"], str) or not row["registration_id"]
               or not _is_hash(row["record_sha256"]) or not isinstance(row["record"], dict)
               for row in rows):
            raise RegistryError("competitive campaign record bindings are malformed")
        if (rows[0]["registration_kind"] != "benchmark" or any(row["registration_kind"] != "train" for row in rows[1:7])
                or rows[7]["registration_kind"] != "validation" or rows[8]["registration_kind"] != "final"):
            raise RegistryError("campaign records must be benchmark, six training, validation, and final")
        if len({row["registration_id"] for row in rows}) != 9 or len({row["record"].get("run_uuid") for row in rows}) != 9:
            raise RegistryError("competitive campaign registration IDs and run UUIDs must be unique")
        if any(hashlib.sha256(canonical_bytes(row["record"])).hexdigest() != row["record_sha256"] for row in rows):
            raise RegistryError("competitive campaign record digest mismatch")
        if sum(row["record"].get("session_timeout_seconds", 0) for row in rows[1:7]) > d["training_gpu_seconds_cap"]:
            raise RegistryError("competitive campaign training reservations exceed frozen cap")
        if sum(row["record"].get("session_timeout_seconds", 0) for row in rows[7:]) > d["evaluation_gpu_seconds_cap"]:
            raise RegistryError("competitive campaign evaluation reservations exceed frozen cap")
        if rows[0]["record"].get("session_timeout_seconds", 0) > d["benchmark_gpu_seconds_cap"]:
            raise RegistryError("competitive campaign benchmark reservation exceeds frozen cap")
        ids = set()
        pairs = {}
        for pos, row in enumerate(rows):
            record, kind = row["record"], row["registration_kind"]
            if (record.get("campaign_id") != d["campaign_id"] or record.get("kernel_id") != d["kernel_id"]
                    or record.get("accelerator_count") != 1 or record.get("machine_shape") != "NvidiaTeslaT4"
                    or record.get("registration_kind") != kind or record.get("open_space_allowed") is not False):
                raise RegistryError("competitive campaign record changed kernel, hardware, role, or final-only boundary")
            for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                        "backend_sha256", "trainer_sha256", "engine_sha256"):
                if not _is_hash(record.get(key)):
                    raise RegistryError("competitive campaign record has an invalid frozen artifact/config hash")
            if record["seed_plan_sha256"] != _RECOVERY_SEED_PLAN_SHA256:
                raise RegistryError("competitive campaign changed frozen seed-plan hash")
            probs = record.get("opponent_probabilities")
            if kind == "train":
                role = record.get("strategy_role")
                if (role not in {"baseline", "mixed"} or not isinstance(record.get("pair_id"), str)
                        or not record["pair_id"] or not isinstance(record.get("ppo_seed"), int)
                        or isinstance(record.get("ppo_seed"), bool) or not isinstance(record.get("root_seed"), int)
                        or isinstance(record.get("root_seed"), bool) or record["root_seed"] < 2_000_000
                        or record["root_seed"] > 2_999_999 or not isinstance(record.get("update_budget"), int)
                        or isinstance(record.get("update_budget"), bool) or record["update_budget"] <= 0
                        or not _is_hash(record.get("shared_config_sha256"))
                        or not isinstance(record.get("ppo_config"), dict)
                        or hashlib.sha256(canonical_bytes(record["ppo_config"])).hexdigest() != record["shared_config_sha256"]):
                    raise RegistryError("competitive training job lacks frozen paired seed/config/update metadata")
                if (not isinstance(probs, dict) or set(probs) != {"latest", "step0", "cruise", "local_mobility", "interceptor"}
                        or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in probs.values())
                        or sum(probs.values()) != 10_000):
                    raise RegistryError("competitive opponent probabilities must be nonnegative fixed basis points summing to 10000")
                if role == "baseline" and probs != {"latest": 10_000, "step0": 0, "cruise": 0, "local_mobility": 0, "interceptor": 0}:
                    raise RegistryError("baseline must use latest-only opponent distribution")
                if role == "mixed" and (probs["latest"] == 10_000 or any(probs[name] == 0 for name in ("step0", "cruise", "local_mobility", "interceptor"))):
                    raise RegistryError("mixed strategy must use the frozen nonzero opponent pool")
                pair = pairs.setdefault(record["pair_id"], {})
                if role in pair:
                    raise RegistryError("competitive campaign duplicates a paired arm")
                pair[role] = record
            elif kind == "benchmark":
                if pos != 0 or probs is not None or record.get("strategy_role") != "benchmark":
                    raise RegistryError("benchmark registration must be the unique benchmark workload")
            else:
                roster = rosters[kind]
                if (record.get("roster_sha256") != roster["roster_sha256"]
                        or record.get("opponent_code_sha256") != roster["opponent_code_sha256"]
                        or record.get("game_count") != 512 or record.get("seat_order") != [0, 1]
                        or "open-space" in canonical_bytes(record).decode("utf-8")
                        or "root_seeds" in record or "roots" in record):
                    raise RegistryError("evaluation registration must use hash-only frozen roster commitment")
                if kind == "final" and record.get("pre_final_data_visible") is not False:
                    raise RegistryError("final opponent identity and root roster must remain hidden pre-final")
        seed_pairs = [(pair["baseline"]["ppo_seed"], pair["baseline"]["root_seed"]) for pair in pairs.values()]
        all_root_seeds = [record["root_seed"] for pair in pairs.values() for record in pair.values()]
        if (len(pairs) != 3 or len({root for _, root in seed_pairs}) != 3
                or len(set(all_root_seeds)) != 6 or {seed for seed, _ in seed_pairs} != {1234}
                or any(set(pair) != {"baseline", "mixed"} for pair in pairs.values())):
            raise RegistryError("competitive campaign requires three complete blocks on frozen PPO seed 1234")
        for pair in pairs.values():
            base, mixed = pair["baseline"], pair["mixed"]
            invariant = ("ppo_seed", "update_budget", "shared_config_sha256", "source_sha256",
                         "backend_sha256", "trainer_sha256", "engine_sha256", "architecture", "ppo_config")
            if any(base.get(k) != mixed.get(k) for k in invariant):
                raise RegistryError("paired training jobs differ outside opponent probabilities/config digest")
        mixed_distributions = [pair["mixed"]["opponent_probabilities"] for pair in pairs.values()]
        if any(probs != mixed_distributions[0] for probs in mixed_distributions[1:]):
            raise RegistryError("mixed opponent distribution must be fixed across all three seed pairs")
    elif event_type == "g008_competitive_campaign_amendment_v2":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "prior_final_event_index",
                    "benchmark_artifact_event_index", "benchmark_artifact_event_hash",
                    "benchmark_lock_event_index", "benchmark_lock_event_hash",
                    "benchmark_attestation_event_index", "benchmark_attestation_event_hash",
                    "selected_training_registration_event_index", "selected_training_registration_event_hash",
                    "selected_training_attempt_event_index", "selected_training_attempt_event_hash",
                    "selected_training_run_uuid",
                    "kernel_id", "current_version_number", "training_gpu_seconds_cap",
                    "evaluation_gpu_seconds_cap", "total_gpu_seconds_cap",
                    "protected_final_reserve_gpu_seconds", "output_caps_bytes",
                    "quota_gpu_seconds_remaining_before", "validation_roster",
                    "candidate_selection_rule_id", "records"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-competitive-campaign" or d["schema_version"] != 2
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or not isinstance(d["prior_head_hash"], str) or not _is_hash(d["prior_head_hash"])
                or d["prior_final_event_index"] != 54
                or not isinstance(d["kernel_id"], str) or not d["kernel_id"]
                or d["current_version_number"] != 17):
            raise RegistryError("invalid G008 competitive campaign v2 parent or cursor")
        anchors = (("benchmark_artifact_event_index", "benchmark_artifact_event_hash", 48),
                   ("benchmark_lock_event_index", "benchmark_lock_event_hash", 49),
                   ("benchmark_attestation_event_index", "benchmark_attestation_event_hash", 50),
                   ("selected_training_registration_event_index", "selected_training_registration_event_hash", 29),
                   ("selected_training_attempt_event_index", "selected_training_attempt_event_hash", 54))
        for index_key, hash_key, expected_index in anchors:
            if (d[index_key] != expected_index or not isinstance(d[hash_key], str)
                    or not _is_hash(d[hash_key])):
                raise RegistryError("campaign v2 provenance anchor is malformed")
        if (d["training_gpu_seconds_cap"] != 7_200 or d["evaluation_gpu_seconds_cap"] != 4_200
                or d["total_gpu_seconds_cap"] != 11_400
                or d["protected_final_reserve_gpu_seconds"] != 7_200
                or not isinstance(d["quota_gpu_seconds_remaining_before"], int)
                or isinstance(d["quota_gpu_seconds_remaining_before"], bool)
                or d["quota_gpu_seconds_remaining_before"] < 18_600
                or d["output_caps_bytes"] != {"aggregate": 5_000_000_000, "per_run": 100_000_000}
                or not isinstance(d["selected_training_run_uuid"], str) or not d["selected_training_run_uuid"]):
            raise RegistryError("campaign v2 exceeds frozen compute, quota, or output budget")
        if (d["candidate_selection_rule_id"] !=
                "paired-score-gain-t95-lower-positive-no-pair-regresses-over-5pp-then-median-arm-checkpoint-v1"):
            raise RegistryError("campaign v2 candidate-selection rule is not the frozen preregistered rule")
        validation_roster = d["validation_roster"]
        if (not isinstance(validation_roster, dict) or set(validation_roster) != {
                "roster_sha256", "opponent_code_sha256", "game_count", "seat_order", "paired_seeds_sha256"}
                or validation_roster.get("roster_sha256") != _VALIDATION_OPPONENT_ROSTER_SHA256
                or validation_roster.get("opponent_code_sha256") != _VALIDATION_OPPONENT_SHA256
                or validation_roster.get("game_count") != 512
                or validation_roster.get("seat_order") != [0, 1]
                or not _is_hash(validation_roster.get("paired_seeds_sha256"))):
            raise RegistryError("campaign v2 must freeze the exact hash-only paired non-open-space validation roster")
        rows = d["records"]
        if (not isinstance(rows, list) or len(rows) != 13
                or any(not isinstance(row, dict) or set(row) != {
                    "registration_id", "registration_kind", "record_sha256", "record"}
                    or not isinstance(row["registration_id"], str) or not row["registration_id"]
                    or not _is_hash(row["record_sha256"]) or not isinstance(row["record"], dict)
                    for row in rows)):
            raise RegistryError("campaign v2 must freeze six training, six validation, and final registrations")
        if (any(row["registration_kind"] != "train" for row in rows[:6])
                or any(row["registration_kind"] != "validation" for row in rows[6:12])
                or rows[12]["registration_kind"] != "final"
                or len({row["registration_id"] for row in rows}) != 13
                or len({row["record"].get("run_uuid") for row in rows}) != 13
                or any(hashlib.sha256(canonical_bytes(row["record"])).hexdigest() != row["record_sha256"]
                       for row in rows)):
            raise RegistryError("campaign v2 registration identities or digests are invalid")
        if any(row["record"].get("session_timeout_seconds") != 1_200 for row in rows[:6]):
            raise RegistryError("campaign v2 must reserve six 1200-second training workloads")
        if sum(row["record"].get("session_timeout_seconds", 0) for row in rows[:6]) != 7_200:
            raise RegistryError("campaign v2 training reservations must total exactly 7200 seconds")
        if any(row["record"].get("session_timeout_seconds") != 600 for row in rows[6:]):
            raise RegistryError("campaign v2 must reserve 600 seconds for each of six validations and final")
        if sum(row["record"].get("session_timeout_seconds", 0) for row in rows[6:]) != 4_200:
            raise RegistryError("campaign v2 evaluation reservations must total exactly 4200 seconds")
        pair_rows: dict[str, dict[str, dict[str, Any]]] = {}
        ppo_seeds: set[int] = set()
        root_seeds: set[int] = set()
        for i, row in enumerate(rows):
            record, kind = row["record"], row["registration_kind"]
            if (record.get("campaign_id") != d["campaign_id"]
                    or record.get("registration_kind") != kind
                    or record.get("kernel_id") != d["kernel_id"]
                    or record.get("accelerator_count") != 1
                    or record.get("machine_shape") != "NvidiaTeslaT4"
                    or record.get("open_space_allowed") is not False):
                raise RegistryError("campaign v2 record changed frozen kernel/hardware/final-only boundary")
            for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                        "backend_sha256", "trainer_sha256", "engine_sha256"):
                if not _is_hash(record.get(key)):
                    raise RegistryError("campaign v2 record has an invalid frozen artifact/config hash")
            if (not isinstance(record.get("workload_output_bytes_cap"), int)
                    or isinstance(record.get("workload_output_bytes_cap"), bool)
                    or record["workload_output_bytes_cap"] <= 0
                    or record["workload_output_bytes_cap"] > d["output_caps_bytes"]["per_run"]
                    or not isinstance(record.get("workload_gpu_seconds_cap"), int)
                    or isinstance(record.get("workload_gpu_seconds_cap"), bool)
                    or record["workload_gpu_seconds_cap"] > record["session_timeout_seconds"]):
                raise RegistryError("campaign v2 record exceeds per-run output cap")
            if record["workload_output_bytes_cap"] != 100_000_000:
                raise RegistryError("campaign v2 must freeze the 100 MB output reservation for every run")
            if kind == "train":
                role, pair_id = record.get("strategy_role"), record.get("pair_id")
                if (role not in {"baseline", "mixed"} or not isinstance(pair_id, str) or not pair_id
                        or not isinstance(record.get("ppo_seed"), int) or isinstance(record.get("ppo_seed"), bool)
                        or not isinstance(record.get("root_seed"), int) or isinstance(record.get("root_seed"), bool)
                        or not isinstance(record.get("update_budget"), int)
                        or isinstance(record.get("update_budget"), bool) or record["update_budget"] <= 0
                        or not isinstance(record.get("architecture"), dict)
                        or not isinstance(record.get("train_config"), dict)
                        or not isinstance(record.get("opponent_probabilities"), dict)):
                    raise RegistryError("campaign v2 paired training record lacks frozen seed/config metadata")
                expected_pair_no = (int(pair_id.removeprefix("seed-pair-"))
                                    if pair_id.startswith("seed-pair-") else 0)
                numeric_train_config = _g008_v2_numeric_train_config(record["train_config"])
                expected_architecture = {"width": numeric_train_config["width"],
                                         "height": numeric_train_config["height"],
                                         "hidden_size": numeric_train_config["hidden_size"]}
                if (expected_pair_no not in {1, 2, 3}
                        or record["ppo_seed"] != 3_300 + expected_pair_no
                        or record["root_seed"] != 2_000_000 + expected_pair_no * 100_000 + 1
                        or record["update_budget"] != 1_000
                        or record["session_timeout_seconds"] != 1_200
                        or record["workload_gpu_seconds_cap"] != 1_200
                        or numeric_train_config["seed"] != record["ppo_seed"]
                        or numeric_train_config["total_updates"] != record["update_budget"]
                        or numeric_train_config["checkpoint_every"] != 50
                        or "ppo_config" in record):
                    raise RegistryError("campaign v2 training schedule/seeds differ from frozen 1000-update plan")
                if record["architecture"] != expected_architecture:
                    raise RegistryError("campaign v2 architecture differs from exact TrainConfig dimensions")
                if record["config_sha256"] != _g008_v2_train_config_sha256(record["train_config"]):
                    raise RegistryError("campaign v2 config_sha256 differs from full numeric TrainConfig")
                if record["shared_config_sha256"] != _g008_v2_shared_config_sha256(record["train_config"]):
                    raise RegistryError("campaign v2 shared config digest differs from invariant TrainConfig projection")
                if not _is_hash(record.get("schedule_descriptor_sha256")):
                    raise RegistryError("campaign v2 training record lacks its frozen schedule descriptor hash")
                probs = record["opponent_probabilities"]
                if (set(probs) != {"latest", "step0", "cruise", "local_mobility", "interceptor"}
                        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                               for value in probs.values()) or sum(probs.values()) != 10_000):
                    raise RegistryError("campaign v2 opponent probabilities must be fixed basis points")
                ppo_seeds.add(record["ppo_seed"])
                root_seeds.add(record["root_seed"])
                pair = pair_rows.setdefault(pair_id, {})
                if role in pair:
                    raise RegistryError("campaign v2 duplicates a paired training arm")
                pair[role] = record
            elif kind == "validation":
                if (record.get("pair_id") not in {f"seed-pair-{i}" for i in range(1, 4)}
                        or record.get("strategy_role") not in {"baseline", "mixed"}
                        or not isinstance(record.get("training_registration_id"), str)
                        or not isinstance(record.get("training_run_uuid"), str)
                        or not _is_hash(record.get("training_record_sha256"))
                        or record.get("roster_sha256") != validation_roster["roster_sha256"]
                        or record.get("opponent_code_sha256") != validation_roster["opponent_code_sha256"]
                        or record.get("game_count") != 512 or record.get("seat_order") != [0, 1]
                        or record.get("workload_gpu_seconds_cap") != 600
                        or record.get("paired_seeds_sha256") != validation_roster["paired_seeds_sha256"]):
                    raise RegistryError("campaign v2 validation row must link one arm to the frozen paired roster")
                if "open-space" in canonical_bytes(record).decode("utf-8"):
                    raise RegistryError("campaign v2 validation data must not contain the final opponent identity")
            else:
                if (record.get("game_count") != 512 or record.get("seat_order") != [0, 1]
                        or record.get("workload_gpu_seconds_cap") != 600
                        or not _is_hash(record.get("roster_sha256"))
                        or not _is_hash(record.get("opponent_code_sha256"))
                        or "open-space" in canonical_bytes(record).decode("utf-8")
                        or "root_seeds" in record or "roots" in record):
                    raise RegistryError("campaign v2 evaluations must use hash-only non-final roster commitments")
                if kind == "final" and record.get("pre_final_data_visible") is not False:
                    raise RegistryError("campaign v2 final roster must remain hidden before final evaluation")
                if kind == "final" and (record.get("roster_sha256") != _FINAL_OPPONENT_ROSTER_SHA256
                                         or record.get("opponent_code_sha256") != _FINAL_OPPONENT_CODE_SHA256):
                    raise RegistryError("campaign v2 final registration differs from frozen open-space roster hashes")
        if (len(pair_rows) != 3 or len(ppo_seeds) != 3 or len(root_seeds) != 3
                or any(set(pair) != {"baseline", "mixed"} for pair in pair_rows.values())):
            raise RegistryError("campaign v2 requires three complete paired seed blocks with unique seeds")
        paired_seed_commitment = hashlib.sha256(canonical_bytes([
            {"pair_id": f"seed-pair-{pair_no}",
             "ppo_seed": pair_rows[f"seed-pair-{pair_no}"]["baseline"]["ppo_seed"],
             "root_seed": pair_rows[f"seed-pair-{pair_no}"]["baseline"]["root_seed"],
             "seat_order": [0, 1]}
            for pair_no in range(1, 4)
        ])).hexdigest()
        if validation_roster["paired_seeds_sha256"] != paired_seed_commitment:
            raise RegistryError("campaign v2 validation roster seed commitment differs from paired training seeds")
        validations = {}
        for row in rows[6:12]:
            record = row["record"]
            pair_role = (record["pair_id"], record["strategy_role"])
            if pair_role in validations:
                raise RegistryError("campaign v2 duplicates a paired validation arm")
            validations[pair_role] = record
            training = pair_rows.get(record["pair_id"], {}).get(record["strategy_role"])
            train_row = next((item for item in rows[:6]
                              if item["record"] is training), None)
            if (training is None or train_row is None
                    or record["training_registration_id"] != train_row["registration_id"]
                    or record["training_run_uuid"] != training["run_uuid"]
                    or record["training_record_sha256"] != train_row["record_sha256"]):
                raise RegistryError("campaign v2 validation row must bind its exact frozen training arm")
        if set(validations) != {(f"seed-pair-{i}", role)
                                for i in range(1, 4) for role in ("baseline", "mixed")}:
            raise RegistryError("campaign v2 requires one validation registration per exact training arm")
        for pair in pair_rows.values():
            baseline, mixed = pair["baseline"], pair["mixed"]
            if baseline["ppo_seed"] != mixed["ppo_seed"]:
                raise RegistryError("campaign v2 paired arms must share PPO seed")
            base_probs, mixed_probs = baseline["opponent_probabilities"], mixed["opponent_probabilities"]
            if (base_probs != {"latest": 10_000, "step0": 0, "cruise": 0,
                               "local_mobility": 0, "interceptor": 0}
                    or mixed_probs["latest"] == 10_000
                    or any(mixed_probs[name] == 0 for name in ("step0", "cruise", "local_mobility", "interceptor"))):
                raise RegistryError("campaign v2 must compare latest-only against the frozen diverse opponent pool")
            # The package embeds the per-run run-config (UUID, schedule and arm), so
            # its digest is unique even when source/component identities are shared.
            allowed_differences = {"run_uuid", "strategy_role", "opponent_probabilities",
                                   "schedule_descriptor_sha256", "config_sha256", "train_config",
                                   "package_sha256"}
            if any(baseline.get(key) != mixed.get(key)
                   for key in set(baseline) | set(mixed) if key not in allowed_differences):
                raise RegistryError("campaign v2 paired arms differ in a shared source/component/config identity")
            base_config = _g008_v2_numeric_train_config(baseline["train_config"])
            mixed_config = _g008_v2_numeric_train_config(mixed["train_config"])
            if any(base_config[key] != mixed_config[key]
                   for key in set(base_config) | set(mixed_config)
                   if key not in {"snapshot_opponent_probability", "heuristic_opponent_probability"}):
                raise RegistryError("campaign v2 paired full TrainConfig differs beyond opponent probabilities")
            expected_strategy_probabilities = {
                "baseline": (0.0, 0.0), "mixed": (0.1, 1 / 3),
            }
            if ((base_config["snapshot_opponent_probability"], base_config["heuristic_opponent_probability"])
                    != expected_strategy_probabilities["baseline"]
                    or (mixed_config["snapshot_opponent_probability"],
                        mixed_config["heuristic_opponent_probability"])
                    != expected_strategy_probabilities["mixed"]):
                raise RegistryError("campaign v2 TrainConfig opponent probabilities differ from paired strategies")
            if baseline.get("opponent_probabilities") == mixed.get("opponent_probabilities"):
                raise RegistryError("campaign v2 paired arms must differ in opponent probabilities")
    elif event_type == "g008_competitive_campaign_v2_package_revision_v1":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "amendment_event_index", "amendment_event_hash", "prior_head_hash",
                    "failed_registration_event_index", "failed_registration_event_hash",
                    "failed_attempt_event_index", "failed_attempt_event_hash", "package_bindings"}
        d = _keys(data, expected, event_type)
        binding_keys = {"registration_event_index", "registration_event_hash", "package_sha256",
                        "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256",
                        "trainer_sha256", "engine_sha256"}
        if (d["schema_id"] != "g008-competitive-campaign-v2-package-revision"
                or d["schema_version"] != 1
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "amendment_event_hash", "prior_head_hash", "failed_registration_event_hash",
                    "failed_attempt_event_hash"))
                or d["amendment_event_index"] != 55
                or not isinstance(d["failed_registration_event_index"], int)
                or isinstance(d["failed_registration_event_index"], bool)
                or d["failed_registration_event_index"] not in range(56, 62)
                or not isinstance(d["failed_attempt_event_index"], int)
                or isinstance(d["failed_attempt_event_index"], bool)
                or not isinstance(d["package_bindings"], list) or len(d["package_bindings"]) != 6
                or any(not isinstance(binding, dict) or set(binding) != binding_keys
                       or not isinstance(binding.get("registration_event_index"), int)
                       or isinstance(binding.get("registration_event_index"), bool)
                       or binding["registration_event_index"] not in range(56, 62)
                       or any(not _is_hash(binding.get(key)) for key in binding_keys - {
                           "registration_event_index", "registration_event_hash"})
                       or not _is_hash(binding.get("registration_event_hash"))
                       for binding in d["package_bindings"])
                or [binding["registration_event_index"] for binding in d["package_bindings"]] != list(range(56, 62))):
            raise RegistryError("campaign v2 package revision schema or six-train binding is malformed")
    elif event_type == "g008_competitive_campaign_v2_retry_budget_v1":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "package_revision_event_index", "package_revision_event_hash",
                    "failed_attempt_event_index", "failed_attempt_event_hash",
                    "failed_registration_event_index", "failed_registration_event_hash",
                    "retry_attempt_id", "retry_timeout_seconds", "additional_gpu_seconds",
                    "effective_training_gpu_seconds_cap", "effective_evaluation_gpu_seconds_cap",
                    "effective_total_gpu_seconds_cap", "project_experiment_gpu_seconds_cap",
                    "project_total_gpu_seconds_cap", "protected_final_reserve_gpu_seconds"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-competitive-campaign-v2-retry-budget"
                or d["schema_version"] != 1
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "prior_head_hash", "package_revision_event_hash", "failed_attempt_event_hash",
                    "failed_registration_event_hash"))
                or d["package_revision_event_index"] != 73
                or d["failed_attempt_event_index"] != 72
                or d["failed_registration_event_index"] != 56
                or d["retry_attempt_id"] != "g008-v2-pair1-baseline-a2"
                or d["retry_timeout_seconds"] != 1_200
                or d["additional_gpu_seconds"] != 516
                or d["effective_training_gpu_seconds_cap"] != 7_716
                or d["effective_evaluation_gpu_seconds_cap"] != 4_200
                or d["effective_total_gpu_seconds_cap"] != 11_916
                or any(not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] <= 0
                       for key in ("project_experiment_gpu_seconds_cap", "project_total_gpu_seconds_cap",
                                   "protected_final_reserve_gpu_seconds"))):
            raise RegistryError("campaign v2 retry budget authorization is malformed")
    elif event_type == "g008_competitive_campaign_v2_package_revision_v2":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "prior_package_revision_event_index", "prior_package_revision_event_hash",
                    "prior_retry_budget_event_index", "prior_retry_budget_event_hash",
                    "failed_registration_event_index", "failed_registration_event_hash",
                    "failed_attempt_event_index", "failed_attempt_event_hash", "package_binding"}
        d = _keys(data, expected, event_type)
        binding_keys = {"registration_event_index", "registration_event_hash", "package_sha256",
                        "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256",
                        "trainer_sha256", "engine_sha256"}
        binding = d["package_binding"]
        if (d["schema_id"] != "g008-competitive-campaign-v2-package-revision"
                or d["schema_version"] != 2
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "prior_head_hash", "prior_package_revision_event_hash", "prior_retry_budget_event_hash",
                    "failed_registration_event_hash", "failed_attempt_event_hash"))
                or d["prior_package_revision_event_index"] != 73
                or d["prior_retry_budget_event_index"] != 74
                or d["failed_registration_event_index"] != 56
                or d["failed_attempt_event_index"] != 78
                or not isinstance(binding, dict) or set(binding) != binding_keys
                or binding.get("registration_event_index") != 56
                or not _is_hash(binding.get("registration_event_hash"))
                or any(not _is_hash(binding.get(key)) for key in binding_keys - {
                    "registration_event_index", "registration_event_hash"})
                or binding.get("package_sha256") != "31500d68f6fcbdb4eded6247e20b83d10b450f08915f1bf21e7534b865329906"
                or binding.get("source_sha256") != "3ac7f5bcc24a62f9e5015d2e77bbb270f075802fffc8b2f50bdfe256830f8f52"
                or binding.get("trainer_sha256") != "30f571f9f225b7ce88257e59cea772972488a88f9260b3ef24a3c6a21765f2a7"):
            raise RegistryError("campaign v2 second package revision schema or row-56 binding is malformed")
    elif event_type == "g008_competitive_campaign_v2_retry_budget_v2":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "package_revision_event_index", "package_revision_event_hash",
                    "prior_retry_budget_event_index", "prior_retry_budget_event_hash",
                    "failed_attempt_event_index", "failed_attempt_event_hash",
                    "failed_registration_event_index", "failed_registration_event_hash",
                    "retry_attempt_id", "retry_timeout_seconds", "additional_gpu_seconds",
                    "effective_training_gpu_seconds_cap", "effective_evaluation_gpu_seconds_cap",
                    "effective_total_gpu_seconds_cap", "project_experiment_gpu_seconds_cap",
                    "project_total_gpu_seconds_cap", "protected_final_reserve_gpu_seconds"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-competitive-campaign-v2-retry-budget"
                or d["schema_version"] != 2
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "prior_head_hash", "package_revision_event_hash", "prior_retry_budget_event_hash",
                    "failed_attempt_event_hash", "failed_registration_event_hash"))
                or d["package_revision_event_index"] != 79
                or d["prior_retry_budget_event_index"] != 74
                or d["failed_attempt_event_index"] != 78
                or d["failed_registration_event_index"] != 56
                or d["retry_attempt_id"] != "g008-v2-pair1-baseline-a3"
                or d["retry_timeout_seconds"] != 1_200
                or d["additional_gpu_seconds"] != 36
                or d["effective_training_gpu_seconds_cap"] != 7_752
                or d["effective_evaluation_gpu_seconds_cap"] != 4_200
                or d["effective_total_gpu_seconds_cap"] != 11_952
                or any(not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] <= 0
                       for key in ("project_experiment_gpu_seconds_cap", "project_total_gpu_seconds_cap",
                                   "protected_final_reserve_gpu_seconds"))):
            raise RegistryError("campaign v2 second retry budget authorization is malformed")
    elif event_type == "g008_competitive_campaign_v2_package_revision_v3":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "prior_package_revision_event_index", "prior_package_revision_event_hash",
                    "prior_retry_budget_event_index", "prior_retry_budget_event_hash",
                    "prior_package_revision_v2_event_index", "prior_package_revision_v2_event_hash",
                    "prior_retry_budget_v2_event_index", "prior_retry_budget_v2_event_hash",
                    "failed_registration_event_index", "failed_registration_event_hash",
                    "failed_attempt_event_index", "failed_attempt_event_hash", "package_bindings"}
        d = _keys(data, expected, event_type)
        binding_keys = {"registration_event_index", "registration_event_hash", "package_sha256",
                        "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256",
                        "trainer_sha256", "engine_sha256"}
        bindings = d["package_bindings"]
        if (d["schema_id"] != "g008-competitive-campaign-v2-package-revision" or d["schema_version"] != 3
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "prior_head_hash", "prior_package_revision_event_hash", "prior_retry_budget_event_hash",
                    "prior_package_revision_v2_event_hash", "prior_retry_budget_v2_event_hash",
                    "failed_registration_event_hash", "failed_attempt_event_hash"))
                or any(type(d.get(key)) is not int or d[key] < 0 for key in (
                    "prior_package_revision_event_index", "prior_retry_budget_event_index",
                    "prior_package_revision_v2_event_index", "prior_retry_budget_v2_event_index",
                    "failed_registration_event_index", "failed_attempt_event_index"))
                or not isinstance(bindings, list) or len(bindings) != 5
                or [item.get("registration_event_index") for item in bindings if isinstance(item, dict)]
                   != list(range(57, 62))
                or any(not isinstance(item, dict) or set(item) != binding_keys
                       or type(item.get("registration_event_index")) is not int
                       or not _is_hash(item.get(key)) for item in bindings
                       for key in binding_keys - {"registration_event_index"})):
            raise RegistryError("campaign v2 third package revision schema is malformed")
    elif event_type == "g008_competitive_campaign_v2_package_revision_v4":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_event_index", "prior_head_hash", "rejected_attempt_event_indexes",
                    "rejected_attempt_event_hashes", "package_bindings"}
        d = _keys(data, expected, event_type)
        binding_keys = {"registration_event_index", "registration_event_hash", "package_sha256",
                        "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256",
                        "trainer_sha256", "engine_sha256"}
        bindings = d["package_bindings"]
        if (d["schema_id"] != "g008-competitive-campaign-v2-package-revision" or d["schema_version"] != 4
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or d["prior_head_event_index"] != 123 or not _is_hash(d["prior_head_hash"])
                or d["rejected_attempt_event_indexes"] != [119, 123]
                or not isinstance(d["rejected_attempt_event_hashes"], list)
                or len(d["rejected_attempt_event_hashes"]) != 2
                or any(not _is_hash(value) for value in d["rejected_attempt_event_hashes"])
                or not isinstance(bindings, list) or len(bindings) != 6
                or [item.get("registration_event_index") for item in bindings if isinstance(item, dict)]
                   != list(range(62, 68))
                or any(not isinstance(item, dict) or set(item) != binding_keys
                       or type(item.get("registration_event_index")) is not int
                       or not _is_hash(item.get(key)) for item in bindings
                       for key in binding_keys - {"registration_event_index"})):
            raise RegistryError("campaign v2 validation package revision schema is malformed")
    elif event_type == "g008_competitive_campaign_v2_package_revision_v5":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_event_index", "prior_head_hash", "rejected_attempt_event_indexes",
                    "rejected_attempt_event_hashes", "package_bindings"}
        d = _keys(data, expected, event_type)
        binding_keys = {"registration_event_index", "registration_event_hash", "package_sha256",
                        "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256",
                        "trainer_sha256", "engine_sha256"}
        bindings = d["package_bindings"]
        if (d["schema_id"] != "g008-competitive-campaign-v2-package-revision" or d["schema_version"] != 5
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or d["prior_head_event_index"] != 136 or not _is_hash(d["prior_head_hash"])
                or d["rejected_attempt_event_indexes"] != [119, 123, 128, 132, 136]
                or not isinstance(d["rejected_attempt_event_hashes"], list)
                or len(d["rejected_attempt_event_hashes"]) != 5
                or any(not _is_hash(value) for value in d["rejected_attempt_event_hashes"])
                or not isinstance(bindings, list) or len(bindings) != 6
                or [item.get("registration_event_index") for item in bindings if isinstance(item, dict)]
                   != list(range(62, 68))
                or any(not isinstance(item, dict) or set(item) != binding_keys
                       or type(item.get("registration_event_index")) is not int
                       or not _is_hash(item.get(key)) for item in bindings
                       for key in binding_keys - {"registration_event_index"})):
            raise RegistryError("campaign v2 fifth validation package revision schema is malformed")
    elif event_type == "g008_competitive_campaign_v2_retry_budget_v3":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "package_revision_event_index", "package_revision_event_hash",
                    "prior_retry_budget_event_index", "prior_retry_budget_event_hash",
                    "failed_attempt_event_index", "failed_attempt_event_hash",
                    "failed_registration_event_index", "failed_registration_event_hash",
                    "retry_attempt_id", "retry_timeout_seconds", "additional_gpu_seconds",
                    "effective_training_gpu_seconds_cap", "effective_evaluation_gpu_seconds_cap",
                    "effective_total_gpu_seconds_cap", "project_experiment_gpu_seconds_cap",
                    "project_total_gpu_seconds_cap", "protected_final_reserve_gpu_seconds"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-competitive-campaign-v2-retry-budget" or d["schema_version"] != 3
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "prior_head_hash", "package_revision_event_hash", "prior_retry_budget_event_hash",
                    "failed_attempt_event_hash", "failed_registration_event_hash"))
                or any(type(d.get(key)) is not int or d[key] < 0 for key in (
                    "package_revision_event_index", "prior_retry_budget_event_index",
                    "failed_attempt_event_index", "failed_registration_event_index"))
                or d["failed_registration_event_index"] != 57
                or d["retry_attempt_id"] != "g008-v2-pair1-mixed-a2"
                or d["retry_timeout_seconds"] != 1_200 or d["additional_gpu_seconds"] != 37
                or d["effective_training_gpu_seconds_cap"] != 7_752
                or d["effective_evaluation_gpu_seconds_cap"] != 4_200
                or d["effective_total_gpu_seconds_cap"] != 11_952
                or any(not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] <= 0
                       for key in ("project_experiment_gpu_seconds_cap", "project_total_gpu_seconds_cap",
                                   "protected_final_reserve_gpu_seconds"))):
            raise RegistryError("campaign v2 third retry budget authorization is malformed")
    elif event_type == "g008_competitive_campaign_v2_dispatch_retry_v1":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "rejected_attempt_event_index", "rejected_attempt_event_hash",
                    "registration_event_index", "registration_event_hash", "retry_attempt_id",
                    "expected_current_version", "no_outside_writer_confirmed"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-campaign-v2-no-assignment-retry" or d["schema_version"] != 1
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "prior_head_hash", "rejected_attempt_event_hash", "registration_event_hash"))
                or any(type(d.get(key)) is not int or d[key] < 0 for key in (
                    "rejected_attempt_event_index", "registration_event_index", "expected_current_version"))
                or d["registration_event_index"] != 62
                or d["retry_attempt_id"] != "g008-v2-validation-seed-pair-1-baseline-a2"
                or d["expected_current_version"] != 26
                or d["no_outside_writer_confirmed"] is not True):
            raise RegistryError("campaign v2 dispatch retry authorization is malformed")
    elif event_type == "g008_competitive_campaign_v2_dispatch_retry_v2":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "package_revision_event_index", "package_revision_event_hash",
                    "rejected_attempt_event_index", "rejected_attempt_event_hash",
                    "registration_event_index", "registration_event_hash", "retry_attempt_id",
                    "expected_current_version", "no_outside_writer_confirmed"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-campaign-v2-no-assignment-retry" or d["schema_version"] != 2
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "prior_head_hash", "package_revision_event_hash", "rejected_attempt_event_hash",
                    "registration_event_hash"))
                or d["package_revision_event_index"] != 124 or d["rejected_attempt_event_index"] != 123
                or d["registration_event_index"] != 62
                or d["retry_attempt_id"] != "g008-v2-validation-seed-pair-1-baseline-a3"
                or d["expected_current_version"] != 26 or d["no_outside_writer_confirmed"] is not True):
            raise RegistryError("campaign v2 second dispatch retry authorization is malformed")
    elif event_type == "g008_competitive_campaign_v2_dispatch_retry_v3":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "package_revision_event_index", "package_revision_event_hash",
                    "rejected_attempt_event_index", "rejected_attempt_event_hash",
                    "registration_event_index", "registration_event_hash", "retry_attempt_id",
                    "expected_current_version", "no_outside_writer_confirmed"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-campaign-v2-no-assignment-retry" or d["schema_version"] != 3
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "prior_head_hash", "package_revision_event_hash", "rejected_attempt_event_hash",
                    "registration_event_hash"))
                or d["package_revision_event_index"] != 124 or d["rejected_attempt_event_index"] != 128
                or d["registration_event_index"] != 62
                or d["retry_attempt_id"] != "g008-v2-validation-seed-pair-1-baseline-a4"
                or d["expected_current_version"] != 26 or d["no_outside_writer_confirmed"] is not True):
            raise RegistryError("campaign v2 third dispatch retry authorization is malformed")
    elif event_type == "g008_competitive_campaign_v2_dispatch_retry_v4":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "package_revision_event_index", "package_revision_event_hash",
                    "rejected_attempt_event_index", "rejected_attempt_event_hash",
                    "registration_event_index", "registration_event_hash", "retry_attempt_id",
                    "expected_current_version", "no_outside_writer_confirmed"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-campaign-v2-no-assignment-retry" or d["schema_version"] != 4
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "prior_head_hash", "package_revision_event_hash", "rejected_attempt_event_hash",
                    "registration_event_hash"))
                or d["package_revision_event_index"] != 124 or d["rejected_attempt_event_index"] != 132
                or d["registration_event_index"] != 62
                or d["retry_attempt_id"] != "g008-v2-validation-seed-pair-1-baseline-a5"
                or d["expected_current_version"] != 26 or d["no_outside_writer_confirmed"] is not True):
            raise RegistryError("campaign v2 fourth dispatch retry authorization is malformed")
    elif event_type == "g008_competitive_campaign_v2_dispatch_retry_v5":
        expected = {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                    "prior_head_hash", "package_revision_event_index", "package_revision_event_hash",
                    "rejected_attempt_event_index", "rejected_attempt_event_hash",
                    "registration_event_index", "registration_event_hash", "retry_attempt_id",
                    "expected_current_version", "no_outside_writer_confirmed"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-campaign-v2-no-assignment-retry" or d["schema_version"] != 5
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not _is_hash(d.get(key)) for key in (
                    "prior_head_hash", "package_revision_event_hash", "rejected_attempt_event_hash",
                    "registration_event_hash"))
                or d["package_revision_event_index"] != 137 or d["rejected_attempt_event_index"] != 136
                or d["registration_event_index"] != 62
                or d["retry_attempt_id"] != "g008-v2-validation-seed-pair-1-baseline-a6"
                or d["expected_current_version"] != 26 or d["no_outside_writer_confirmed"] is not True):
            raise RegistryError("campaign v2 fifth dispatch retry authorization is malformed")
    elif event_type == "competitive_training_artifact_v2":
        d = _keys(data, {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                         "training_registration_event_index", "training_registration_event_hash",
                         "training_attempt_event_index", "training_attempt_event_hash", "step",
                         "checkpoint_sha256", "output_manifest_sha256", "metrics_sha256",
                         "package_sha256", "source_sha256", "config_sha256",
                         "schedule_descriptor_sha256"}, event_type)
        if (d["schema_id"] != "g008-training-artifact-attestation" or d["schema_version"] != 2
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < 0 for key in (
                    "training_registration_event_index", "training_attempt_event_index", "step"))
                or d["step"] == 0
                or any(not _is_hash(d[key]) for key in (
                    "training_registration_event_hash", "training_attempt_event_hash", "checkpoint_sha256",
                    "output_manifest_sha256", "metrics_sha256", "package_sha256", "source_sha256",
                    "config_sha256", "schedule_descriptor_sha256"))):
            raise RegistryError("invalid campaign v2 training artifact attestation")
    elif event_type == "competitive_validation_result_v2":
        d = _keys(data, {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                         "training_registration_event_index", "training_registration_event_hash",
                         "training_attempt_event_index", "training_attempt_event_hash",
                         "training_artifact_event_index", "training_artifact_event_hash",
                         "validation_registration_event_index", "validation_registration_event_hash",
                         "validation_attempt_event_index", "validation_attempt_event_hash",
                         "checkpoint_sha256", "report_sha256", "output_manifest_sha256",
                         "game_count", "wins", "draws", "losses", "win_rate",
                         "wilson_95_lower", "wilson_95_upper", "opponent_id"}, event_type)
        if (d["schema_id"] != "g008-competitive-validation-result" or d["schema_version"] != 2
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < 0 for key in (
                    "training_registration_event_index", "training_attempt_event_index",
                    "training_artifact_event_index", "validation_registration_event_index",
                    "validation_attempt_event_index", "game_count", "wins", "draws", "losses"))
                or any(not _is_hash(d[key]) for key in (
                    "training_registration_event_hash", "training_attempt_event_hash",
                    "training_artifact_event_hash", "validation_registration_event_hash",
                    "validation_attempt_event_hash", "checkpoint_sha256", "report_sha256",
                    "output_manifest_sha256"))
                or d["game_count"] != 512 or d["wins"] + d["draws"] + d["losses"] != 512
                or not isinstance(d["opponent_id"], str) or d["opponent_id"] == "open-space"):
            raise RegistryError("invalid campaign v2 per-arm validation result")
        stats = competitive_final_statistics(d["wins"], d["draws"], d["losses"])
        if any(d[key] != stats[key] for key in ("win_rate", "wilson_95_lower", "wilson_95_upper")):
            raise RegistryError("campaign v2 validation Wilson metrics differ from W/D/L")
    elif event_type == "competitive_strategy_comparison_v2":
        d = _keys(data, {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                         "validation_result_event_indices", "validation_result_event_hashes",
                         "paired_score_deltas", "mean_paired_score_delta", "paired_score_ci95_lower",
                         "paired_score_ci95_upper", "max_pair_regression", "selected_strategy",
                         "selected_candidate", "mixed_strategy_passed"}, event_type)
        if (d["schema_id"] != "g008-strategy-comparison" or d["schema_version"] != 2
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or not isinstance(d["validation_result_event_indices"], list)
                or len(d["validation_result_event_indices"]) != 6
                or not isinstance(d["validation_result_event_hashes"], list)
                or len(d["validation_result_event_hashes"]) != 6
                or any(not isinstance(index, int) or isinstance(index, bool) or index < 0
                       for index in d["validation_result_event_indices"])
                or any(not _is_hash(value) for value in d["validation_result_event_hashes"])
                or not isinstance(d["paired_score_deltas"], list) or len(d["paired_score_deltas"]) != 3
                or not isinstance(d["selected_candidate"], dict)
                or d["selected_strategy"] not in {"baseline", "mixed"}
                or not isinstance(d["mixed_strategy_passed"], bool)):
            raise RegistryError("invalid campaign v2 aggregate strategy comparison")
        for key in ("mean_paired_score_delta", "paired_score_ci95_lower",
                    "paired_score_ci95_upper", "max_pair_regression"):
            try:
                value = float(d[key])
            except (TypeError, ValueError, OverflowError) as exc:
                raise RegistryError("invalid campaign v2 paired comparison metric") from exc
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise RegistryError("campaign v2 paired comparison metric is out of range")
        if d["selected_strategy"] != ("mixed" if d["mixed_strategy_passed"] else "baseline"):
            raise RegistryError("campaign v2 selected strategy differs from frozen paired decision")
    elif event_type == "competitive_candidate_freeze_v2":
        d = _keys(data, {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                         "strategy_comparison_event_index", "strategy_comparison_event_hash",
                         "candidate_registration_event_index", "candidate_registration_event_hash",
                         "train_attempt_event_index", "train_attempt_event_hash",
                         "training_artifact_event_index", "training_artifact_event_hash",
                         "validation_registration_event_index", "validation_registration_event_hash",
                         "validation_attempt_event_index", "validation_attempt_event_hash",
                         "validation_result_event_index", "validation_result_event_hash",
                         "checkpoint_sha256", "checkpoint_artifact_sha256", "output_manifest_sha256"}, event_type)
        if (d["schema_id"] != "g008-candidate-freeze" or d["schema_version"] != 2
                or d["campaign_id"] != "g008-competitive-20261021-v2"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < 0 for key in (
                    "strategy_comparison_event_index", "candidate_registration_event_index",
                    "train_attempt_event_index", "training_artifact_event_index",
                    "validation_registration_event_index", "validation_attempt_event_index",
                    "validation_result_event_index"))
                or any(not _is_hash(d[key]) for key in (
                    "strategy_comparison_event_hash", "candidate_registration_event_hash",
                    "train_attempt_event_hash", "training_artifact_event_hash",
                    "validation_registration_event_hash", "validation_attempt_event_hash",
                    "validation_result_event_hash", "checkpoint_sha256",
                    "checkpoint_artifact_sha256", "output_manifest_sha256"))
                or d["checkpoint_artifact_sha256"] != d["checkpoint_sha256"]):
            raise RegistryError("invalid campaign v2 selected candidate freeze")
    elif event_type == "competitive_final_result_v2":
        v1_data = dict(data) if isinstance(data, dict) else data
        if isinstance(v1_data, dict):
            v1_data["schema_version"] = 1
            v1_data["campaign_id"] = "g008-competitive-20261020-v1"
        _validate_data("competitive_final_result", v1_data)
        if data.get("schema_version") != 2 or data.get("campaign_id") != "g008-competitive-20261021-v2":
            raise RegistryError("invalid campaign v2 final result identity")
    elif event_type == "competitive_candidate_freeze":
        d = _keys(data, {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                         "candidate_registration_event_index", "candidate_registration_event_hash",
                         "train_attempt_event_index", "train_attempt_event_hash",
                         "checkpoint_sha256", "checkpoint_artifact_sha256",
                         "output_manifest_sha256"}, event_type)
        if (d["schema_id"] != "g008-candidate-freeze" or d["schema_version"] != 1
                or d["campaign_id"] != "g008-competitive-20261020-v1"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or not isinstance(d["candidate_registration_event_index"], int)
                or isinstance(d["candidate_registration_event_index"], bool)
                or d["candidate_registration_event_index"] < 0
                or not isinstance(d["train_attempt_event_index"], int)
                or isinstance(d["train_attempt_event_index"], bool) or d["train_attempt_event_index"] < 0
                or not _is_hash(d["candidate_registration_event_hash"])
                or not _is_hash(d["train_attempt_event_hash"])
                or not _is_hash(d["checkpoint_sha256"])
                or not _is_hash(d["checkpoint_artifact_sha256"])
                or not _is_hash(d["output_manifest_sha256"])
                or d["checkpoint_artifact_sha256"] != d["checkpoint_sha256"]):
            raise RegistryError("invalid competitive candidate-freeze evidence")
    elif event_type == "competitive_validation_result":
        d = _keys(data, {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                         "candidate_freeze_event_index", "candidate_freeze_event_hash",
                         "validation_registration_event_index", "validation_registration_event_hash",
                         "validation_attempt_event_index", "validation_attempt_event_hash",
                         "candidate_checkpoint_sha256", "report_sha256", "output_manifest_sha256",
                         "reference_report_sha256", "reference_game_count", "reference_wins",
                         "reference_draws", "reference_losses", "noninferiority_margin",
                         "game_count", "wins", "draws", "losses", "win_rate", "wilson_95_lower",
                         "wilson_95_upper", "win_rate_difference_95_lower",
                         "win_rate_difference_95_upper", "passed", "opponent_id"}, event_type)
        if (d["schema_id"] != "g008-candidate-validation-result" or d["schema_version"] != 1
                or d["campaign_id"] != "g008-competitive-20261020-v1"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not isinstance(d[k], int) or isinstance(d[k], bool) or d[k] < 0
                       for k in ("candidate_freeze_event_index", "validation_registration_event_index",
                                 "validation_attempt_event_index", "game_count", "wins", "draws", "losses"))
                or any(not isinstance(d[k], int) or isinstance(d[k], bool) or d[k] < 0
                       for k in ("reference_game_count", "reference_wins", "reference_draws", "reference_losses"))
                or not _is_hash(d["candidate_freeze_event_hash"])
                or not _is_hash(d["validation_registration_event_hash"])
                or not _is_hash(d["validation_attempt_event_hash"])
                or not _is_hash(d["candidate_checkpoint_sha256"])
                or not _is_hash(d["report_sha256"])
                or not _is_hash(d["output_manifest_sha256"])
                or not _is_hash(d["reference_report_sha256"])
                or not isinstance(d["passed"], bool) or not isinstance(d["opponent_id"], str)
                or not d["opponent_id"]):
            raise RegistryError("invalid competitive validation result")
        if (d["reference_report_sha256"] != _G008_VALIDATION_REFERENCE_SHA256
                or (d["reference_game_count"], d["reference_wins"], d["reference_draws"],
                    d["reference_losses"], d["noninferiority_margin"])
                != (512, 217, 51, 244, "0.065000")):
            raise RegistryError("validation result changed the frozen G008 v15 reference or margin")
        try:
            stats = competitive_noninferiority_statistics(d["wins"], d["draws"], d["losses"])
        except (RegistryError, TypeError, ValueError, OverflowError) as exc:
            raise RegistryError("validation statistics cannot be computed") from exc
        if (d["game_count"] != 512 or
                any(d[key] != stats[key] for key in (
                    "reference_report_sha256", "reference_game_count", "reference_wins", "reference_draws",
                    "reference_losses", "noninferiority_margin", "win_rate", "wilson_95_lower",
                    "wilson_95_upper", "win_rate_difference_95_lower", "win_rate_difference_95_upper", "passed"))):
            raise RegistryError("validation result metrics or non-inferiority decision do not match W/D/L")
    elif event_type == "competitive_final_result":
        d = _keys(data, {"schema_id", "schema_version", "campaign_id", "created_at_utc",
                         "candidate_freeze_event_index", "candidate_freeze_event_hash",
                         "final_registration_event_index", "final_registration_event_hash",
                         "final_attempt_event_index", "final_attempt_event_hash",
                         "candidate_checkpoint_sha256", "report_sha256", "output_manifest_sha256", "roster_sha256",
                         "opponent_code_sha256", "game_count", "wins", "draws", "losses",
                         "win_rate", "wilson_95_lower", "wilson_95_upper", "opponent_id", "passed"}, event_type)
        if (d["schema_id"] != "g008-final-evaluation-result" or d["schema_version"] != 1
                or d["campaign_id"] != "g008-competitive-20261020-v1"
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not isinstance(d[k], int) or isinstance(d[k], bool) or d[k] < 0
                       for k in ("candidate_freeze_event_index", "final_registration_event_index",
                                 "final_attempt_event_index", "game_count", "wins", "draws", "losses"))
                or any(not _is_hash(d[k]) for k in ("candidate_freeze_event_hash", "final_registration_event_hash",
                                                     "final_attempt_event_hash", "candidate_checkpoint_sha256",
                                                     "report_sha256", "output_manifest_sha256",
                                                     "roster_sha256", "opponent_code_sha256"))
                or d["game_count"] != 512 or d["wins"] + d["draws"] + d["losses"] != 512
                or d["opponent_id"] != "open-space" or not isinstance(d["passed"], bool)):
            raise RegistryError("invalid final 512-game open-space evaluation result")
        try:
            stats = competitive_final_statistics(d["wins"], d["draws"], d["losses"])
        except (RegistryError, TypeError, ValueError, OverflowError) as exc:
            raise RegistryError("final result Wilson interval cannot be computed") from exc
        if (d["win_rate"], d["wilson_95_lower"], d["wilson_95_upper"]) != (
                stats["win_rate"], stats["wilson_95_lower"], stats["wilson_95_upper"]):
            raise RegistryError("final win rate or Wilson 95% interval does not match W/D/L")
        if d["passed"] is not stats["passed"]:
            raise RegistryError("final pass flag must equal the frozen 60% observed-win criterion")
    elif event_type == "benchmark_artifact_validation_rejected":
        expected = {"schema_id", "schema_version", "cycle_id", "created_at_utc",
                    "attempt_event_index", "attempt_event_hash", "registration_event_index",
                    "registration_event_hash", "artifact_sha256", "output_sha256", "validator_id",
                    "validation_codes"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-benchmark-artifact-validation-rejection" or d["schema_version"] != 1
                or d["cycle_id"] not in {"g008-recovery-cycle-3", "g008-competitive-campaign-v1"}):
            raise RegistryError("unsupported benchmark artifact rejection schema")
        if (not isinstance(d["attempt_event_index"], int) or isinstance(d["attempt_event_index"], bool)
                or d["attempt_event_index"] < 0
                or not isinstance(d["registration_event_index"], int)
                or isinstance(d["registration_event_index"], bool) or d["registration_event_index"] < 0
                or not _is_hash(d["attempt_event_hash"]) or not _is_hash(d["registration_event_hash"])
                or any(value is not None and not _is_hash(value)
                       for value in (d["artifact_sha256"], d["output_sha256"]))
                or not isinstance(d["validator_id"], str) or not d["validator_id"]
                or not isinstance(d["validation_codes"], list) or not d["validation_codes"]
                or any(not isinstance(code, str) or not code for code in d["validation_codes"])
                or len(set(d["validation_codes"])) != len(d["validation_codes"])
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")):
            raise RegistryError("benchmark artifact rejection must bind exact validation evidence")
    elif event_type == "cycle3_retry_policy_amendment_v1":
        expected = {"schema_id", "schema_version", "cycle_id", "created_at_utc", "policy_sha256",
                    "prior_head_hash", "prior_final_event_index", "registration_event_index",
                    "registration_event_hash", "registration_id", "prior_attempt_id",
                    "prior_attempt_event_index", "prior_attempt_event_hash", "prior_terminal_status",
                    "validation_rejection_event_index", "validation_rejection_event_hash"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-cycle3-retry-policy-amendment" or d["schema_version"] != 1
                or d["cycle_id"] != "g008-recovery-cycle-3-retry"
                or d["policy_sha256"] != _CYCLE3_RETRY_POLICY_SHA256):
            raise RegistryError("unsupported Cycle-3 retry policy amendment")
        if (not _is_hash(d["prior_head_hash"]) or not isinstance(d["prior_final_event_index"], int)
                or isinstance(d["prior_final_event_index"], bool) or d["prior_final_event_index"] < 0
                or not isinstance(d["registration_event_index"], int)
                or isinstance(d["registration_event_index"], bool) or d["registration_event_index"] < 0
                or not _is_hash(d["registration_event_hash"])
                or not isinstance(d["registration_id"], str) or not d["registration_id"]
                or not isinstance(d["prior_attempt_id"], str) or not d["prior_attempt_id"]
                or not isinstance(d["prior_attempt_event_index"], int)
                or isinstance(d["prior_attempt_event_index"], bool) or d["prior_attempt_event_index"] < 0
                or not _is_hash(d["prior_attempt_event_hash"])
                or d["prior_terminal_status"] not in {"ERROR", "CANCELLED", "TIMEOUT", "COMPLETE"}
                or not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")):
            raise RegistryError("Cycle-3 retry amendment has invalid exact-head or registration binding")
        rejection_index, rejection_hash = d["validation_rejection_event_index"], d["validation_rejection_event_hash"]
        if d["prior_terminal_status"] == "COMPLETE":
            if (not isinstance(rejection_index, int) or isinstance(rejection_index, bool) or rejection_index < 0
                    or not _is_hash(rejection_hash)):
                raise RegistryError("COMPLETE retry requires explicit artifact-validation rejection evidence")
        elif rejection_index is not None or rejection_hash is not None:
            raise RegistryError("failed benchmark retry cannot bind artifact-validation rejection")
    elif event_type == "exploratory_kernel_observation_v1":
        expected = {
            "schema_id", "schema_version", "cycle_id", "created_at_utc", "kernel_id",
            "version_number", "run_uuid", "terminal_status", "exploratory", "promotable",
            "continuation_event_index", "continuation_event_hash", "artifact_sha256",
            "actual_gpu_seconds", "output_bytes",
        }
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-exploratory-kernel-observation" or d["schema_version"] != 1
                or d["cycle_id"] != "g008-recovery-cycle-3-retry"
                or d["exploratory"] is not True or d["promotable"] is not False):
            raise RegistryError("unsupported or promotable exploratory kernel observation")
        artifacts = d["artifact_sha256"]
        if (not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or not isinstance(d["kernel_id"], str) or not d["kernel_id"]
                or not isinstance(d["version_number"], int) or isinstance(d["version_number"], bool)
                or d["version_number"] < 1
                or not isinstance(d["run_uuid"], str) or not d["run_uuid"]
                or d["terminal_status"] not in {"COMPLETE", "ERROR", "CANCELLED", "TIMEOUT"}
                or not isinstance(d["continuation_event_index"], int)
                or isinstance(d["continuation_event_index"], bool) or d["continuation_event_index"] < 0
                or not _is_hash(d["continuation_event_hash"])
                or any(not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < 0
                       for key in ("actual_gpu_seconds", "output_bytes"))
                or not isinstance(artifacts, dict) or not artifacts
                or any(not isinstance(name, str) or not name or not _is_hash(digest)
                       for name, digest in artifacts.items())):
            raise RegistryError("exploratory kernel observation must bind terminal identity and artifact hashes")
    elif event_type == "cycle3_retry_rebased_authorization_v1":
        expected = {
            "schema_id", "schema_version", "cycle_id", "created_at_utc",
            "prior_head_hash", "prior_final_event_index", "observation_event_index",
            "observation_event_hash", "registration_event_index", "registration_event_hash",
            "registration_id", "kernel_id", "expected_current_version", "observed_run_uuid",
            "observed_terminal_status", "attempt_id", "session_timeout_seconds",
            "reserved_gpu_seconds", "package_sha256", "source_sha256", "config_sha256",
            "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256",
        }
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-cycle3-retry-rebased-authorization" or d["schema_version"] != 1
                or d["cycle_id"] != "g008-recovery-cycle-3-retry"):
            raise RegistryError("unsupported Cycle-3 retry rebase authorization")
        if (not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z")
                or any(not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < 0
                       for key in ("prior_final_event_index", "observation_event_index", "registration_event_index",
                                   "expected_current_version", "session_timeout_seconds", "reserved_gpu_seconds"))
                or any(not _is_hash(d[key]) for key in ("prior_head_hash", "observation_event_hash",
                                                        "registration_event_hash", "package_sha256", "source_sha256",
                                                        "config_sha256", "seed_plan_sha256", "backend_sha256",
                                                        "trainer_sha256", "engine_sha256"))
                or any(not isinstance(d[key], str) or not d[key]
                       for key in ("registration_id", "kernel_id", "observed_run_uuid", "attempt_id"))
                or d["observed_terminal_status"] not in {"COMPLETE", "ERROR", "CANCELLED", "TIMEOUT"}
                or d["session_timeout_seconds"] <= 0
                or d["reserved_gpu_seconds"] != d["session_timeout_seconds"]):
            raise RegistryError("Cycle-3 retry rebase authorization has invalid head, workload, or cap bindings")
    elif event_type == "recovery_cycle_retry_amendment_v1":
        expected = {
            "schema_id", "schema_version", "cycle_id", "created_at_utc", "prior_head_hash",
            "prior_final_event_index", "cycle3_amendment_event_index", "cycle3_amendment_event_hash",
            "proposal_sha256", "retry_registration_id", "registration_kind", "retry_run_uuid", "retry_record_sha256",
            "registration_binding_sha256",
            "preflight_context_sha256", "kernel_id", "current_version_number",
            "current_terminal_status", "current_status_evidence", "quota_gpu_seconds_remaining_before",
            "quota_evidence", "user_authorization",
        }
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-kaggle-recovery-cycle-retry-amendment" or d["schema_version"] != 1
                or d["cycle_id"] != "g008-recovery-cycle-3-retry"):
            raise RegistryError("unsupported cycle-3 retry amendment schema")
        if (d["prior_final_event_index"] != _CYCLE3_RETRY_PARENT_EVENT_INDEX
                or d["prior_head_hash"] != _CYCLE3_RETRY_PARENT_EVENT_HASH
                or d["cycle3_amendment_event_index"] != _CYCLE3_EVENT_INDEX
                or not _is_hash(d["cycle3_amendment_event_hash"])):
            raise RegistryError("cycle-3 retry amendment must bind exact failed attempt and original amendment")
        if (not _is_hash(d["proposal_sha256"])
                or d["proposal_sha256"] != _CYCLE3_RETRY_PROPOSAL_SHA256):
            raise RegistryError("cycle-3 retry amendment does not bind frozen proposal digest")
        if (not isinstance(d["retry_registration_id"], str) or not d["retry_registration_id"]
                or d["registration_kind"] != "benchmark"
                or not isinstance(d["retry_run_uuid"], str) or not d["retry_run_uuid"]
                or not _is_hash(d["retry_record_sha256"])):
            raise RegistryError("cycle-3 retry amendment registration identity or record digest is invalid")
        if (not _is_hash(d["registration_binding_sha256"])
                or d["registration_binding_sha256"] != cycle3_retry_binding_sha256(
                    d["retry_registration_id"], d["registration_kind"], d["retry_run_uuid"], d["retry_record_sha256"])):
            raise RegistryError("cycle-3 retry registration binding digest is invalid")
        if (not _is_hash(d["preflight_context_sha256"])
                or d["preflight_context_sha256"] != cycle3_retry_preflight_context_sha256(d)):
            raise RegistryError("cycle-3 retry preflight context digest is invalid")
        if (not isinstance(d["kernel_id"], str) or not d["kernel_id"]
                or not isinstance(d["current_version_number"], int)
                or isinstance(d["current_version_number"], bool) or d["current_version_number"] <= 0
                or d["current_terminal_status"] not in {"ERROR", "CANCELLED", "TIMEOUT"}
                or d["current_status_evidence"] != {
                    "status": d["current_terminal_status"],
                    "current_version_number": d["current_version_number"], "kernel_id": d["kernel_id"]}):
            raise RegistryError("cycle-3 retry amendment requires exact terminal Kaggle status")
        quota, qev = d["quota_gpu_seconds_remaining_before"], d["quota_evidence"]
        if (not isinstance(quota, int) or isinstance(quota, bool) or quota < 20_000
                or not isinstance(qev, dict)
                or set(qev) != {"remaining_gpu_seconds", "used_gpu_seconds", "source"}
                or qev.get("remaining_gpu_seconds") != quota
                or not isinstance(qev.get("used_gpu_seconds"), int)
                or isinstance(qev.get("used_gpu_seconds"), bool) or qev["used_gpu_seconds"] < 0
                or not isinstance(qev.get("source"), str) or not qev["source"]):
            raise RegistryError("cycle-3 retry requires exact live quota evidence and protected runway")
        authorization = d["user_authorization"]
        if (not isinstance(authorization, dict)
                or set(authorization) != {"proposal_sha256", "record_sha256", "preflight_context_sha256",
                                         "statement", "confirmed_at_utc"}
                or authorization.get("proposal_sha256") != d["proposal_sha256"]
                or authorization.get("record_sha256") != d["retry_record_sha256"]
                or authorization.get("preflight_context_sha256") != d["preflight_context_sha256"]
                or authorization.get("statement") != cycle3_retry_authorization_statement(
                    d["proposal_sha256"], d["retry_record_sha256"], d["preflight_context_sha256"])
                or not isinstance(authorization.get("confirmed_at_utc"), str)
                or not authorization["confirmed_at_utc"].endswith("Z")):
            raise RegistryError("cycle-3 retry requires digest-bound user authorization")
        if not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z"):
            raise RegistryError("cycle-3 retry amendment timestamp must be UTC")
    elif event_type == "recovery_cycle_amendment_v2":
        expected = {"schema_id", "schema_version", "cycle_id", "created_at_utc", "prior_head_hash",
                    "prior_final_event_index", "proposal_sha256", "retired_registration_event_indices",
                    "recovery_registrations", "registration_bindings_sha256", "preflight_context_sha256",
                    "compute_caps_gpu_seconds", "project_gpu_seconds_before",
                    "project_output_bytes_before", "quota_gpu_seconds_remaining_before",
                    "protected_final_reserve_gpu_seconds", "output_caps_bytes",
                    "output_category_reservations_bytes", "kernel_id", "current_version_number",
                    "current_terminal_status", "current_status_evidence", "quota_evidence",
                    "user_authorization"}
        d = _keys(data, expected, event_type)
        if (d["schema_id"] != "g008-kaggle-recovery-cycle-amendment" or d["schema_version"] != 2
                or d["cycle_id"] != "g008-recovery-cycle-3"):
            raise RegistryError("unsupported recovery cycle-3 amendment schema")
        if (d["prior_final_event_index"] != _CYCLE3_EVENT_INDEX - 1
                or d["prior_head_hash"] != _CYCLE3_PARENT_EVENT_HASH):
            raise RegistryError("cycle-3 amendment must bind exact terminal cycle-2 head")
        if not _is_hash(d["proposal_sha256"]) or d["proposal_sha256"] != _CYCLE3_PROPOSAL_SHA256:
            raise RegistryError("cycle-3 amendment does not bind the frozen proposal digest")
        if (not _is_hash(d["registration_bindings_sha256"])
                or d["registration_bindings_sha256"] != cycle3_registration_bindings_sha256(d["recovery_registrations"])):
            raise RegistryError("cycle-3 amendment registration binding digest is invalid")
        if (not _is_hash(d["preflight_context_sha256"])
                or d["preflight_context_sha256"] != cycle3_preflight_context_sha256(d)):
            raise RegistryError("cycle-3 amendment preflight context digest is invalid")
        authorization = d["user_authorization"]
        if (not isinstance(authorization, dict)
                or set(authorization) != {"proposal_sha256", "registration_bindings_sha256", "preflight_context_sha256",
                                         "statement", "confirmed_at_utc"}
                or authorization.get("proposal_sha256") != _CYCLE3_PROPOSAL_SHA256
                or authorization.get("registration_bindings_sha256") != d["registration_bindings_sha256"]
                or authorization.get("preflight_context_sha256") != d["preflight_context_sha256"]
                or not isinstance(authorization.get("statement"), str) or not authorization["statement"]
                or authorization.get("statement") != cycle3_authorization_statement(
                    _CYCLE3_PROPOSAL_SHA256, d["registration_bindings_sha256"], d["preflight_context_sha256"])
                or not isinstance(authorization.get("confirmed_at_utc"), str)
                or not authorization["confirmed_at_utc"].endswith("Z")):
            raise RegistryError("cycle-3 amendment requires authorization bound to the frozen proposal")
        if d["retired_registration_event_indices"] != [21, 22]:
            raise RegistryError("cycle-3 amendment must retire both cycle-2 registrations")
        if d["compute_caps_gpu_seconds"] != {"benchmark": _CYCLE3_BENCHMARK_GPU_SECONDS,
                                             "train": _CYCLE3_TRAIN_GPU_SECONDS,
                                             "project_experiment": _CYCLE3_PROJECT_EXPERIMENT_GPU_SECONDS,
                                             "project_total": _CYCLE3_PROJECT_TOTAL_GPU_SECONDS}:
            raise RegistryError("cycle-3 compute caps differ from the provisional reviewed budget")
        if (d["project_gpu_seconds_before"] != 8664 or d["project_output_bytes_before"] != 34685):
            raise RegistryError("cycle-3 baseline must count cycle-1/2 committed use exactly once")
        if (d["project_gpu_seconds_before"] + _CYCLE3_BENCHMARK_GPU_SECONDS
                + _CYCLE3_TRAIN_GPU_SECONDS > _CYCLE3_PROJECT_EXPERIMENT_GPU_SECONDS):
            raise RegistryError("cycle-3 projected GPU use exceeds experiment cap")
        if (d["project_gpu_seconds_before"] + _CYCLE3_BENCHMARK_GPU_SECONDS
                + _CYCLE3_TRAIN_GPU_SECONDS + _CYCLE3_FINAL_RESERVE_GPU_SECONDS > _CYCLE3_PROJECT_TOTAL_GPU_SECONDS):
            raise RegistryError("cycle-3 projected GPU use exceeds project cap with reserve")
        if not isinstance(d["kernel_id"], str) or not d["kernel_id"]:
            raise RegistryError("cycle-3 amendment requires exact Kaggle kernel identity")
        if d["current_version_number"] != 12 or isinstance(d["current_version_number"], bool):
            raise RegistryError("cycle-3 amendment must bind current Kaggle version 12")
        if d["current_terminal_status"] != "ERROR":
            raise RegistryError("cycle-3 amendment must bind reconciled cycle-2 ERROR status")
        if d["current_status_evidence"] != {"status": "ERROR", "current_version_number": 12,
                                             "kernel_id": d["kernel_id"]}:
            raise RegistryError("cycle-3 status evidence must exactly bind version 12 and kernel")
        quota = d["quota_gpu_seconds_remaining_before"]
        qev = d["quota_evidence"]
        if (not isinstance(quota, int) or isinstance(quota, bool) or quota < 0
                or not isinstance(qev, dict)
                or set(qev) != {"remaining_gpu_seconds", "used_gpu_seconds", "source"}
                or qev.get("remaining_gpu_seconds") != quota
                or not isinstance(qev.get("used_gpu_seconds"), int)
                or isinstance(qev.get("used_gpu_seconds"), bool) or qev["used_gpu_seconds"] < 0
                or not isinstance(qev.get("source"), str) or not qev["source"]):
            raise RegistryError("cycle-3 amendment requires exact live quota evidence")
        if quota < (_CYCLE3_BENCHMARK_GPU_SECONDS + _CYCLE3_TRAIN_GPU_SECONDS
                    + _CYCLE3_FINAL_RESERVE_GPU_SECONDS):
            raise RegistryError("cycle-3 quota cannot cover registered caps and protected reserve")
        if d["protected_final_reserve_gpu_seconds"] != _CYCLE3_FINAL_RESERVE_GPU_SECONDS:
            raise RegistryError("cycle-3 must retain the protected final reserve")
        if d["output_caps_bytes"] != {"aggregate": _RECOVERY_MAX_OUTPUT_BYTES,
                                      "per_run": _RECOVERY_MAX_RUN_OUTPUT_BYTES}:
            raise RegistryError("cycle-3 output caps differ from the recovery envelope")
        categories = {"benchmark": _CYCLE3_OUTPUT_CATEGORY_CAP_BYTES,
                      "training": _CYCLE3_OUTPUT_CATEGORY_CAP_BYTES,
                      "validation": _CYCLE3_OUTPUT_CATEGORY_CAP_BYTES,
                      "final": _CYCLE3_OUTPUT_CATEGORY_CAP_BYTES}
        if d["output_category_reservations_bytes"] != categories:
            raise RegistryError("cycle-3 output category reservations differ from the recovery envelope")
        if d["project_output_bytes_before"] + sum(categories.values()) > _RECOVERY_MAX_OUTPUT_BYTES:
            raise RegistryError("cycle-3 output reservations exceed aggregate cap")
        regs = d["recovery_registrations"]
        if (not isinstance(regs, list) or len(regs) != 2
                or sorted(r.get("registration_kind") for r in regs if isinstance(r, dict)) != ["benchmark", "train"]):
            raise RegistryError("cycle-3 amendment must name exactly one benchmark and one train")
        if any(set(r) != {"registration_id", "registration_kind", "run_uuid", "record_sha256"}
               or not all(isinstance(r[k], str) and r[k] for k in r) for r in regs):
            raise RegistryError("cycle-3 registration identities are malformed")
        if any(not _is_hash(r["record_sha256"]) for r in regs):
            raise RegistryError("cycle-3 registration record binding is not a SHA-256")
        if (len({r["registration_id"] for r in regs}) != 2
                or len({r["run_uuid"] for r in regs}) != 2):
            raise RegistryError("cycle-3 registration IDs and UUIDs must be unique")
        if not isinstance(d["created_at_utc"], str) or not d["created_at_utc"].endswith("Z"):
            raise RegistryError("cycle-3 amendment timestamp must be UTC")
    elif event_type == "recovery_cycle_amendment":
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
        if "cycle_id" in d and d["cycle_id"] not in {"g008-recovery-cycle-2", "g008-recovery-cycle-3", "g008-competitive-campaign-v1", "g008-competitive-campaign-v2"}:
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
        cycle_ids = {"g008-recovery-cycle-2", "g008-recovery-cycle-3", "g008-competitive-campaign-v1", "g008-competitive-campaign-v2"}
        cycle_schema = required | {"cycle_id"}
        d = _keys(data, cycle_schema if isinstance(data, dict) and set(data) == cycle_schema else required, event_type)
        if "cycle_id" in d and d["cycle_id"] not in cycle_ids:
            raise RegistryError("unsupported workload cycle_id")
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
        cycle_schema = required | {"cycle_id"}
        d = _keys(data, cycle_schema if isinstance(data, dict) and set(data) == cycle_schema else required, event_type)
        if "cycle_id" in d and d["cycle_id"] not in {"g008-recovery-cycle-2", "g008-recovery-cycle-3", "g008-competitive-campaign-v1", "g008-competitive-campaign-v2"}:
            raise RegistryError("unsupported workload cycle_id")
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
        cycle_schema = required | {"cycle_id"}
        d = _keys(data, cycle_schema if isinstance(data, dict) and set(data) == cycle_schema else required, event_type)
        if "cycle_id" in d and d["cycle_id"] not in {"g008-recovery-cycle-2", "g008-recovery-cycle-3", "g008-competitive-campaign-v1", "g008-competitive-campaign-v2"}:
            raise RegistryError("unsupported workload cycle_id")
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
        cycle_schema = required | {"cycle_id"}
        d = _keys(data, cycle_schema if isinstance(data, dict) and set(data) == cycle_schema else required, event_type)
        if "cycle_id" in d and d["cycle_id"] not in {"g008-recovery-cycle-2", "g008-recovery-cycle-3", "g008-competitive-campaign-v1", "g008-competitive-campaign-v2"}:
            raise RegistryError("unsupported workload cycle_id")
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
            v2_campaign_registration = (data.get("cycle_id") == "g008-competitive-campaign-v2"
                                       and any(previous["event_type"] ==
                                               "g008_competitive_campaign_amendment_v2"
                                               for previous in events))
            if lock_hash is not None and not v2_campaign_registration:
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
    cycle3_indices = [i for i, event in enumerate(events)
                      if event["event_type"] == "recovery_cycle_amendment_v2"]
    cycle2_indices = [i for i, event in enumerate(events) if event["event_type"] == "recovery_cycle_amendment"]
    campaign_v2_indices = [i for i, event in enumerate(events)
                           if event["event_type"] == "g008_competitive_campaign_amendment_v2"]
    if cycle3_indices:
        campaign_indices = [i for i, event in enumerate(events)
                            if event["event_type"] == "g008_competitive_campaign_amendment_v1"]
        if campaign_indices and campaign_v2_indices:
            raise RegistryError("campaign v1 and v2 are separate immutable lifecycle branches")
        if campaign_v2_indices:
            _validate_g008_competitive_campaign_v2(events)
        elif campaign_indices:
            _validate_g008_competitive_campaign(events)
        else:
            _validate_recovery_cycle3_suffix(events)
    elif cycle2_indices:
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


def _validate_recovery_cycle3_suffix(events: list[dict[str, Any]]) -> None:
    """Replay Cycle 3 after validating the immutable Cycle 1 and 2 prefixes."""
    if (len(events) > _CYCLE3_RETRY_PARENT_EVENT_INDEX + 1
            and events[_CYCLE3_RETRY_PARENT_EVENT_INDEX + 1]["event_type"] == "recovery_cycle_retry_amendment_v1"):
        # Validate the original Cycle-3 suffix exactly as it was before retry support,
        # then replay the appended retry as its own constrained continuation.
        _validate_recovery_cycle3_suffix(events[:_CYCLE3_RETRY_PARENT_EVENT_INDEX + 1])
        _validate_recovery_cycle3_retry_suffix(events)
        return
    indices = [i for i, event in enumerate(events)
               if event["event_type"] == "recovery_cycle_amendment_v2"]
    if indices != [_CYCLE3_EVENT_INDEX] or len(events) <= _CYCLE3_EVENT_INDEX:
        raise RegistryError("cycle-3 amendment must be the sole v2 amendment at index 27")
    parent = events[26]
    if parent["event_hash"] != _CYCLE3_PARENT_EVENT_HASH or parent["event_type"] != "workload_attempt":
        raise RegistryError("cycle-3 requires exact terminal cycle-2 event 26")
    _validate_recovery_suffix(events[:20])
    _validate_recovery_cycle2_suffix(events[:27])
    amendment_event = events[27]
    amendment = amendment_event["event_data"]
    c2 = events[20]["event_data"]
    terminal = parent["event_data"]
    if terminal["outcome"] != "terminal":
        raise RegistryError("cycle-3 parent must be an assigned terminal cycle-2 attempt")
    if (amendment["project_gpu_seconds_before"] != c2["project_gpu_seconds_before"] + terminal["actual_gpu_seconds"]
            or amendment["project_output_bytes_before"] != c2["project_output_bytes_before"] + terminal["output_bytes"]):
        raise RegistryError("cycle-3 baseline must count committed cycle-2 use exactly once")
    if (amendment["kernel_id"] != terminal["terminal_metadata"]["kernel_id"]
            or amendment["current_version_number"] != terminal["assigned_version"]
            or amendment["current_terminal_status"] != terminal["terminal_status"]
            or amendment["current_status_evidence"] != {
                "status": terminal["terminal_status"], "current_version_number": terminal["assigned_version"],
                "kernel_id": terminal["terminal_metadata"]["kernel_id"]}):
        raise RegistryError("cycle-3 amendment does not bind exact cycle-2 terminal status")

    named = {r["registration_id"]: r for r in amendment["recovery_registrations"]}
    registrations: dict[int, dict[str, Any]] = {}
    variants_by_kind: dict[str, dict[str, dict[str, Any]]] = {}
    ids: set[str] = set()
    attempts: dict[str, dict[str, Any]] = {}
    artifact_index = lock_index = None
    attestation_index = None
    artifact_data = None
    current_version = amendment["current_version_number"]
    for i, event in enumerate(events[28:], 28):
        typ, data = event["event_type"], event["event_data"]
        if typ == "recovery_cycle_amendment_v2":
            raise RegistryError("a second Cycle-3 amendment is forbidden")
        if typ == "registration":
            expected_kind = "benchmark" if i == 28 else "train"
            if i > 29 or len(registrations) >= 2 or data.get("cycle_id") != "g008-recovery-cycle-3" or data["registration_kind"] != expected_kind:
                raise RegistryError("cycle-3 permits exactly two consecutive benchmark/train registrations")
            rid, kind, record = data["registration_id"], data["registration_kind"], data["record"]
            rule = named.get(rid)
            if rule is None or rid in ids or rule["registration_kind"] != kind or record.get("run_uuid") != rule["run_uuid"]:
                raise RegistryError("cycle-3 registration differs from its amended identity")
            if cycle3_registration_record_sha256(record) != rule["record_sha256"]:
                raise RegistryError("cycle-3 registration record differs from its user-authorized package binding")
            prior_ids = {e["event_data"]["registration_id"] for e in events[:i] if e["event_type"] == "registration"}
            prior_uuids = {e["event_data"]["record"].get("run_uuid") for e in events[:i] if e["event_type"] == "registration"}
            if rid in prior_ids or rule["run_uuid"] in prior_uuids:
                raise RegistryError("cycle-3 registration reuses a retired identity")
            if (record.get("cycle_id") != "g008-recovery-cycle-3"
                    or record.get("recovery_cycle_amendment_event_index") != 27
                    or record.get("recovery_cycle_amendment_event_hash") != amendment_event["event_hash"]):
                raise RegistryError("cycle-3 registration does not pin its amendment")
            expected_cap = amendment["compute_caps_gpu_seconds"][kind]
            timeout = 2800 if kind == "benchmark" else 7200
            output_cap = 100_000_000 if kind == "benchmark" else _CYCLE3_OUTPUT_CATEGORY_CAP_BYTES
            if (record.get("workload_gpu_seconds_cap") != expected_cap
                    or record.get("session_timeout_seconds") != timeout
                    or record.get("workload_output_bytes_cap") != output_cap
                    or record.get("accelerator_count") != 1 or record.get("machine_shape") != "NvidiaTeslaT4"
                    or record.get("kernel_id") != amendment["kernel_id"]):
                raise RegistryError("cycle-3 registration changed frozen resources or kernel")
            for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256"):
                if not _is_hash(record.get(key)):
                    raise RegistryError("cycle-3 registration has invalid workload hashes")
            variant_rows = record.get("implementation_variants" if kind == "benchmark" else "allowed_implementation_variants")
            variant_keys = {"variant_id", "package_sha256", "source_sha256", "trainer_sha256", "backend_sha256",
                            "engine_sha256", "config_sha256", "seed_plan_sha256"}
            if not isinstance(variant_rows, list) or len(variant_rows) != 2:
                raise RegistryError("cycle-3 registration must freeze exactly two implementation variants")
            parsed: dict[str, dict[str, Any]] = {}
            for variant in variant_rows:
                if (not isinstance(variant, dict) or set(variant) != variant_keys
                        or variant.get("variant_id") not in {"incumbent", "candidate"}
                        or variant["variant_id"] in parsed):
                    raise RegistryError("cycle-3 variants must bind incumbent and candidate exactly once")
                if any(not _is_hash(variant[key]) for key in variant_keys - {"variant_id"}):
                    raise RegistryError("cycle-3 variant has invalid implementation/runtime hash")
                if variant["config_sha256"] != record["config_sha256"]:
                    raise RegistryError("cycle-3 variant changed registered config hash")
                parsed[variant["variant_id"]] = variant
            variants_by_kind[kind] = parsed
            registrations[i] = {"event": event, "record": record, "kind": kind}
            ids.add(rid)
            if set(variants_by_kind) == {"benchmark", "train"}:
                shared = ("source_sha256", "trainer_sha256", "backend_sha256", "engine_sha256",
                          "config_sha256", "seed_plan_sha256")
                if any(variants_by_kind["benchmark"][variant_id][key] != variants_by_kind["train"][variant_id][key]
                       for variant_id in ("incumbent", "candidate") for key in shared):
                    raise RegistryError("cycle-3 benchmark/train registrations differ in shared variant hashes")
            continue
        if typ in {"workload_preflight", "workload_push_dispatched", "workload_push_accepted", "workload_attempt"}:
            if len(registrations) != 2 or data.get("cycle_id") != "g008-recovery-cycle-3":
                raise RegistryError("both cycle-3 registrations must precede execution")
            aid = data["attempt_id"]
            if typ == "workload_preflight":
                ri = data["registration_event_index"]
                reg = registrations.get(ri)
                if aid in attempts or reg is None or reg["event"]["event_hash"] != data["registration_event_hash"]:
                    raise RegistryError("cycle-3 preflight reuses an attempt or registration")
                if data["expected_current_version"] != current_version or data["kernel_id"] != reg["record"]["kernel_id"]:
                    raise RegistryError("cycle-3 preflight does not bind current version/kernel")
                if data["session_timeout_seconds"] != reg["record"]["session_timeout_seconds"]:
                    raise RegistryError("cycle-3 preflight timeout differs from registration")
                if reg["kind"] == "benchmark" and (data["implementation_variant_id"] != "paired" or artifact_index is not None):
                    raise RegistryError("cycle-3 benchmark must be paired and precede artifact")
                if reg["kind"] == "train" and (artifact_index is None or lock_index is None or attestation_index is None):
                    raise RegistryError("cycle-3 training requires benchmark artifact, lock, and attestation")
                if reg["kind"] == "train":
                    selected = (artifact_data or {}).get("selection_evidence", {}).get("selected_variant_id")
                    if data["implementation_variant_id"] != selected:
                        raise RegistryError("cycle-3 training differs from benchmark-selected variant")
                    if data["lock_head_hash"] != events[lock_index]["event_hash"]:
                        raise RegistryError("cycle-3 training preflight must bind exact lock")
                    expected_hashes = variants_by_kind["train"].get(selected)
                else:
                    if data["lock_head_hash"] is not None:
                        raise RegistryError("cycle-3 benchmark preflight cannot bind future lock")
                    expected_hashes = reg["record"]
                for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                            "backend_sha256", "trainer_sha256", "engine_sha256"):
                    if expected_hashes is None or data[key] != expected_hashes[key]:
                        raise RegistryError(f"cycle-3 preflight changed registered {key}")
                attempts[aid] = {"phase": "preflight", "reg": ri, "data": data,
                                 "preflight_index": i, "preflight_hash": event["event_hash"], "last": i}
                continue
            state = attempts.get(aid)
            if state is None or state["phase"] == "terminal" or i != state["last"] + 1:
                raise RegistryError("cycle-3 attempt lifecycle must be contiguous")
            if data["registration_event_index"] != state["reg"]:
                raise RegistryError("cycle-3 lifecycle changed registration binding")
            if typ == "workload_push_dispatched":
                if (state["phase"] != "preflight" or data["preflight_event_index"] != state["preflight_index"]
                        or data["preflight_event_hash"] != state["preflight_hash"]):
                    raise RegistryError("cycle-3 dispatch must immediately follow preflight")
                for key in ("kernel_id", "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                            "backend_sha256", "trainer_sha256", "engine_sha256", "session_timeout_seconds",
                            "accelerator_count", "reserved_gpu_seconds", "implementation_variant_id", "lock_head_hash"):
                    if data[key] != state["data"][key]:
                        raise RegistryError("cycle-3 dispatch changed preflight workload binding")
                state.update(phase="dispatched", dispatch=i, last=i)
            elif typ == "workload_push_accepted":
                if state["phase"] != "dispatched" or data["dispatch_event_index"] != state["dispatch"]:
                    raise RegistryError("cycle-3 acceptance must bind dispatch")
                if (data["preflight_event_index"] != state["preflight_index"]
                        or data["preflight_event_hash"] != state["preflight_hash"]
                        or data["implementation_variant_id"] != state["data"]["implementation_variant_id"]
                        or data["lock_head_hash"] != state["data"]["lock_head_hash"]):
                    raise RegistryError("cycle-3 acceptance changed preflight/variant binding")
                if data["assigned_version"] <= state["data"]["expected_current_version"]:
                    raise RegistryError("cycle-3 acceptance must assign a new version")
                state.update(phase="accepted", accepted=i, version=data["assigned_version"], last=i)
            else:
                if state["phase"] not in {"dispatched", "accepted"}:
                    raise RegistryError("cycle-3 terminal event lacks dispatch")
                if state["phase"] == "accepted" and (data["outcome"] != "terminal" or data["accepted_event_index"] != state["accepted"] or data["assigned_version"] != state["version"]):
                    raise RegistryError("cycle-3 attempt differs from accepted receipt")
                if state["phase"] == "dispatched" and data["outcome"] != "pre_assignment_rejected":
                    raise RegistryError("cycle-3 dispatched attempt needs acceptance or proven rejection")
                if state["phase"] == "dispatched":
                    no_assignment = data["no_assignment_metadata"]
                    if (data["rejection_evidence"].get("server_rejected_before_assignment") is not True
                            or no_assignment.get("no_version_assigned") is not True
                            or no_assignment.get("pre_push_current_version") != state["data"]["expected_current_version"]
                            or no_assignment.get("post_push_current_version") != state["data"]["expected_current_version"]
                            or data["actual_gpu_seconds"] != 0):
                        raise RegistryError("cycle-3 pre-assignment rejection lacks exact no-assignment proof")
                rec = registrations[state["reg"]]["record"]
                if (data["preflight_event_index"] != state["preflight_index"]
                        or data["preflight_event_hash"] != state["preflight_hash"]
                        or data["dispatch_event_index"] != state["dispatch"]
                        or data["dispatch_event_hash"] != events[state["dispatch"]]["event_hash"]
                        or data["implementation_variant_id"] != state["data"]["implementation_variant_id"]):
                    raise RegistryError("cycle-3 terminal attempt changed preflight/dispatch/variant binding")
                for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                            "backend_sha256", "trainer_sha256", "engine_sha256"):
                    if data[key] != state["data"][key]:
                        raise RegistryError("cycle-3 terminal attempt changed frozen workload hash")
                if data["actual_gpu_seconds"] > rec["session_timeout_seconds"] or data["output_bytes"] > rec["workload_output_bytes_cap"]:
                    raise RegistryError("cycle-3 actual use exceeds registered timeout/output")
                if data["outcome"] == "terminal":
                    current_version = data["assigned_version"]
                state.update(phase="terminal", terminal=i, terminal_data=data, last=i)
            continue
        if typ == "benchmark_artifact":
            ai = data.get("attempt_event_index")
            if artifact_index is not None or lock_index is not None or not isinstance(ai, int) or isinstance(ai, bool) or ai >= i:
                raise RegistryError("cycle-3 benchmark artifact is duplicate or malformed")
            if not any(s["phase"] == "terminal" and s.get("terminal") == ai
                       and s["terminal_data"].get("terminal_status") == "COMPLETE"
                       and events[ai]["event_hash"] == data.get("attempt_event_hash")
                       and data.get("registration_event_index") == s["reg"]
                       and data.get("registration_event_hash") == events[s["reg"]]["event_hash"]
                       for s in attempts.values()):
                raise RegistryError("cycle-3 artifact must bind successful terminal benchmark")
            artifact_index, artifact_data = i, data
            continue
        if typ == "registration_closed":
            if artifact_index is None or i != artifact_index + 1:
                raise RegistryError("cycle-3 artifact must immediately precede lock")
            if data != {"pre_lock_head_hash": events[artifact_index]["event_hash"],
                        "pre_lock_final_event_index": artifact_index}:
                raise RegistryError("cycle-3 lock must pin exact benchmark artifact prefix")
            lock_index = i
            continue
        if typ == "benchmark_attestation":
            if lock_index is None or i != lock_index + 1 or artifact_data is None:
                raise RegistryError("cycle-3 attestation must immediately follow lock")
            if data != {**artifact_data, "lock_head_hash": events[lock_index]["event_hash"]}:
                raise RegistryError("cycle-3 attestation must bind exact artifact and lock")
            attestation_index = i
            continue
        raise RegistryError("event type is not allowed in cycle-3 suffix")
    if attempts and ids != set(named):
        raise RegistryError("both cycle-3 registrations must precede execution")
    pending = [s for s in attempts.values() if s["phase"] != "terminal"]
    if pending and (len(pending) != 1 or pending[0]["last"] != len(events) - 1):
        raise RegistryError("pending cycle-3 attempt blocks later events")
    actual_gpu = {"benchmark": 0, "train": 0}
    reserved_gpu = {"benchmark": 0, "train": 0}
    actual_output = {"benchmark": 0, "training": 0}
    reserved_output = {"benchmark": 0, "training": 0}
    for state in attempts.values():
        kind = registrations[state["reg"]]["kind"]
        category = "benchmark" if kind == "benchmark" else "training"
        if state["phase"] == "terminal":
            actual_gpu[kind] += state["terminal_data"]["actual_gpu_seconds"]
            actual_output[category] += state["terminal_data"]["output_bytes"]
        else:
            record = registrations[state["reg"]]["record"]
            reserved_gpu[kind] += record["session_timeout_seconds"]
            reserved_output[category] += record["workload_output_bytes_cap"]
    for kind in actual_gpu:
        if actual_gpu[kind] + reserved_gpu[kind] > amendment["compute_caps_gpu_seconds"][kind]:
            raise RegistryError("cycle-3 workload GPU use/reservation exceeds cap")
    for category in actual_output:
        if actual_output[category] + reserved_output[category] > amendment["output_category_reservations_bytes"][category]:
            raise RegistryError("cycle-3 workload output use/reservation exceeds category cap")
    project_gpu = amendment["project_gpu_seconds_before"] + sum(actual_gpu.values()) + sum(reserved_gpu.values())
    if (project_gpu > _CYCLE3_PROJECT_EXPERIMENT_GPU_SECONDS
            or project_gpu + _CYCLE3_FINAL_RESERVE_GPU_SECONDS > _CYCLE3_PROJECT_TOTAL_GPU_SECONDS):
        raise RegistryError("cycle-3 actual/reserved GPU use exceeds project budget")
    if amendment["project_output_bytes_before"] + sum(amendment["output_category_reservations_bytes"].values()) > _RECOVERY_MAX_OUTPUT_BYTES:
        raise RegistryError("cycle-3 output reservations exceed aggregate cap")


def _validate_recovery_cycle3_retry_suffix(events: list[dict[str, Any]]) -> None:
    """Replay the one authorized benchmark retry after the terminal Cycle-3 attempt."""
    amendment_event = events[34]
    original_amendment = events[27]
    parent = events[33]
    if (amendment_event["event_type"] != "recovery_cycle_retry_amendment_v1"
            or parent["event_type"] != "workload_attempt"
            or parent["event_hash"] != _CYCLE3_RETRY_PARENT_EVENT_HASH
            or parent["event_data"]["outcome"] != "terminal"
            or parent["event_data"]["terminal_status"] not in {"ERROR", "CANCELLED", "TIMEOUT"}):
        raise RegistryError("cycle-3 retry requires the exact terminal failed benchmark at event 33")
    amendment = amendment_event["event_data"]
    _validate_data("recovery_cycle_retry_amendment_v1", amendment)
    failed = parent["event_data"]
    original = original_amendment["event_data"]
    if (amendment["prior_head_hash"] != parent["event_hash"]
            or amendment["cycle3_amendment_event_hash"] != original_amendment["event_hash"]
            or amendment["cycle3_amendment_event_index"] != 27
            or amendment["kernel_id"] != failed["terminal_metadata"]["kernel_id"]
            or amendment["current_version_number"] != failed["assigned_version"]
            or amendment["current_terminal_status"] != failed["terminal_status"]
            or amendment["current_status_evidence"] != {
                "status": failed["terminal_status"], "current_version_number": failed["assigned_version"],
                "kernel_id": failed["terminal_metadata"]["kernel_id"]}):
        raise RegistryError("cycle-3 retry does not bind exact failed attempt, amendment, and live status")
    if len(events) < 36:
        raise RegistryError("cycle-3 retry amendment must be followed by its benchmark registration")
    registration_event = events[35]
    if registration_event["event_type"] != "registration":
        raise RegistryError("cycle-3 retry benchmark registration must immediately follow amendment")
    data = registration_event["event_data"]
    record = data["record"]
    if (data.get("cycle_id") != "g008-recovery-cycle-3" or data["registration_kind"] != "benchmark"
            or data["registration_id"] != amendment["retry_registration_id"]
            or record.get("run_uuid") != amendment["retry_run_uuid"]
            or cycle3_retry_registration_record_sha256(record) != amendment["retry_record_sha256"]
            or record.get("cycle_id") != "g008-recovery-cycle-3"
            or record.get("cycle3_retry_amendment_event_index") != 34
            or record.get("cycle3_retry_amendment_event_hash") != amendment_event["event_hash"]
            or record.get("recovery_cycle_amendment_event_index") != 27
            or record.get("recovery_cycle_amendment_event_hash") != original_amendment["event_hash"]):
        raise RegistryError("cycle-3 retry registration differs from authorized identity, record, or pointers")
    old_ids = {event["event_data"]["registration_id"] for event in events[:35]
               if event["event_type"] == "registration"}
    old_uuids = {event["event_data"]["record"].get("run_uuid") for event in events[:35]
                 if event["event_type"] == "registration"}
    if amendment["retry_registration_id"] in old_ids or amendment["retry_run_uuid"] in old_uuids:
        raise RegistryError("cycle-3 retry must use a fresh registration ID and UUID")
    if (record.get("workload_gpu_seconds_cap") != original["compute_caps_gpu_seconds"]["benchmark"]
            or record.get("session_timeout_seconds") != 2800
            or record.get("workload_output_bytes_cap") != 100_000_000
            or record.get("accelerator_count") != 1 or record.get("machine_shape") != "NvidiaTeslaT4"
            or record.get("kernel_id") != original["kernel_id"]):
        raise RegistryError("cycle-3 retry changed frozen benchmark resources or kernel")
    for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                "backend_sha256", "trainer_sha256", "engine_sha256"):
        if not _is_hash(record.get(key)):
            raise RegistryError("cycle-3 retry has invalid workload hashes")
    variants = record.get("implementation_variants")
    train_event = events[29]
    train_variants = {row["variant_id"]: row for row in train_event["event_data"]["record"]["allowed_implementation_variants"]}
    if not isinstance(variants, list) or len(variants) != 2:
        raise RegistryError("cycle-3 retry must preserve the two registered implementation variants")
    parsed = {}
    for variant in variants:
        if (not isinstance(variant, dict) or variant.get("variant_id") not in {"incumbent", "candidate"}
                or variant["variant_id"] in parsed):
            raise RegistryError("cycle-3 retry variant IDs must be unique incumbent and candidate")
        parsed[variant["variant_id"]] = variant
        for key in ("source_sha256", "trainer_sha256", "engine_sha256", "backend_sha256",
                    "config_sha256", "seed_plan_sha256"):
            if variant.get(key) != train_variants[variant["variant_id"]][key]:
                raise RegistryError("cycle-3 retry changed registered training source/component hashes")
    if set(parsed) != {"incumbent", "candidate"}:
        raise RegistryError("cycle-3 retry must include incumbent and candidate variants")

    retry_attempts: dict[str, dict[str, Any]] = {}
    retry_rejections: dict[int, dict[str, Any]] = {}
    continuation: dict[str, Any] | None = None
    rebase_authorization: dict[str, Any] | None = None
    artifact = lock = attestation = None
    train_state = None
    current_version = amendment["current_version_number"]
    for i, event in enumerate(events[36:], 36):
        typ, ed = event["event_type"], event["event_data"]
        if typ == "cycle3_retry_policy_amendment_v1":
            if continuation is not None or artifact is not None or lock is not None or train_state is not None:
                raise RegistryError("Cycle-3 retry amendment is unexpected after pending amendment, artifact, or training")
            prior = retry_attempts.get(ed.get("prior_attempt_id"))
            if (prior is None or prior["phase"] != "terminal"
                    or prior["terminal"] != ed.get("prior_attempt_event_index")
                    or events[prior["terminal"]]["event_hash"] != ed.get("prior_attempt_event_hash")
                    or events[i - 1]["event_hash"] != ed.get("prior_head_hash")
                    or ed.get("prior_final_event_index") != i - 1
                    or ed.get("registration_event_index") != 35
                    or ed.get("registration_event_hash") != registration_event["event_hash"]
                    or ed.get("registration_id") != amendment["retry_registration_id"]):
                raise RegistryError("Cycle-3 retry amendment must bind current head, attempt, and registration")
            status = prior["terminal_data"].get("terminal_status")
            if status != ed.get("prior_terminal_status"):
                raise RegistryError("Cycle-3 retry amendment changed prior terminal status")
            if status == "COMPLETE":
                rejection = retry_rejections.get(prior["terminal"])
                if (rejection is None or rejection["event_index"] != ed.get("validation_rejection_event_index")
                        or rejection["event_hash"] != ed.get("validation_rejection_event_hash")
                        or rejection["event_index"] != i - 1
                        or prior["last"] != rejection["event_index"] - 1):
                    raise RegistryError("COMPLETE retry requires immediately preceding explicit artifact rejection")
            elif (prior["last"] != i - 1 or ed.get("validation_rejection_event_index") is not None):
                raise RegistryError("failed retry amendment cannot bind artifact rejection")
            continuation = {"event_index": i, "event_hash": event["event_hash"],
                            "prior_attempt_id": ed["prior_attempt_id"]}
            continue
        if typ == "exploratory_kernel_observation_v1":
            prior = retry_attempts.get(continuation["prior_attempt_id"]) if continuation else None
            rejection = (retry_rejections.get(prior["terminal"])
                         if prior is not None and prior.get("phase") == "terminal" else None)
            prior_registration_uuids = {
                item["event_data"]["record"].get("run_uuid")
                for item in events[:i] if item["event_type"] == "registration"
            }
            observed_uuids = {
                item["event_data"]["run_uuid"]
                for item in events[:i] if item["event_type"] == "exploratory_kernel_observation_v1"
            }
            if (continuation is None or continuation["event_index"] != i - 1
                    or continuation["event_hash"] != ed.get("continuation_event_hash")
                    or ed.get("continuation_event_index") != i - 1
                    or prior is None or prior["terminal_data"].get("terminal_status") != "COMPLETE"
                    or rejection is None or rejection["event_index"] != i - 2
                    or ed.get("kernel_id") != record["kernel_id"]
                    or ed.get("version_number", 0) <= current_version
                    or ed.get("run_uuid") in prior_registration_uuids | observed_uuids
                    or artifact is not None or lock is not None or train_state is not None):
                raise RegistryError("exploratory observation must supersede the exact pending continuation and advance its kernel cursor")
            # This records what already exists remotely. It consumes the one-shot continuation
            # without treating that unregistered run as a benchmark result or push receipt.
            current_version = ed["version_number"]
            continuation = None
            continue
        if typ == "cycle3_retry_rebased_authorization_v1":
            observation = events[i - 1] if i > 0 else None
            obs = observation["event_data"] if observation else {}
            if (rebase_authorization is not None or observation is None
                    or observation["event_type"] != "exploratory_kernel_observation_v1"
                    or ed.get("prior_final_event_index") != i - 1
                    or ed.get("prior_head_hash") != observation["event_hash"]
                    or ed.get("observation_event_index") != i - 1
                    or ed.get("observation_event_hash") != observation["event_hash"]
                    or ed.get("registration_event_index") != 35
                    or ed.get("registration_event_hash") != registration_event["event_hash"]
                    or ed.get("registration_id") != amendment["retry_registration_id"]
                    or ed.get("kernel_id") != record["kernel_id"]
                    or ed.get("expected_current_version") != current_version
                    or ed.get("expected_current_version") != obs.get("version_number")
                    or ed.get("observed_run_uuid") != obs.get("run_uuid")
                    or ed.get("observed_terminal_status") != obs.get("terminal_status")
                    or ed.get("observed_terminal_status") != "COMPLETE"
                    or ed.get("attempt_id") in retry_attempts
                    or ed.get("session_timeout_seconds") != record["session_timeout_seconds"]
                    or ed.get("reserved_gpu_seconds") != record["session_timeout_seconds"]):
                raise RegistryError("Cycle-3 rebase authorization must bind exact exploratory observation, registration, and fresh retry")
            for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                        "backend_sha256", "trainer_sha256", "engine_sha256"):
                if ed.get(key) != record.get(key):
                    raise RegistryError("Cycle-3 retry rebase authorization changed registered workload hashes")
            rebase_authorization = {"event_index": i, "event_hash": event["event_hash"],
                                    "attempt_id": ed["attempt_id"], "data": ed}
            continue
        if typ == "benchmark_artifact_validation_rejected":
            aid = next((aid for aid, state in retry_attempts.items()
                        if state.get("terminal") == ed.get("attempt_event_index")), None)
            state = retry_attempts.get(aid) if aid is not None else None
            if (state is None or state["phase"] != "terminal"
                    or state["terminal_data"].get("terminal_status") != "COMPLETE"
                    or state["last"] != i - 1
                    or events[state["terminal"]]["event_hash"] != ed.get("attempt_event_hash")
                    or ed.get("registration_event_index") != 35
                    or ed.get("registration_event_hash") != registration_event["event_hash"]
                    or state["terminal"] in retry_rejections):
                raise RegistryError("artifact rejection must bind a fresh COMPLETE attempt and its registration")
            retry_rejections[state["terminal"]] = {"event_index": i, "event_hash": event["event_hash"],
                                                     "event_data": ed}
            continue
        if typ == "workload_preflight" and ed.get("registration_event_index") == 29:
            if attestation is None:
                raise RegistryError("cycle-3 training requires retry benchmark attestation")
            selected = artifact and events[artifact]["event_data"]["selection_evidence"]["selected_variant_id"]
            train_record = events[29]["event_data"]["record"]
            variants = {row["variant_id"]: row for row in train_record["allowed_implementation_variants"]}
            variant = variants.get(selected)
            if (train_state is not None or variant is None
                    or ed.get("cycle_id") != "g008-recovery-cycle-3"
                    or ed.get("registration_event_hash") != events[29]["event_hash"]
                    or ed.get("expected_current_version") != current_version
                    or ed.get("kernel_id") != train_record["kernel_id"]
                    or ed.get("implementation_variant_id") != selected
                    or ed.get("session_timeout_seconds") != train_record["session_timeout_seconds"]
                    or ed.get("lock_head_hash") != events[lock]["event_hash"]):
                raise RegistryError("cycle-3 training preflight changed selected variant, version, or exact lock")
            for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                        "backend_sha256", "trainer_sha256", "engine_sha256"):
                expected = train_record[key] if key == "config_sha256" else variant[key]
                if ed.get(key) != expected:
                    raise RegistryError("cycle-3 training preflight differs from selected registered package")
            train_state = {"phase": "preflight", "last": i, "data": ed, "record": train_record,
                           "variant": variant, "registration_index": 29}
            continue
        if typ == "workload_preflight" and ed.get("registration_event_index") == 35:
            if artifact is not None or lock is not None:
                raise RegistryError("cycle-3 retry attempt must precede benchmark artifact and lock")
            aid = ed["attempt_id"]
            if aid in retry_attempts or ed.get("cycle_id") != "g008-recovery-cycle-3":
                raise RegistryError("cycle-3 retry attempts require fresh attempt IDs")
            if retry_attempts:
                direct_continuation = continuation is not None and continuation["event_index"] == i - 1
                rebased = (rebase_authorization is not None
                           and rebase_authorization["event_index"] == i - 1
                           and rebase_authorization["attempt_id"] == aid)
                if not (direct_continuation or rebased):
                    raise RegistryError("a retry after terminal completion requires an exact-head continuation or rebase authorization")
            if (ed.get("registration_event_hash") != registration_event["event_hash"]
                    or ed.get("registration_id") != amendment["retry_registration_id"]
                    or ed.get("expected_current_version") != current_version
                    or ed.get("kernel_id") != record["kernel_id"]
                    or ed.get("implementation_variant_id") != "paired"
                    or ed.get("session_timeout_seconds") != record["session_timeout_seconds"]
                    or ed.get("lock_head_hash") is not None):
                raise RegistryError("cycle-3 retry preflight changed authorized benchmark binding")
            for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                        "backend_sha256", "trainer_sha256", "engine_sha256"):
                if ed.get(key) != record[key]:
                    raise RegistryError("cycle-3 retry preflight changed registered package hash")
            retry_attempts[aid] = {"phase": "preflight", "last": i, "reg": 35, "data": ed,
                                   "preflight": i, "preflight_hash": event["event_hash"]}
            continuation = None
            rebase_authorization = None
            continue
        if typ in {"workload_push_dispatched", "workload_push_accepted", "workload_attempt"}:
            aid = ed["attempt_id"]
            if ed.get("registration_event_index") == 29:
                if train_state is None or aid != train_state["data"]["attempt_id"] or i != train_state["last"] + 1:
                    raise RegistryError("cycle-3 training lifecycle must follow its exact preflight")
                if typ == "workload_push_dispatched":
                    if (train_state["phase"] != "preflight"
                            or ed.get("preflight_event_index") != train_state["last"]
                            or ed.get("preflight_event_hash") != events[train_state["last"]]["event_hash"]):
                        raise RegistryError("cycle-3 training dispatch must bind exact preflight")
                    for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                                "backend_sha256", "trainer_sha256", "engine_sha256", "implementation_variant_id",
                                "lock_head_hash"):
                        if ed.get(key) != train_state["data"][key]:
                            raise RegistryError("cycle-3 training dispatch changed its preflight binding")
                    train_state.update(phase="dispatched", last=i, dispatch=i)
                elif typ == "workload_push_accepted":
                    if (train_state["phase"] != "dispatched" or ed.get("dispatch_event_index") != train_state["dispatch"]
                            or ed.get("implementation_variant_id") != train_state["data"]["implementation_variant_id"]
                            or ed.get("lock_head_hash") != events[lock]["event_hash"]
                            or ed.get("assigned_version", 0) <= current_version):
                        raise RegistryError("cycle-3 training acceptance changed its selected workload")
                    train_state.update(phase="accepted", last=i, accepted=i, version=ed["assigned_version"])
                else:
                    if (train_state["phase"] != "accepted" or ed.get("accepted_event_index") != train_state["accepted"]
                            or ed.get("assigned_version") != train_state["version"]
                            or ed.get("implementation_variant_id") != train_state["data"]["implementation_variant_id"]
                            or ed.get("lock_head_hash") != events[lock]["event_hash"]):
                        raise RegistryError("cycle-3 training terminal event changed accepted workload")
                    for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                                "backend_sha256", "trainer_sha256", "engine_sha256"):
                        if ed.get(key) != train_state["data"][key]:
                            raise RegistryError("cycle-3 training terminal event changed frozen hashes")
                    if (ed.get("actual_gpu_seconds", 0) > train_state["record"]["session_timeout_seconds"]
                            or ed.get("output_bytes", 0) > train_state["record"]["workload_output_bytes_cap"]):
                        raise RegistryError("cycle-3 training actual use exceeds registered limits")
                    train_state.update(phase="terminal", last=i, terminal=i, terminal_data=ed)
                    current_version = ed["assigned_version"]
                continue
            state = retry_attempts.get(aid)
            if state is None:
                raise RegistryError("cycle-3 retry lifecycle references an unregistered attempt")
            if state["phase"] == "terminal" or i != state["last"] + 1:
                raise RegistryError("cycle-3 retry attempt lifecycle must be contiguous")
            if ed.get("registration_event_index") != 35:
                raise RegistryError("cycle-3 retry lifecycle changed registration")
            if typ == "workload_push_dispatched":
                if (state["phase"] != "preflight"
                        or ed.get("registration_event_hash") != registration_event["event_hash"]
                        or ed.get("preflight_event_index") != state["preflight"]
                        or ed.get("preflight_event_hash") != state["preflight_hash"]):
                    raise RegistryError("cycle-3 retry dispatch must immediately follow preflight")
                for key in ("kernel_id", "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                            "backend_sha256", "trainer_sha256", "engine_sha256", "session_timeout_seconds",
                            "accelerator_count", "reserved_gpu_seconds", "implementation_variant_id", "lock_head_hash"):
                    if ed.get(key) != state["data"].get(key):
                        raise RegistryError("cycle-3 retry dispatch changed preflight workload binding")
                state.update(phase="dispatched", last=i, dispatch=i, dispatch_hash=event["event_hash"], dispatch_data=ed)
            elif typ == "workload_push_accepted":
                if (state["phase"] != "dispatched"
                        or ed.get("registration_event_hash") != registration_event["event_hash"]
                        or ed.get("preflight_event_index") != state["preflight"]
                        or ed.get("preflight_event_hash") != state["preflight_hash"]
                        or ed.get("dispatch_event_index") != state["dispatch"]
                        or ed.get("dispatch_event_hash") != state["dispatch_hash"]
                        or ed.get("implementation_variant_id") != state["data"]["implementation_variant_id"]
                        or ed.get("lock_head_hash") != state["data"]["lock_head_hash"]
                        or ed.get("confirming_metadata", {}).get("kernel_id") != record["kernel_id"]):
                    raise RegistryError("cycle-3 retry acceptance must bind dispatch")
                if ed.get("assigned_version", 0) <= amendment["current_version_number"]:
                    raise RegistryError("cycle-3 retry must assign a fresh Kaggle version")
                state.update(phase="accepted", last=i, accepted=i, accepted_hash=event["event_hash"],
                             version=ed["assigned_version"])
            else:
                if state["phase"] not in {"dispatched", "accepted"}:
                    raise RegistryError("cycle-3 retry terminal event lacks a dispatched push")
                if (ed.get("registration_event_hash") != registration_event["event_hash"]
                        or ed.get("preflight_event_index") != state["preflight"]
                        or ed.get("preflight_event_hash") != state["preflight_hash"]
                        or ed.get("dispatch_event_index") != state["dispatch"]
                        or ed.get("dispatch_event_hash") != state["dispatch_hash"]
                        or ed.get("implementation_variant_id") != "paired"
                        or ed.get("lock_head_hash") != state["data"]["lock_head_hash"]):
                    raise RegistryError("cycle-3 retry terminal event changed preflight, dispatch, or registration binding")
                for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                            "backend_sha256", "trainer_sha256", "engine_sha256"):
                    if ed.get(key) != state["data"][key]:
                        raise RegistryError("cycle-3 retry terminal result changed workload hash")
                if (ed.get("actual_gpu_seconds", 0) > record["session_timeout_seconds"]
                        or ed.get("output_bytes", 0) > record["workload_output_bytes_cap"]):
                    raise RegistryError("cycle-3 retry actual use exceeds registered limit")
                if state["phase"] == "dispatched":
                    no_assignment = ed["no_assignment_metadata"]
                    if (ed.get("outcome") != "pre_assignment_rejected"
                            or ed.get("accepted_event_index") is not None
                            or ed.get("accepted_event_hash") is not None
                            or ed.get("assigned_version") is not None
                            or ed.get("actual_gpu_seconds") != 0
                            or ed.get("rejection_evidence", {}).get("server_rejected_before_assignment") is not True
                            or no_assignment.get("no_version_assigned") is not True
                            or no_assignment.get("pre_push_current_version") != state["data"]["expected_current_version"]
                            or no_assignment.get("post_push_current_version") != state["data"]["expected_current_version"]):
                        raise RegistryError("cycle-3 retry rejection lacks exact no-assignment proof")
                elif (ed.get("outcome") != "terminal"
                      or ed.get("accepted_event_index") != state["accepted"]
                      or ed.get("accepted_event_hash") != state["accepted_hash"]
                      or ed.get("assigned_version") != state["version"]
                      or ed.get("terminal_metadata", {}).get("kernel_id") != record["kernel_id"]):
                    raise RegistryError("cycle-3 retry terminal result changed accepted paired workload")
                state.update(phase="terminal", last=i, terminal=i, terminal_data=ed)
                if ed.get("outcome") == "terminal":
                    current_version = ed["assigned_version"]
            continue
        if typ == "benchmark_artifact":
            terminal_states = [state for state in retry_attempts.values() if state.get("phase") == "terminal"]
            latest = max(terminal_states, key=lambda state: state["terminal"], default=None)
            if (artifact is not None or lock is not None or latest is None
                    or latest.get("phase") != "terminal"
                    or latest["terminal_data"].get("terminal_status") != "COMPLETE"
                    or latest["terminal"] in retry_rejections
                    or latest["terminal"] != ed.get("attempt_event_index")
                    or event["event_data"].get("registration_event_index") != 35
                    or event["event_data"].get("registration_event_hash") != registration_event["event_hash"]
                    or events[latest["terminal"]]["event_hash"] != ed.get("attempt_event_hash")):
                raise RegistryError("cycle-3 retry artifact requires its COMPLETE benchmark attempt")
            artifact = i
            continue
        if typ == "registration_closed":
            if artifact is None or i != artifact + 1 or ed != {
                    "pre_lock_head_hash": events[artifact]["event_hash"], "pre_lock_final_event_index": artifact}:
                raise RegistryError("cycle-3 retry lock must pin exact retry artifact")
            lock = i
            continue
        if typ == "benchmark_attestation":
            if lock is None or i != lock + 1 or artifact is None or ed != {
                    **events[artifact]["event_data"], "lock_head_hash": events[lock]["event_hash"]}:
                raise RegistryError("cycle-3 retry attestation must bind exact artifact and lock")
            attestation = i
            continue
        raise RegistryError("event type is not allowed in cycle-3 retry suffix")

    state = max(retry_attempts.values(), key=lambda item: item["last"], default=None)
    if state and state["phase"] != "terminal" and state["last"] != len(events) - 1:
        raise RegistryError("pending cycle-3 retry attempt blocks later events")
    if continuation is not None and continuation["event_index"] != len(events) - 1:
        raise RegistryError("pending Cycle-3 retry policy amendment blocks later events")
    if rebase_authorization is not None and rebase_authorization["event_index"] != len(events) - 1:
        raise RegistryError("pending Cycle-3 retry rebase authorization blocks later events")
    if state and state["phase"] == "terminal" and state["terminal_data"]["terminal_status"] != "COMPLETE":
        if artifact is not None or lock is not None or attestation is not None or train_state is not None:
            raise RegistryError("failed cycle-3 retry cannot produce artifact, lock, attestation, or training")
        if state["last"] != len(events) - 1 and continuation is None:
            raise RegistryError("failed cycle-3 retry requires an exact-head policy amendment before continuing")
    if state and state["phase"] == "terminal" and state["terminal_data"]["terminal_status"] == "COMPLETE":
        if state["terminal"] not in retry_rejections and state["terminal"] != (events[artifact]["event_data"].get("attempt_event_index") if artifact is not None else None):
            if state["last"] != len(events) - 1:
                raise RegistryError("COMPLETE retry requires artifact validation before any further event")
    retry_actual = sum(item["terminal_data"]["actual_gpu_seconds"] for item in retry_attempts.values()
                       if item["phase"] == "terminal")
    retry_reserved = sum(record["session_timeout_seconds"] for item in retry_attempts.values()
                         if item["phase"] != "terminal")
    rebase_reserved = (rebase_authorization["data"]["reserved_gpu_seconds"]
                       if rebase_authorization is not None else 0)
    exploratory_observations = [event["event_data"] for event in events
                                if event["event_type"] == "exploratory_kernel_observation_v1"]
    exploratory_actual = sum(item["actual_gpu_seconds"] for item in exploratory_observations)
    first_attempt = parent["event_data"]
    if (first_attempt["actual_gpu_seconds"] + exploratory_actual + retry_actual + retry_reserved + rebase_reserved
            > original["compute_caps_gpu_seconds"]["benchmark"]):
        raise RegistryError("cycle-3 first attempt plus retries exceeds cumulative benchmark GPU cap")
    train_actual = (train_state["terminal_data"]["actual_gpu_seconds"]
                    if train_state and train_state["phase"] == "terminal" else 0)
    train_reserved = (train_state["record"]["session_timeout_seconds"]
                      if train_state and train_state["phase"] != "terminal" else 0)
    if train_actual + train_reserved > original["compute_caps_gpu_seconds"]["train"]:
        raise RegistryError("cycle-3 training use/reservation exceeds registered training cap")
    retry_output = (sum(item["terminal_data"]["output_bytes"] for item in retry_attempts.values()
                        if item["phase"] == "terminal")
                    + sum(record["workload_output_bytes_cap"] for item in retry_attempts.values()
                          if item["phase"] != "terminal"))
    rebase_output_reserved = (record["workload_output_bytes_cap"]
                              if rebase_authorization is not None else 0)
    exploratory_output = sum(item["output_bytes"] for item in exploratory_observations)
    benchmark_output = first_attempt["output_bytes"] + exploratory_output + retry_output + rebase_output_reserved
    train_output = (train_state["terminal_data"]["output_bytes"]
                    if train_state and train_state["phase"] == "terminal"
                    else train_state["record"]["workload_output_bytes_cap"] if train_state else 0)
    output_caps = original["output_category_reservations_bytes"]
    if (benchmark_output > output_caps["benchmark"] or train_output > output_caps["training"]):
        raise RegistryError("cycle-3 retry and training output use/reservations exceed category cap")
    project_output = (original["project_output_bytes_before"] + benchmark_output + train_output
                      + output_caps["validation"] + output_caps["final"])
    if project_output > original["output_caps_bytes"]["aggregate"]:
        raise RegistryError("cycle-3 retry and training output use/reservations exceed aggregate cap")
    if train_state and train_state["phase"] != "terminal" and train_state["last"] != len(events) - 1:
        raise RegistryError("pending cycle-3 training attempt blocks later events")
    if train_state and train_state["phase"] == "terminal" and train_state["last"] != len(events) - 1:
        raise RegistryError("terminal cycle-3 training attempt blocks additional workloads")
    project_gpu = (original["project_gpu_seconds_before"] + first_attempt["actual_gpu_seconds"]
                   + exploratory_actual + retry_actual + retry_reserved + train_actual + train_reserved)
    project_gpu += rebase_reserved
    if (project_gpu > original["compute_caps_gpu_seconds"]["project_experiment"]
            or project_gpu + original["protected_final_reserve_gpu_seconds"]
            > original["compute_caps_gpu_seconds"]["project_total"]):
        raise RegistryError("cycle-3 retry and training use/reservations exceed project budget")


def _validate_g008_competitive_campaign(events: list[dict[str, Any]]) -> None:
    """Replay the immutable paired-seed campaign suffix, including benchmark lock ordering."""
    campaign_indices = [i for i, e in enumerate(events)
                        if e["event_type"] == "g008_competitive_campaign_amendment_v1"]
    if not campaign_indices:
        return
    if campaign_indices != [len(events[:campaign_indices[0]])]:  # retain an explicit single-amendment invariant
        raise RegistryError("competitive campaign amendment is duplicated")
    ci = campaign_indices[0]
    _validate_recovery_cycle3_retry_suffix(events[:ci])
    amendment_event = events[ci]
    amendment = amendment_event["event_data"]
    _validate_data("g008_competitive_campaign_amendment_v1", amendment)
    if (ci < 1 or amendment["prior_final_event_index"] != ci - 1
            or amendment["prior_head_hash"] != events[ci - 1]["event_hash"]
            or amendment["rejection_event_index"] != ci - 1
            or events[ci - 1]["event_type"] != "benchmark_artifact_validation_rejected"
            or amendment["rejection_event_hash"] != events[ci - 1]["event_hash"]):
        raise RegistryError("competitive campaign must extend the exact rejected-artifact head")
    retry = events[35]["event_data"]["record"]
    rejected_attempt = events[amendment["rejection_event_index"]]["event_data"]["attempt_event_index"]
    if (amendment["kernel_id"] != retry["kernel_id"]
            or amendment["current_version_number"] != events[rejected_attempt]["event_data"]["assigned_version"]):
        raise RegistryError("competitive campaign changed the exact retry kernel/version")
    cycle3 = events[27]["event_data"]
    prior_gpu = sum(e["event_data"]["actual_gpu_seconds"] for e in events[33:ci]
                    if e["event_type"] == "workload_attempt" and e["event_data"].get("outcome") == "terminal")
    projected = cycle3["project_gpu_seconds_before"] + prior_gpu + amendment["total_gpu_seconds_cap"]
    if (projected > cycle3["compute_caps_gpu_seconds"]["project_experiment"]
            or projected + amendment["protected_final_reserve_gpu_seconds"]
            > cycle3["compute_caps_gpu_seconds"]["project_total"]):
        raise RegistryError("campaign budget plus prior use exceeds project cap/protected reserve")

    registrations: dict[int, dict[str, Any]] = {}
    amendment_rows = amendment["records"]
    old_ids = {e["event_data"]["registration_id"] for e in events[:ci]
               if e["event_type"] == "registration"}
    old_uuids = {e["event_data"]["record"].get("run_uuid") for e in events[:ci]
                 if e["event_type"] == "registration"}
    for offset, row in enumerate(amendment_rows, 1):
        record = dict(row["record"])
        if row["registration_id"] in old_ids or record["run_uuid"] in old_uuids:
            raise RegistryError("competitive campaign reuses a prior registration ID or UUID")
        record.update({"competitive_campaign_event_index": ci,
                       "competitive_campaign_event_hash": amendment_event["event_hash"]})
        idx = ci + offset
        if idx >= len(events) or events[idx]["event_type"] != "registration":
            if idx < len(events):
                raise RegistryError("all seven campaign registrations must be consecutive after amendment")
            break
        ev, data = events[idx], events[idx]["event_data"]
        if (data.get("registration_id") != row["registration_id"]
                or data.get("registration_kind") != row["registration_kind"]
                or data.get("cycle_id") != "g008-competitive-campaign-v1"
                or data.get("record") != record
                or hashlib.sha256(canonical_bytes(row["record"])).hexdigest() != row["record_sha256"]):
            raise RegistryError("campaign registration differs from immutable amendment record/hash")
        registrations[idx] = {"event": ev, "record": data["record"], "kind": data["registration_kind"]}

    if len(registrations) < 9:
        if len(events) != ci + 1 + len(registrations):
            raise RegistryError("campaign registration prefix is incomplete and cannot contain later events")
        return

    attempts: dict[str, dict[str, Any]] = {}
    latest_version = amendment["current_version_number"]
    artifact = lock = attestation = None
    candidate_freeze = validation_result = None
    candidate_checkpoint_sha256 = None
    validation_passed = False
    for i in range(ci + 10, len(events)):
        event = events[i]
        typ, data = event["event_type"], event["event_data"]
        if typ == "competitive_candidate_freeze":
            if (candidate_freeze is not None or attestation is None
                    or len([s for s in attempts.values() if s["phase"] == "terminal"
                            and s["record"]["registration_kind"] == "train"
                            and s["terminal"].get("terminal_status") == "COMPLETE"]) != 6):
                raise RegistryError("candidate freeze requires attested benchmark and six terminal training jobs")
            candidate_idx = data.get("candidate_registration_event_index")
            train_attempt_idx = data.get("train_attempt_event_index")
            train_state = next((s for s in attempts.values()
                                if s.get("terminal_index") == train_attempt_idx), None)
            if (candidate_idx not in registrations or registrations[candidate_idx]["kind"] != "train"
                    or data.get("candidate_registration_event_hash") != events[candidate_idx]["event_hash"]
                    or registrations[candidate_idx]["record"].get("strategy_role") != "mixed"
                    or registrations[candidate_idx]["record"].get("pair_id") != "seed-pair-3"
                    or registrations[candidate_idx]["record"].get("update_budget") != 236
                    or train_state is None or train_state["registration_index"] != candidate_idx
                    or train_state["record"]["registration_kind"] != "train"
                    or train_state["terminal"].get("terminal_status") != "COMPLETE"
                    or data.get("train_attempt_event_hash") != events[train_attempt_idx]["event_hash"]
                    or data.get("checkpoint_artifact_sha256") != data.get("checkpoint_sha256")
                    or not all(_is_hash(data.get(key)) for key in (
                        "checkpoint_sha256", "checkpoint_artifact_sha256", "output_manifest_sha256"))):
                raise RegistryError("candidate freeze must bind the exact COMPLETE third mixed-arm attempt and its artifacts")
            candidate_freeze = i
            candidate_checkpoint_sha256 = data["checkpoint_sha256"]
            continue
        if typ == "competitive_validation_result":
            validation_idx = data.get("validation_registration_event_index")
            validation_attempt_idx = data.get("validation_attempt_event_index")
            validation_state = next((s for s in attempts.values()
                                     if s.get("terminal_index") == validation_attempt_idx), None)
            if (candidate_freeze is None or validation_result is not None
                    or data.get("candidate_freeze_event_index") != candidate_freeze
                    or data.get("candidate_freeze_event_hash") != events[candidate_freeze]["event_hash"]
                    or validation_idx not in registrations or registrations[validation_idx]["kind"] != "validation"
                    or data.get("validation_registration_event_hash") != events[validation_idx]["event_hash"]
                    or validation_state is None or validation_state["registration_index"] != validation_idx
                    or validation_state["record"]["registration_kind"] != "validation"
                    or validation_state["terminal"].get("terminal_status") != "COMPLETE"
                    or data.get("validation_attempt_event_hash") != events[validation_attempt_idx]["event_hash"]
                    or validation_state["last"] != i - 1
                    or data.get("candidate_checkpoint_sha256") != candidate_checkpoint_sha256
                    or not all(_is_hash(data.get(key)) for key in (
                        "candidate_checkpoint_sha256", "report_sha256", "output_manifest_sha256"))
                    or data.get("game_count") != 512 or data.get("opponent_id") == "open-space"):
                raise RegistryError("validation must bind frozen checkpoint, exact COMPLETE attempt, report, manifest, and non-final roster")
            wins, draws, losses = data.get("wins"), data.get("draws"), data.get("losses")
            if (not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (wins, draws, losses))
                    or wins + draws + losses != 512 or type(data.get("passed")) is not bool):
                raise RegistryError("campaign validation must reconcile exactly 512 games")
            validation_passed = data["passed"]
            validation_result = i
            continue
        if typ == "competitive_final_result":
            final_idx = data.get("final_registration_event_index")
            attempt_idx = data.get("final_attempt_event_index")
            final_state = next((s for s in attempts.values() if s.get("terminal_index") == attempt_idx), None)
            if (candidate_freeze is None or validation_result is None or i != len(events) - 1
                    or final_idx not in registrations or registrations[final_idx]["kind"] != "final"
                    or data.get("candidate_freeze_event_index") != candidate_freeze
                    or data.get("candidate_freeze_event_hash") != events[candidate_freeze]["event_hash"]
                    or data.get("final_registration_event_hash") != events[final_idx]["event_hash"]
                    or final_state is None or final_state["record"]["registration_kind"] != "final"
                    or final_state["terminal"].get("terminal_status") != "COMPLETE"
                    or final_state["registration_index"] != final_idx
                    or data.get("final_attempt_event_hash") != events[attempt_idx]["event_hash"]
                    or final_state["last"] != i - 1
                    or data.get("candidate_checkpoint_sha256") != candidate_checkpoint_sha256
                    or not all(_is_hash(data.get(key)) for key in (
                        "candidate_checkpoint_sha256", "report_sha256", "output_manifest_sha256"))
                    or data.get("roster_sha256") != registrations[final_idx]["record"].get("roster_sha256")
                    or data.get("opponent_code_sha256") != registrations[final_idx]["record"].get("opponent_code_sha256")
                    or data.get("opponent_id") != "open-space"):
                raise RegistryError("final result must reconcile the frozen checkpoint and exact COMPLETE final attempt/report/manifest")
            final_result = i
            continue
        if typ == "workload_preflight":
            ri = data["registration_event_index"]
            reg = registrations.get(ri)
            if (reg is None or reg["event"]["event_hash"] != data["registration_event_hash"]
                    or reg["event"]["event_data"]["registration_id"] != data["registration_id"]
                    or data.get("cycle_id") != "g008-competitive-campaign-v1"
                    or data["kernel_id"] != reg["record"]["kernel_id"]
                    or data["expected_current_version"] != latest_version
                    or data["session_timeout_seconds"] != reg["record"]["session_timeout_seconds"]
                    or data["lock_head_hash"] != (events[lock]["event_hash"] if lock is not None else None)):
                raise RegistryError("campaign preflight changed exact registered record/version/lock")
            if reg["kind"] == "train" and data["implementation_variant_id"] != reg["record"]["strategy_role"]:
                raise RegistryError("campaign preflight changed paired training strategy arm")
            if reg["kind"] in {"validation", "final"} and data["implementation_variant_id"] != "frozen-candidate":
                raise RegistryError("campaign evaluation must run the exact frozen candidate")
            if reg["kind"] == "validation" and candidate_freeze is None:
                raise RegistryError("validation cannot run before candidate freeze")
            if reg["kind"] == "benchmark":
                if artifact is not None or lock is not None or data["implementation_variant_id"] != "paired":
                    raise RegistryError("campaign benchmark attempts must precede the lock and use paired mode")
            elif reg["kind"] == "train" and attestation is None:
                raise RegistryError("campaign training requires benchmark attestation")
            elif reg["kind"] == "final" and (validation_result is None or not validation_passed):
                raise RegistryError("final-only evaluation requires a candidate that passes frozen non-inferiority validation")
            elif reg["kind"] == "final" and data.get("opponent_id") == "open-space":
                raise RegistryError("open-space identity must remain absent from pre-final events")
            aid = data["attempt_id"]
            if aid in attempts or any(s["registration_index"] == ri for s in attempts.values()
                                      if s["record"]["registration_kind"] == "train"):
                raise RegistryError("campaign attempt ID or training workload is duplicated")
            for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                        "backend_sha256", "trainer_sha256", "engine_sha256"):
                if data[key] != reg["record"][key]:
                    raise RegistryError("campaign preflight changed frozen package/config/seed hashes")
            attempts[aid] = {"phase": "preflight", "last": i, "registration_index": ri,
                             "record": {**reg["record"], "registration_kind": reg["kind"]},
                             "data": data, "preflight_hash": event["event_hash"]}
            continue
        if typ in {"workload_push_dispatched", "workload_push_accepted", "workload_attempt"}:
            aid = data["attempt_id"]
            state = attempts.get(aid)
            if state is None or state["phase"] == "terminal" or i != state["last"] + 1:
                raise RegistryError("campaign push lifecycle must immediately follow its prior phase")
            if data.get("cycle_id") != "g008-competitive-campaign-v1":
                raise RegistryError("campaign push lifecycle has wrong campaign identity")
            pf = state["data"]
            if (data["registration_event_index"] != state["registration_index"]
                    or data["registration_event_hash"] != events[state["registration_index"]]["event_hash"]):
                raise RegistryError("campaign push lifecycle changed registration binding")
            if typ == "workload_push_dispatched":
                if (state["phase"] != "preflight" or data["preflight_event_index"] != state["last"]
                        or data["preflight_event_hash"] != state["preflight_hash"]):
                    raise RegistryError("campaign dispatch must immediately bind exact preflight")
                for key in ("kernel_id", "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                            "backend_sha256", "trainer_sha256", "engine_sha256", "session_timeout_seconds",
                            "accelerator_count", "reserved_gpu_seconds", "implementation_variant_id", "lock_head_hash"):
                    if data[key] != pf[key]:
                        raise RegistryError("campaign dispatch changed preflight fields")
                state.update(phase="dispatched", last=i, dispatch_hash=event["event_hash"])
            elif typ == "workload_push_accepted":
                if (state["phase"] != "dispatched" or data["preflight_event_index"] != state["last"] - 1
                        or data["dispatch_event_index"] != state["last"]
                        or data["dispatch_event_hash"] != state["dispatch_hash"]
                        or data["assigned_version"] <= latest_version
                        or data["implementation_variant_id"] != pf["implementation_variant_id"]
                        or data["lock_head_hash"] != pf["lock_head_hash"]):
                    raise RegistryError("campaign acceptance changed dispatch, version, or lock")
                state.update(phase="accepted", last=i, version=data["assigned_version"])
            else:
                rejected_before_assignment = data["outcome"] == "pre_assignment_rejected"
                if rejected_before_assignment:
                    if (state["phase"] != "dispatched" or data["dispatch_event_index"] != state["last"]
                            or data["dispatch_event_hash"] != state["dispatch_hash"]
                            or data["assigned_version"] is not None or data["accepted_event_index"] is not None
                            or data["actual_gpu_seconds"] != 0
                            or data["no_assignment_metadata"].get("no_version_assigned") is not True
                            or data["no_assignment_metadata"].get("pre_push_current_version") != pf["expected_current_version"]
                            or data["no_assignment_metadata"].get("post_push_current_version") != pf["expected_current_version"]):
                        raise RegistryError("campaign pre-assignment rejection lacks exact no-assignment proof")
                elif (state["phase"] != "accepted" or data["accepted_event_index"] != state["last"]
                      or data["assigned_version"] != state["version"]):
                    raise RegistryError("campaign terminal attempt changed acceptance or version")
                if data["lock_head_hash"] != pf["lock_head_hash"]:
                    raise RegistryError("campaign terminal attempt changed exact lock")
                for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                            "backend_sha256", "trainer_sha256", "engine_sha256"):
                    if data[key] != pf[key]:
                        raise RegistryError("campaign terminal attempt changed frozen hashes")
                cap_key = "benchmark_gpu_seconds_cap" if state["record"]["registration_kind"] == "benchmark" else "training_gpu_seconds_cap"
                if data["actual_gpu_seconds"] > state["record"]["workload_gpu_seconds_cap"] or data["output_bytes"] > state["record"]["workload_output_bytes_cap"]:
                    raise RegistryError("campaign attempt exceeds per-workload resource caps")
                state.update(phase="terminal", last=i, terminal_index=i, terminal=data)
                if not rejected_before_assignment:
                    latest_version = data["assigned_version"]
            continue
        if typ == "benchmark_artifact_validation_rejected":
            state = next((s for s in attempts.values() if s.get("terminal_index") == data["attempt_event_index"]), None)
            if (state is None or state["record"]["registration_kind"] != "benchmark"
                    or state["phase"] != "terminal" or state["terminal"].get("terminal_status") != "COMPLETE"
                    or state["last"] != i - 1 or data["attempt_event_hash"] != events[i - 1]["event_hash"]
                    or data["registration_event_index"] != state["registration_index"]
                    or data["registration_event_hash"] != events[state["registration_index"]]["event_hash"]):
                raise RegistryError("campaign artifact rejection must immediately bind a COMPLETE benchmark")
            state["rejected"] = True
            continue
        if typ == "benchmark_artifact":
            state = next((s for s in attempts.values() if s.get("terminal_index") == data.get("attempt_event_index")), None)
            if (artifact is not None or state is None or state["record"]["registration_kind"] != "benchmark"
                    or state["terminal"].get("terminal_status") != "COMPLETE"
                    or state.get("rejected", False)):
                raise RegistryError("campaign artifact must bind a non-rejected COMPLETE benchmark")
            if (i != state["last"] + 1 or data["registration_event_index"] != state["registration_index"]
                    or data["registration_event_hash"] != events[state["registration_index"]]["event_hash"]
                    or data["attempt_event_hash"] != events[state["terminal_index"]]["event_hash"]):
                raise RegistryError("campaign artifact must immediately bind latest successful benchmark")
            artifact = i
            continue
        if typ == "registration_closed":
            if artifact is None or i != artifact + 1 or data != {
                    "pre_lock_head_hash": events[artifact]["event_hash"], "pre_lock_final_event_index": artifact}:
                raise RegistryError("campaign registration lock must immediately follow exact benchmark artifact")
            lock = i
            continue
        if typ == "benchmark_attestation":
            if lock is None or i != lock + 1 or data != {
                    **events[artifact]["event_data"], "lock_head_hash": events[lock]["event_hash"]}:
                raise RegistryError("campaign benchmark attestation must immediately follow exact lock")
            attestation = i
            continue
        raise RegistryError("event type is not permitted in competitive campaign suffix")

    for aid, state in attempts.items():
        if state["phase"] != "terminal" and state["last"] != len(events) - 1:
            raise RegistryError("pending campaign attempt blocks later events")
        if state["phase"] == "terminal" and state["terminal"].get("terminal_status") == "COMPLETE":
            is_benchmark = state["record"]["registration_kind"] == "benchmark"
            if is_benchmark and not state.get("rejected") and state.get("terminal_index") != (events[artifact]["event_data"].get("attempt_event_index") if artifact is not None else None):
                if state["last"] != len(events) - 1:
                    raise RegistryError("unresolved COMPLETE benchmark requires artifact or explicit rejection")
    actual_benchmark = sum(s["terminal"]["actual_gpu_seconds"] for s in attempts.values()
                           if s["phase"] == "terminal" and s["record"]["registration_kind"] == "benchmark")
    reserved_benchmark = sum(s["record"]["session_timeout_seconds"] for s in attempts.values()
                             if s["phase"] != "terminal" and s["record"]["registration_kind"] == "benchmark")
    actual_train = sum(s["terminal"]["actual_gpu_seconds"] for s in attempts.values()
                       if s["phase"] == "terminal" and s["record"]["registration_kind"] == "train")
    reserved_train = sum(s["record"]["session_timeout_seconds"] for s in attempts.values()
                         if s["phase"] != "terminal" and s["record"]["registration_kind"] == "train")
    actual_eval = sum(s["terminal"]["actual_gpu_seconds"] for s in attempts.values()
                      if s["phase"] == "terminal" and s["record"]["registration_kind"] in {"validation", "final"})
    reserved_eval = sum(s["record"]["session_timeout_seconds"] for s in attempts.values()
                        if s["phase"] != "terminal" and s["record"]["registration_kind"] in {"validation", "final"})
    if (actual_benchmark + reserved_benchmark > amendment["benchmark_gpu_seconds_cap"]
            or actual_train + reserved_train > amendment["training_gpu_seconds_cap"]
            or actual_eval + reserved_eval > amendment["evaluation_gpu_seconds_cap"]
            or actual_benchmark + reserved_benchmark + actual_train + reserved_train + actual_eval + reserved_eval > amendment["total_gpu_seconds_cap"]):
        raise RegistryError("competitive campaign actual use/reservations exceed frozen cumulative budget")
    if any(s["phase"] == "terminal" and s["record"]["registration_kind"] == "train"
           for s in attempts.values()) and attestation is None:
        raise RegistryError("campaign training cannot start before benchmark attestation")


def _competitive_v2_comparison_payload(events: list[dict[str, Any]], *, created_at_utc: str) -> dict[str, Any]:
    registrations = {event["event_data"]["registration_id"]: event
                    for event in events if event["event_type"] == "registration"
                    and event["event_data"].get("cycle_id") == "g008-competitive-campaign-v2"}
    validation_results = {event["event_data"]["validation_registration_event_index"]: event
                          for event in events if event["event_type"] == "competitive_validation_result_v2"}
    ordered_registration_indices = [index for index, event in enumerate(events)
                                    if event["event_type"] == "registration"
                                    and event["event_data"].get("cycle_id") == "g008-competitive-campaign-v2"
                                    and event["event_data"]["registration_kind"] == "validation"]
    if len(ordered_registration_indices) != 6 or any(index not in validation_results
                                                     for index in ordered_registration_indices):
        raise RegistryError("strategy comparison requires all six registered validation results")
    scores: dict[tuple[str, str], float] = {}
    result_indices = []
    result_hashes = []
    result_by_arm: dict[tuple[str, str], dict[str, Any]] = {}
    for registration_index in ordered_registration_indices:
        event = validation_results[registration_index]
        data = event["event_data"]
        registration = events[registration_index]["event_data"]
        record = registration["record"]
        score = (data["wins"] + 0.5 * data["draws"]) / 512
        key = (record["pair_id"], record["strategy_role"])
        if key in scores:
            raise RegistryError("duplicate validation result for paired campaign arm")
        scores[key] = score
        result_by_arm[key] = event
        result_indices.append(event["event_index"])
        result_hashes.append(event["event_hash"])
    deltas = []
    numeric_deltas = []
    for pair_no in range(1, 4):
        pair_id = f"seed-pair-{pair_no}"
        baseline, mixed = scores.get((pair_id, "baseline")), scores.get((pair_id, "mixed"))
        if baseline is None or mixed is None:
            raise RegistryError("strategy comparison requires baseline and mixed result in every seed block")
        delta = mixed - baseline
        numeric_deltas.append(delta)
        deltas.append({"pair_id": pair_id, "baseline_score": f"{baseline:.6f}",
                       "mixed_score": f"{mixed:.6f}", "score_delta": f"{delta:.6f}"})
    mean_delta = statistics.mean(numeric_deltas)
    stddev = statistics.stdev(numeric_deltas) if len(numeric_deltas) > 1 else 0.0
    margin = 4.302652729911275 * stddev / math.sqrt(3)
    lower, upper = mean_delta - margin, mean_delta + margin
    lower_text, upper_text = f"{lower:.6f}", f"{upper:.6f}"
    max_pair_regression = max(0.0, -min(numeric_deltas))
    mixed_passed = float(lower_text) > 0 and min(numeric_deltas) >= -0.05
    selected_strategy = "mixed" if mixed_passed else "baseline"
    selected_arm_rows = [(scores[(f"seed-pair-{pair_no}", selected_strategy)], pair_no)
                         for pair_no in range(1, 4)]
    selected_arm_rows.sort(key=lambda item: (item[0], item[1]))
    selected_pair_id = f"seed-pair-{selected_arm_rows[1][1]}"
    selected_validation_result = result_by_arm[(selected_pair_id, selected_strategy)]
    selected_validation = selected_validation_result["event_data"]
    selected_training_index = selected_validation["training_registration_event_index"]
    selected_training_attempt_index = selected_validation["training_attempt_event_index"]
    selected_artifact_event = next((event for event in events
                                    if event["event_type"] == "competitive_training_artifact_v2"
                                    and event["event_data"]["training_registration_event_index"]
                                    == selected_training_index), None)
    selected_validation_registration_index = selected_validation["validation_registration_event_index"]
    selected_validation_attempt_index = selected_validation["validation_attempt_event_index"]
    selected_training = events[selected_training_index]["event_data"]
    selected_validation_registration = events[selected_validation_registration_index]["event_data"]
    if selected_artifact_event is None:
        raise RegistryError("selected strategy comparison lacks the selected checkpoint artifact attestation")
    selected_candidate = {
        "strategy_role": selected_strategy, "pair_id": selected_pair_id,
        "training_registration_event_index": selected_training_index,
        "training_registration_event_hash": events[selected_training_index]["event_hash"],
        "training_attempt_event_index": selected_training_attempt_index,
        "training_attempt_event_hash": events[selected_training_attempt_index]["event_hash"],
        "training_artifact_event_index": selected_artifact_event["event_index"],
        "training_artifact_event_hash": selected_artifact_event["event_hash"],
        "validation_registration_event_index": selected_validation_registration_index,
        "validation_registration_event_hash": events[selected_validation_registration_index]["event_hash"],
        "validation_attempt_event_index": selected_validation_attempt_index,
        "validation_attempt_event_hash": events[selected_validation_attempt_index]["event_hash"],
        "validation_result_event_index": selected_validation_result["event_index"],
        "validation_result_event_hash": selected_validation_result["event_hash"],
        "checkpoint_sha256": selected_artifact_event["event_data"]["checkpoint_sha256"],
        "output_manifest_sha256": selected_artifact_event["event_data"]["output_manifest_sha256"],
        "training_run_uuid": selected_training["record"]["run_uuid"],
        "validation_run_uuid": selected_validation_registration["record"]["run_uuid"],
    }
    return {"schema_id": "g008-strategy-comparison", "schema_version": 2,
            "campaign_id": "g008-competitive-20261021-v2", "created_at_utc": created_at_utc,
            "validation_result_event_indices": result_indices,
            "validation_result_event_hashes": result_hashes,
            "paired_score_deltas": deltas,
            "mean_paired_score_delta": f"{mean_delta:.6f}",
            "paired_score_ci95_lower": lower_text,
            "paired_score_ci95_upper": upper_text,
            "max_pair_regression": f"{max_pair_regression:.6f}",
            "selected_strategy": selected_strategy, "selected_candidate": selected_candidate,
            "mixed_strategy_passed": mixed_passed}


def _validate_g008_v2_final_result_roster(final_record: dict[str, Any],
                                          final_result: dict[str, Any]) -> None:
    if (final_result.get("roster_sha256") != final_record.get("roster_sha256")
            or final_result.get("opponent_code_sha256") != final_record.get("opponent_code_sha256")):
        raise RegistryError("campaign v2 final result roster hashes differ from exact registered final roster")


def _validate_g008_competitive_campaign_v2(events: list[dict[str, Any]]) -> None:
    """Replay the v2 paired training, comparative validation, selection, and final lifecycle."""
    indices = [i for i, event in enumerate(events)
               if event["event_type"] == "g008_competitive_campaign_amendment_v2"]
    if indices != [55] or len(events) < 56:
        raise RegistryError("campaign v2 amendment must be unique and append to event 54")
    _validate_recovery_cycle3_retry_suffix(events[:55])
    amendment_event = events[55]
    amendment = amendment_event["event_data"]
    _validate_data("g008_competitive_campaign_amendment_v2", amendment)
    exact_anchors = {
        "prior_head_hash": events[54]["event_hash"],
        "benchmark_artifact_event_hash": events[48]["event_hash"],
        "benchmark_lock_event_hash": events[49]["event_hash"],
        "benchmark_attestation_event_hash": events[50]["event_hash"],
        "selected_training_registration_event_hash": events[29]["event_hash"],
        "selected_training_attempt_event_hash": events[54]["event_hash"],
    }
    if (amendment_event["prev_event_hash"] != events[54]["event_hash"]
            or any(amendment.get(key) != value for key, value in exact_anchors.items())):
        raise RegistryError("campaign v2 provenance anchors do not match exact Cycle-3 events")
    if (events[48]["event_type"] != "benchmark_artifact"
            or events[49]["event_type"] != "registration_closed"
            or events[50]["event_type"] != "benchmark_attestation"
            or events[29]["event_type"] != "registration"
            or events[54]["event_type"] != "workload_attempt"):
        raise RegistryError("campaign v2 provenance references the wrong event types")
    selected_reg, selected_attempt = events[29]["event_data"], events[54]["event_data"]
    if (selected_reg["registration_kind"] != "train"
            or amendment["selected_training_run_uuid"] != selected_reg["record"].get("run_uuid")
            or selected_attempt.get("registration_event_index") != 29
            or selected_attempt.get("registration_event_hash") != events[29]["event_hash"]
            or selected_attempt.get("outcome") != "terminal"
            or selected_attempt.get("terminal_status") != "COMPLETE"
            or selected_attempt.get("assigned_version") != 17
            or selected_attempt.get("implementation_variant_id") != "incumbent"
            or events[50]["event_data"].get("selection_evidence", {}).get("selected_variant_id") != "incumbent"):
        raise RegistryError("campaign v2 may use event54 only as historical selected-incumbent provenance")

    cycle3 = events[27]["event_data"]
    exploratory = events[42]["event_data"]
    if events[42]["event_type"] != "exploratory_kernel_observation_v1":
        raise RegistryError("campaign v2 requires the v15 exploratory usage observation")
    prior_attempt_gpu = sum(event["event_data"].get("actual_gpu_seconds", 0)
                            for event in events[33:55]
                            if event["event_type"] == "workload_attempt"
                            and event["event_data"].get("outcome") == "terminal")
    prior_gpu = cycle3["project_gpu_seconds_before"] + prior_attempt_gpu + exploratory["actual_gpu_seconds"]
    if prior_gpu != 9_242:
        raise RegistryError("campaign v2 prior GPU use must count Cycle-3 attempts and v15 observation once")
    if (prior_gpu + amendment["total_gpu_seconds_cap"]
            > cycle3["compute_caps_gpu_seconds"]["project_experiment"]
            or prior_gpu + amendment["total_gpu_seconds_cap"]
            + amendment["protected_final_reserve_gpu_seconds"]
            > cycle3["compute_caps_gpu_seconds"]["project_total"]):
        raise RegistryError("campaign v2 compute reservations exceed project cap or protected reserve")
    prior_output = cycle3["project_output_bytes_before"] + exploratory["output_bytes"]
    prior_output += sum(event["event_data"].get("output_bytes", 0)
                        for event in events[33:55]
                        if event["event_type"] == "workload_attempt"
                        and event["event_data"].get("outcome") == "terminal")
    if prior_output != 27_225_325:
        raise RegistryError("campaign v2 prior output use must count Cycle-3 attempts and v15 observation once")
    campaign_output_reserve = sum(row["record"]["workload_output_bytes_cap"]
                                  for row in amendment["records"])
    if (campaign_output_reserve > amendment["output_caps_bytes"]["aggregate"]
            or prior_output + campaign_output_reserve > cycle3["output_caps_bytes"]["aggregate"]):
        raise RegistryError("campaign v2 output reservations exceed aggregate output cap")

    prior_ids = {event["event_data"]["registration_id"] for event in events[:55]
                 if event["event_type"] == "registration"}
    prior_uuids = {event["event_data"]["record"].get("run_uuid") for event in events[:55]
                   if event["event_type"] == "registration"}
    registrations: dict[int, dict[str, Any]] = {}
    for offset, row in enumerate(amendment["records"], 1):
        record = dict(row["record"])
        if row["registration_id"] in prior_ids or record["run_uuid"] in prior_uuids:
            raise RegistryError("campaign v2 reuses a prior registration ID or UUID")
        record.update({"competitive_campaign_v2_event_index": 55,
                       "competitive_campaign_v2_event_hash": amendment_event["event_hash"]})
        index = 55 + offset
        if index >= len(events):
            break
        event, data = events[index], events[index]["event_data"]
        if (event["event_type"] != "registration"
                or data.get("registration_id") != row["registration_id"]
                or data.get("registration_kind") != row["registration_kind"]
                or data.get("cycle_id") != "g008-competitive-campaign-v2"
                or data.get("record") != record
                or hashlib.sha256(canonical_bytes(row["record"])).hexdigest() != row["record_sha256"]):
            raise RegistryError("campaign v2 registration differs from frozen amendment binding")
        registrations[index] = {"event": event, "record": record, "kind": row["registration_kind"]}
    if len(registrations) != 13:
        raise RegistryError("campaign v2 atomic batch must include all six train, six validation, and final registrations")

    train_registration_indices = list(range(56, 62))
    validation_registration_indices = list(range(62, 68))
    final_registration_index = 68
    training_by_identity: dict[tuple[str, str], int] = {}
    validation_by_identity: dict[tuple[str, str], int] = {}
    for ri in train_registration_indices:
        record = registrations[ri]["record"]
        training_by_identity[(record["pair_id"], record["strategy_role"])] = ri
    for ri in validation_registration_indices:
        record = registrations[ri]["record"]
        validation_by_identity[(record["pair_id"], record["strategy_role"])] = ri
    attempts: dict[str, dict[str, Any]] = {}
    attempts_by_registration: dict[int, dict[str, Any]] = {}
    attempt_history_by_registration: dict[int, list[dict[str, Any]]] = {}
    effective_records = {ri: dict(registration["record"])
                         for ri, registration in registrations.items()
                         if registration["kind"] in {"train", "validation"}}
    package_revision_event: dict[str, Any] | None = None
    retry_budget_event: dict[str, Any] | None = None
    package_revision_v2_event: dict[str, Any] | None = None
    retry_budget_v2_event: dict[str, Any] | None = None
    package_revision_v3_event: dict[str, Any] | None = None
    retry_budget_v3_event: dict[str, Any] | None = None
    package_revision_v4_event: dict[str, Any] | None = None
    package_revision_v5_event: dict[str, Any] | None = None
    dispatch_retry_v1_event: dict[str, Any] | None = None
    dispatch_retry_v1_used = False
    dispatch_retry_v2_event: dict[str, Any] | None = None
    dispatch_retry_v2_used = False
    dispatch_retry_v3_event: dict[str, Any] | None = None
    dispatch_retry_v3_used = False
    dispatch_retry_v4_event: dict[str, Any] | None = None
    dispatch_retry_v4_used = False
    dispatch_retry_v5_event: dict[str, Any] | None = None
    dispatch_retry_v5_used = False
    artifact_events: dict[int, int] = {}
    validation_results: dict[int, int] = {}
    latest_version = amendment["current_version_number"]
    comparison_index: int | None = None
    candidate_freeze_index: int | None = None
    final_result_index: int | None = None

    for index in range(69, len(events)):
        event, typ, data = events[index], events[index]["event_type"], events[index]["event_data"]
        if typ == "g008_competitive_campaign_v2_package_revision_v1":
            _validate_data(typ, data)
            if (package_revision_event is not None or artifact_events or validation_results
                    or comparison_index is not None or candidate_freeze_index is not None
                    or index == 0
                    or events[index - 1]["event_type"] != "workload_attempt"
                    or data["prior_head_hash"] != events[index - 1]["event_hash"]
                    or data["failed_attempt_event_index"] != index - 1
                    or data["failed_attempt_event_hash"] != events[index - 1]["event_hash"]
                    or data["amendment_event_index"] != 55
                    or data["amendment_event_hash"] != events[55]["event_hash"]):
                raise RegistryError("campaign v2 package revision must append once at the exact failed-attempt head")
            failed_i = data["failed_registration_event_index"]
            failed_state = attempts_by_registration.get(failed_i)
            failed = events[index - 1]["event_data"]
            if (failed_state is None or failed_state["phase"] != "terminal"
                    or failed_state["terminal_index"] != index - 1
                    or failed["attempt_id"] not in attempts
                    or failed.get("terminal_status") != "ERROR" or failed.get("outcome") != "terminal"
                    or failed.get("registration_event_index") != failed_i
                    or failed.get("registration_event_hash") != events[failed_i]["event_hash"]
                    or data["failed_registration_event_hash"] != events[failed_i]["event_hash"]
                    or failed.get("package_sha256") != effective_records[failed_i]["package_sha256"]
                    or any(failed.get(key) != effective_records[failed_i][key] for key in (
                        "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256",
                        "trainer_sha256", "engine_sha256"))
                    or failed_i not in train_registration_indices):
                raise RegistryError("campaign v2 package revision must bind the exact registered terminal ERROR")
            for binding in data["package_bindings"]:
                ri = binding["registration_event_index"]
                original = registrations[ri]["record"]
                if (binding["registration_event_hash"] != events[ri]["event_hash"]
                        or binding["package_sha256"] == effective_records[ri]["package_sha256"]
                        or any(binding[key] != original[key] for key in (
                            "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256",
                            "trainer_sha256", "engine_sha256"))):
                    raise RegistryError("campaign v2 package revision changed a component hash or kept the old package")
            package_revision_event = event
            for binding in data["package_bindings"]:
                effective_records[binding["registration_event_index"]]["package_sha256"] = binding["package_sha256"]
            continue

        if typ == "g008_competitive_campaign_v2_retry_budget_v1":
            _validate_data(typ, data)
            if (retry_budget_event is not None or package_revision_event is None
                    or index != package_revision_event["event_index"] + 1
                    or event["prev_event_hash"] != package_revision_event["event_hash"]
                    or data["prior_head_hash"] != package_revision_event["event_hash"]
                    or data["package_revision_event_index"] != package_revision_event["event_index"]
                    or data["package_revision_event_hash"] != package_revision_event["event_hash"]
                    or data["failed_attempt_event_index"] != package_revision_event["event_data"]["failed_attempt_event_index"]
                    or data["failed_attempt_event_hash"] != package_revision_event["event_data"]["failed_attempt_event_hash"]
                    or data["failed_registration_event_index"] != package_revision_event["event_data"]["failed_registration_event_index"]
                    or data["failed_registration_event_hash"] != package_revision_event["event_data"]["failed_registration_event_hash"]
                    or data["project_experiment_gpu_seconds_cap"] != cycle3["compute_caps_gpu_seconds"]["project_experiment"]
                    or data["project_total_gpu_seconds_cap"] != cycle3["compute_caps_gpu_seconds"]["project_total"]
                    or data["protected_final_reserve_gpu_seconds"] != amendment["protected_final_reserve_gpu_seconds"]
                    or prior_gpu + data["effective_total_gpu_seconds_cap"]
                    > data["project_experiment_gpu_seconds_cap"]
                    or prior_gpu + data["effective_total_gpu_seconds_cap"]
                    + data["protected_final_reserve_gpu_seconds"] > data["project_total_gpu_seconds_cap"]):
                raise RegistryError("campaign v2 retry budget must bind the sole failed attempt and preserve project caps/reserve")
            retry_budget_event = event
            continue

        if typ == "g008_competitive_campaign_v2_package_revision_v2":
            _validate_data(typ, data)
            prior_retry = retry_budget_event["event_data"] if retry_budget_event is not None else None
            failed = events[78]["event_data"] if len(events) > 78 else {}
            binding = data["package_binding"]
            original = registrations[56]["record"]
            if (package_revision_v2_event is not None or retry_budget_v2_event is not None
                    or package_revision_event is None or retry_budget_event is None
                    or artifact_events or validation_results or comparison_index is not None
                    or candidate_freeze_index is not None or index != 79
                    or events[78]["event_type"] != "workload_attempt"
                    or data["prior_head_hash"] != events[78]["event_hash"]
                    or event["prev_event_hash"] != events[78]["event_hash"]
                    or data["prior_package_revision_event_index"] != package_revision_event["event_index"]
                    or data["prior_package_revision_event_hash"] != package_revision_event["event_hash"]
                    or data["prior_retry_budget_event_index"] != retry_budget_event["event_index"]
                    or data["prior_retry_budget_event_hash"] != retry_budget_event["event_hash"]
                    or data["failed_registration_event_index"] != 56
                    or data["failed_registration_event_hash"] != registrations[56]["event"]["event_hash"]
                    or data["failed_attempt_event_index"] != 78
                    or data["failed_attempt_event_hash"] != events[78]["event_hash"]
                    or prior_retry is None or prior_retry["retry_attempt_id"] != "g008-v2-pair1-baseline-a2"
                    or failed.get("attempt_id") != "g008-v2-pair1-baseline-a2"
                    or failed.get("registration_event_index") != 56
                    or failed.get("terminal_status") != "ERROR"
                    or failed.get("actual_gpu_seconds") != 36
                    or failed.get("assigned_version") != 19
                    or binding["registration_event_index"] != 56
                    or binding["registration_event_hash"] != registrations[56]["event"]["event_hash"]
                    or binding["package_sha256"] == effective_records[56]["package_sha256"]
                    or any(binding[key] != original[key] for key in (
                        "config_sha256", "seed_plan_sha256", "backend_sha256", "engine_sha256"))):
                raise RegistryError("campaign v2 second package revision must bind only the exact a2 ERROR and row 56")
            package_revision_v2_event = event
            effective_records[56].update({key: binding[key] for key in (
                "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                "backend_sha256", "trainer_sha256", "engine_sha256")})
            continue

        if typ == "g008_competitive_campaign_v2_retry_budget_v2":
            _validate_data(typ, data)
            prior_retry = retry_budget_event["event_data"] if retry_budget_event is not None else None
            failed = events[78]["event_data"] if len(events) > 78 else {}
            if (retry_budget_v2_event is not None or package_revision_v2_event is None
                    or retry_budget_event is None
                    or index != package_revision_v2_event["event_index"] + 1
                    or event["prev_event_hash"] != package_revision_v2_event["event_hash"]
                    or data["prior_head_hash"] != package_revision_v2_event["event_hash"]
                    or data["package_revision_event_index"] != package_revision_v2_event["event_index"]
                    or data["package_revision_event_hash"] != package_revision_v2_event["event_hash"]
                    or data["prior_retry_budget_event_index"] != retry_budget_event["event_index"]
                    or data["prior_retry_budget_event_hash"] != retry_budget_event["event_hash"]
                    or data["failed_attempt_event_index"] != 78
                    or data["failed_attempt_event_hash"] != events[78]["event_hash"]
                    or data["failed_registration_event_index"] != 56
                    or data["failed_registration_event_hash"] != registrations[56]["event"]["event_hash"]
                    or prior_retry is None or prior_retry["retry_attempt_id"] != "g008-v2-pair1-baseline-a2"
                    or failed.get("attempt_id") != "g008-v2-pair1-baseline-a2"
                    or failed.get("terminal_status") != "ERROR"
                    or data["additional_gpu_seconds"] != failed.get("actual_gpu_seconds")
                    or data["project_experiment_gpu_seconds_cap"] != cycle3["compute_caps_gpu_seconds"]["project_experiment"]
                    or data["project_total_gpu_seconds_cap"] != cycle3["compute_caps_gpu_seconds"]["project_total"]
                    or data["protected_final_reserve_gpu_seconds"] != amendment["protected_final_reserve_gpu_seconds"]
                    or prior_gpu + data["effective_total_gpu_seconds_cap"]
                    > data["project_experiment_gpu_seconds_cap"]
                    or prior_gpu + data["effective_total_gpu_seconds_cap"]
                    + data["protected_final_reserve_gpu_seconds"] > data["project_total_gpu_seconds_cap"]):
                raise RegistryError("campaign v2 second retry budget must bind the exact a2 ERROR and preserve caps")
            retry_budget_v2_event = event
            continue

        if typ == "g008_competitive_campaign_v2_package_revision_v3":
            _validate_data(typ, data)
            failed_event = events[89] if len(events) > 89 else {}
            failed = failed_event.get("event_data", {})
            if (package_revision_v3_event is not None or retry_budget_v3_event is not None
                    or any(ri in artifact_events for ri in range(57, 62))
                    or validation_results or comparison_index is not None
                    or candidate_freeze_index is not None or index != 90
                    or event["prev_event_hash"] != events[89]["event_hash"]
                    or data["prior_head_hash"] != events[89]["event_hash"]
                    or data["prior_package_revision_event_index"] != 73
                    or data["prior_package_revision_event_hash"] != events[73]["event_hash"]
                    or data["prior_retry_budget_event_index"] != 74
                    or data["prior_retry_budget_event_hash"] != events[74]["event_hash"]
                    or data["prior_package_revision_v2_event_index"] != 79
                    or data["prior_package_revision_v2_event_hash"] != events[79]["event_hash"]
                    or data["prior_retry_budget_v2_event_index"] != 80
                    or data["prior_retry_budget_v2_event_hash"] != events[80]["event_hash"]
                    or data["failed_registration_event_index"] != 57
                    or data["failed_registration_event_hash"] != registrations[57]["event"]["event_hash"]
                    or data["failed_attempt_event_index"] != 89
                    or data["failed_attempt_event_hash"] != events[89]["event_hash"]
                    or failed.get("registration_event_index") != 57
                    or failed.get("terminal_status") != "ERROR"
                    or failed.get("actual_gpu_seconds") != 37
                    or failed.get("assigned_version") != 21
                    or failed.get("package_sha256") != effective_records[57]["package_sha256"]
                    or any(failed.get(key) != effective_records[57][key] for key in (
                        "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256",
                        "trainer_sha256", "engine_sha256"))):
                raise RegistryError("campaign v2 third package revision must bind exact row-57 ERROR at head 89")
            row56_fixed = effective_records[56]
            for binding in data["package_bindings"]:
                ri = binding["registration_event_index"]
                original = registrations[ri]["record"]
                if (binding["registration_event_hash"] != registrations[ri]["event"]["event_hash"]
                        or binding["package_sha256"] == effective_records[ri]["package_sha256"]
                        or any(binding[key] != original[key] for key in (
                            "config_sha256", "seed_plan_sha256", "backend_sha256", "engine_sha256"))
                        or binding["source_sha256"] != row56_fixed["source_sha256"]
                        or binding["trainer_sha256"] != row56_fixed["trainer_sha256"]):
                    raise RegistryError("campaign v2 third package revision changed a frozen row or trainer fix")
                effective_records[ri].update({key: binding[key] for key in (
                    "package_sha256", "source_sha256", "trainer_sha256")})
            package_revision_v3_event = event
            continue

        if typ == "g008_competitive_campaign_v2_package_revision_v4":
            _validate_data(typ, data)
            rejected_attempts = [events[i] for i in data["rejected_attempt_event_indexes"]]
            if (package_revision_v4_event is not None or index != 124
                    or event["prev_event_hash"] != events[123]["event_hash"]
                    or data["prior_head_hash"] != events[123]["event_hash"]
                    or data["rejected_attempt_event_hashes"] != [item["event_hash"] for item in rejected_attempts]
                    or any(item.get("event_type") != "workload_attempt"
                           or item["event_data"].get("registration_event_index") != 62
                           or item["event_data"].get("outcome") != "pre_assignment_rejected"
                           or item["event_data"].get("actual_gpu_seconds") != 0
                           or item["event_data"].get("assigned_version") is not None
                           for item in rejected_attempts)
                    or len(artifact_events) != 6 or validation_results or comparison_index is not None
                    or candidate_freeze_index is not None):
                raise RegistryError("campaign v2 validation package revision must follow both exact zero-cost row-62 rejections")
            for binding in data["package_bindings"]:
                ri = binding["registration_event_index"]
                original = registrations[ri]["record"]
                if (binding["registration_event_hash"] != registrations[ri]["event"]["event_hash"]
                        or binding["package_sha256"] == original["package_sha256"]
                        or any(binding[key] != original[key] for key in (
                            "config_sha256", "seed_plan_sha256", "backend_sha256", "engine_sha256"))
                        or binding["source_sha256"] != events[95]["event_data"]["source_sha256"]
                        or binding["trainer_sha256"] != events[95]["event_data"]["trainer_sha256"]):
                    raise RegistryError("campaign v2 validation package revision changed a frozen config or training source")
                effective_records[ri].update({key: binding[key] for key in (
                    "package_sha256", "source_sha256", "trainer_sha256")})
            package_revision_v4_event = event
            continue

        if typ == "g008_competitive_campaign_v2_package_revision_v5":
            _validate_data(typ, data)
            rejected_attempts = [events[i] for i in data["rejected_attempt_event_indexes"]]
            if (package_revision_v5_event is not None or index != 137
                    or package_revision_v4_event is None
                    or event["prev_event_hash"] != events[136]["event_hash"]
                    or data["prior_head_hash"] != events[136]["event_hash"]
                    or data["rejected_attempt_event_hashes"] != [item["event_hash"] for item in rejected_attempts]
                    or any(item.get("event_type") != "workload_attempt"
                           or item["event_data"].get("registration_event_index") != 62
                           or item["event_data"].get("outcome") != "pre_assignment_rejected"
                           or item["event_data"].get("actual_gpu_seconds") != 0
                           or item["event_data"].get("assigned_version") is not None
                           or item["event_data"].get("no_assignment_metadata", {}).get("post_push_recheck_current_version") != 26
                           for item in rejected_attempts)
                    or len(artifact_events) != 6 or validation_results or comparison_index is not None
                    or candidate_freeze_index is not None):
                raise RegistryError("campaign v2 fifth validation package revision must follow all exact zero-cost row-62 rejections")
            for binding in data["package_bindings"]:
                ri = binding["registration_event_index"]
                original = registrations[ri]["record"]
                if (binding["registration_event_hash"] != registrations[ri]["event"]["event_hash"]
                        or binding["package_sha256"] == effective_records[ri]["package_sha256"]
                        or any(binding[key] != original[key] for key in (
                            "config_sha256", "seed_plan_sha256", "backend_sha256", "engine_sha256"))
                        or binding["source_sha256"] != events[95]["event_data"]["source_sha256"]
                        or binding["trainer_sha256"] != events[95]["event_data"]["trainer_sha256"]):
                    raise RegistryError("campaign v2 fifth validation package revision changed frozen config or training source")
                effective_records[ri].update({key: binding[key] for key in (
                    "package_sha256", "source_sha256", "trainer_sha256")})
            package_revision_v5_event = event
            continue

        if typ == "g008_competitive_campaign_v2_retry_budget_v3":
            _validate_data(typ, data)
            prior_retry = retry_budget_v2_event["event_data"] if retry_budget_v2_event is not None else None
            revision = package_revision_v3_event
            if (retry_budget_v3_event is not None or revision is None or prior_retry is None
                    or index != revision["event_index"] + 1
                    or event["prev_event_hash"] != revision["event_hash"]
                    or data["prior_head_hash"] != revision["event_hash"]
                    or data["package_revision_event_index"] != revision["event_index"]
                    or data["package_revision_event_hash"] != revision["event_hash"]
                    or data["prior_retry_budget_event_index"] != retry_budget_v2_event["event_index"]
                    or data["prior_retry_budget_event_hash"] != retry_budget_v2_event["event_hash"]
                    or data["failed_attempt_event_index"] != 89
                    or data["failed_attempt_event_hash"] != events[89]["event_hash"]
                    or data["failed_registration_event_index"] != 57
                    or data["failed_registration_event_hash"] != registrations[57]["event"]["event_hash"]
                    or data["effective_training_gpu_seconds_cap"] != prior_retry["effective_training_gpu_seconds_cap"]
                    or data["effective_evaluation_gpu_seconds_cap"] != prior_retry["effective_evaluation_gpu_seconds_cap"]
                    or data["effective_total_gpu_seconds_cap"] != prior_retry["effective_total_gpu_seconds_cap"]
                    or data["project_experiment_gpu_seconds_cap"] != prior_retry["project_experiment_gpu_seconds_cap"]
                    or data["project_total_gpu_seconds_cap"] != prior_retry["project_total_gpu_seconds_cap"]
                    or data["protected_final_reserve_gpu_seconds"] != prior_retry["protected_final_reserve_gpu_seconds"]
                    or data["additional_gpu_seconds"] != events[89]["event_data"]["actual_gpu_seconds"]
                    or prior_gpu + data["effective_total_gpu_seconds_cap"]
                       > data["project_experiment_gpu_seconds_cap"]
                    or prior_gpu + data["effective_total_gpu_seconds_cap"]
                       + data["protected_final_reserve_gpu_seconds"] > data["project_total_gpu_seconds_cap"]):
                raise RegistryError("campaign v2 third retry budget must preserve caps and bind exact row-57 ERROR")
            retry_budget_v3_event = event
            continue

        if typ == "g008_competitive_campaign_v2_dispatch_retry_v1":
            _validate_data(typ, data)
            rejected = events[data["rejected_attempt_event_index"]]
            terminal = rejected.get("event_data", {})
            preflight = events[117].get("event_data", {}) if len(events) > 117 else {}
            no_assignment = terminal.get("no_assignment_metadata", {})
            if (dispatch_retry_v1_event is not None or dispatch_retry_v1_used
                    or index != 120 or event["prev_event_hash"] != events[119]["event_hash"]
                    or data["prior_head_hash"] != events[119]["event_hash"]
                    or data["rejected_attempt_event_index"] != 119
                    or data["rejected_attempt_event_hash"] != events[119]["event_hash"]
                    or rejected.get("event_type") != "workload_attempt"
                    or terminal.get("attempt_id") != "g008-v2-validation-seed-pair-1-baseline-a1"
                    or terminal.get("registration_event_index") != 62
                    or terminal.get("outcome") != "pre_assignment_rejected"
                    or terminal.get("actual_gpu_seconds") != 0 or terminal.get("output_bytes") != 0
                    or terminal.get("assigned_version") is not None
                    or no_assignment.get("no_version_assigned") is not True
                    or no_assignment.get("pre_push_current_version") != 26
                    or no_assignment.get("post_push_current_version") != 26
                    or no_assignment.get("post_push_recheck_current_version") != 26
                    or no_assignment.get("verification_source") != "pinned Kaggle SDK get_kernel"
                    or no_assignment.get("metadata_check_count") != 2
                    or preflight.get("no_outside_writer_confirmed") is not True
                    or data["registration_event_hash"] != registrations[62]["event"]["event_hash"]
                    or data["expected_current_version"] != latest_version):
                raise RegistryError("campaign v2 dispatch retry must bind exact no-assignment evidence and row-62 preflight")
            dispatch_retry_v1_event = event
            continue

        if typ == "g008_competitive_campaign_v2_dispatch_retry_v2":
            _validate_data(typ, data)
            rejected = events[123]
            terminal = rejected.get("event_data", {})
            if (dispatch_retry_v2_event is not None or dispatch_retry_v2_used or index != 125
                    or package_revision_v4_event is None or package_revision_v4_event["event_index"] != 124
                    or event["prev_event_hash"] != events[124]["event_hash"]
                    or data["prior_head_hash"] != events[124]["event_hash"]
                    or data["package_revision_event_hash"] != events[124]["event_hash"]
                    or data["rejected_attempt_event_hash"] != rejected["event_hash"]
                    or terminal.get("attempt_id") != "g008-v2-validation-seed-pair-1-baseline-a2"
                    or terminal.get("registration_event_index") != 62
                    or terminal.get("outcome") != "pre_assignment_rejected"
                    or terminal.get("actual_gpu_seconds") != 0 or terminal.get("assigned_version") is not None
                    or terminal.get("no_assignment_metadata", {}).get("post_push_recheck_current_version") != 26
                    or data["registration_event_hash"] != registrations[62]["event"]["event_hash"]
                    or data["expected_current_version"] != latest_version):
                raise RegistryError("campaign v2 second dispatch retry must bind exact row-62 no-assignment and revised packages")
            dispatch_retry_v2_event = event
            continue

        if typ == "g008_competitive_campaign_v2_dispatch_retry_v3":
            _validate_data(typ, data)
            rejected = events[128]
            terminal = rejected.get("event_data", {})
            if (dispatch_retry_v3_event is not None or dispatch_retry_v3_used or index != 129
                    or package_revision_v4_event is None
                    or event["prev_event_hash"] != events[128]["event_hash"]
                    or data["prior_head_hash"] != events[128]["event_hash"]
                    or data["package_revision_event_hash"] != package_revision_v4_event["event_hash"]
                    or data["rejected_attempt_event_hash"] != rejected["event_hash"]
                    or terminal.get("attempt_id") != "g008-v2-validation-seed-pair-1-baseline-a3"
                    or terminal.get("registration_event_index") != 62
                    or terminal.get("outcome") != "pre_assignment_rejected"
                    or terminal.get("rejection_evidence", {}).get("http_status") != 400
                    or terminal.get("actual_gpu_seconds") != 0 or terminal.get("assigned_version") is not None
                    or terminal.get("no_assignment_metadata", {}).get("post_push_recheck_current_version") != 26
                    or data["registration_event_hash"] != registrations[62]["event"]["event_hash"]
                    or data["expected_current_version"] != latest_version):
                raise RegistryError("campaign v2 third dispatch retry must bind exact row-62 HTTP-400 rejection")
            dispatch_retry_v3_event = event
            continue

        if typ == "g008_competitive_campaign_v2_dispatch_retry_v4":
            _validate_data(typ, data)
            rejected = events[132]
            terminal = rejected.get("event_data", {})
            if (dispatch_retry_v4_event is not None or dispatch_retry_v4_used or index != 133
                    or package_revision_v4_event is None
                    or event["prev_event_hash"] != events[132]["event_hash"]
                    or data["prior_head_hash"] != events[132]["event_hash"]
                    or data["package_revision_event_hash"] != package_revision_v4_event["event_hash"]
                    or data["rejected_attempt_event_hash"] != rejected["event_hash"]
                    or terminal.get("attempt_id") != "g008-v2-validation-seed-pair-1-baseline-a4"
                    or terminal.get("registration_event_index") != 62
                    or terminal.get("outcome") != "pre_assignment_rejected"
                    or terminal.get("rejection_evidence", {}).get("http_status") != 400
                    or terminal.get("actual_gpu_seconds") != 0 or terminal.get("assigned_version") is not None
                    or terminal.get("no_assignment_metadata", {}).get("post_push_recheck_current_version") != 26
                    or data["registration_event_hash"] != registrations[62]["event"]["event_hash"]
                    or data["expected_current_version"] != latest_version):
                raise RegistryError("campaign v2 fourth dispatch retry must bind exact row-62 HTTP-400 rejection")
            dispatch_retry_v4_event = event
            continue

        if typ == "g008_competitive_campaign_v2_dispatch_retry_v5":
            _validate_data(typ, data)
            rejected = events[136]
            terminal = rejected.get("event_data", {})
            if (dispatch_retry_v5_event is not None or dispatch_retry_v5_used or index != 138
                    or package_revision_v5_event is None or package_revision_v5_event["event_index"] != 137
                    or event["prev_event_hash"] != events[137]["event_hash"]
                    or data["prior_head_hash"] != events[137]["event_hash"]
                    or data["package_revision_event_hash"] != events[137]["event_hash"]
                    or data["rejected_attempt_event_hash"] != rejected["event_hash"]
                    or terminal.get("attempt_id") != "g008-v2-validation-seed-pair-1-baseline-a5"
                    or terminal.get("registration_event_index") != 62
                    or terminal.get("outcome") != "pre_assignment_rejected"
                    or terminal.get("rejection_evidence", {}).get("http_status") != 400
                    or terminal.get("actual_gpu_seconds") != 0 or terminal.get("assigned_version") is not None
                    or terminal.get("no_assignment_metadata", {}).get("post_push_recheck_current_version") != 26
                    or data["registration_event_hash"] != registrations[62]["event"]["event_hash"]
                    or data["expected_current_version"] != latest_version):
                raise RegistryError("campaign v2 fifth dispatch retry must bind exact row-62 HTTP-400 rejection")
            dispatch_retry_v5_event = event
            continue

        if typ == "workload_preflight":
            ri, aid = data["registration_event_index"], data["attempt_id"]
            registration = registrations.get(ri)
            prior_state = attempts_by_registration.get(ri)
            retry_allowed = (prior_state is not None and prior_state["phase"] == "terminal"
                             and prior_state["terminal"].get("terminal_status") == "ERROR"
                             and package_revision_event is not None
                             and retry_budget_event is not None
                             and index == retry_budget_event["event_index"] + 1
                             and aid == retry_budget_event["event_data"]["retry_attempt_id"]
                             and ri == retry_budget_event["event_data"]["failed_registration_event_index"]
                             and len(attempt_history_by_registration.get(ri, [])) == 1)
            retry_v2_allowed = (prior_state is not None and prior_state["phase"] == "terminal"
                                and prior_state["terminal"].get("terminal_status") == "ERROR"
                                and package_revision_v2_event is not None
                                and retry_budget_v2_event is not None
                                and index == retry_budget_v2_event["event_index"] + 1
                                and aid == retry_budget_v2_event["event_data"]["retry_attempt_id"]
                                and ri == 56
                                and len(attempt_history_by_registration.get(ri, [])) == 2
                                and prior_state["terminal_index"] == 78)
            retry_v3_allowed = (prior_state is not None and prior_state["phase"] == "terminal"
                                and prior_state["terminal"].get("terminal_status") == "ERROR"
                                and retry_budget_v3_event is not None
                                and index == retry_budget_v3_event["event_index"] + 1
                                and aid == retry_budget_v3_event["event_data"]["retry_attempt_id"]
                                and ri == 57
                                and len(attempt_history_by_registration.get(ri, [])) == 1
                                and prior_state["terminal_index"] == 89)
            dispatch_retry_allowed = (prior_state is not None and prior_state["phase"] == "terminal"
                                      and prior_state["terminal"].get("outcome") == "pre_assignment_rejected"
                                      and dispatch_retry_v1_event is not None and not dispatch_retry_v1_used
                                      and index == dispatch_retry_v1_event["event_index"] + 1
                                      and aid == dispatch_retry_v1_event["event_data"]["retry_attempt_id"]
                                      and ri == dispatch_retry_v1_event["event_data"]["registration_event_index"]
                                      and len(attempt_history_by_registration.get(ri, [])) == 1
                                      and prior_state["terminal_index"] == 119)
            dispatch_retry_v2_allowed = (prior_state is not None and prior_state["phase"] == "terminal"
                                         and prior_state["terminal"].get("outcome") == "pre_assignment_rejected"
                                         and dispatch_retry_v2_event is not None and not dispatch_retry_v2_used
                                         and index == dispatch_retry_v2_event["event_index"] + 1
                                         and aid == dispatch_retry_v2_event["event_data"]["retry_attempt_id"]
                                         and ri == 62
                                         and len(attempt_history_by_registration.get(ri, [])) == 2
                                         and prior_state["terminal_index"] == 123)
            dispatch_retry_v3_allowed = (prior_state is not None and prior_state["phase"] == "terminal"
                                         and prior_state["terminal"].get("outcome") == "pre_assignment_rejected"
                                         and dispatch_retry_v3_event is not None and not dispatch_retry_v3_used
                                         and index == dispatch_retry_v3_event["event_index"] + 1
                                         and aid == dispatch_retry_v3_event["event_data"]["retry_attempt_id"]
                                         and ri == 62
                                         and len(attempt_history_by_registration.get(ri, [])) == 3
                                         and prior_state["terminal_index"] == 128)
            dispatch_retry_v4_allowed = (prior_state is not None and prior_state["phase"] == "terminal"
                                         and prior_state["terminal"].get("outcome") == "pre_assignment_rejected"
                                         and dispatch_retry_v4_event is not None and not dispatch_retry_v4_used
                                         and index == dispatch_retry_v4_event["event_index"] + 1
                                         and aid == dispatch_retry_v4_event["event_data"]["retry_attempt_id"]
                                         and ri == 62
                                         and len(attempt_history_by_registration.get(ri, [])) == 4
                                         and prior_state["terminal_index"] == 132)
            dispatch_retry_v5_allowed = (prior_state is not None and prior_state["phase"] == "terminal"
                                         and prior_state["terminal"].get("outcome") == "pre_assignment_rejected"
                                         and dispatch_retry_v5_event is not None and not dispatch_retry_v5_used
                                         and index == dispatch_retry_v5_event["event_index"] + 1
                                         and aid == dispatch_retry_v5_event["event_data"]["retry_attempt_id"]
                                         and ri == 62
                                         and len(attempt_history_by_registration.get(ri, [])) == 5
                                         and prior_state["terminal_index"] == 136)
            if (registration is None or aid in attempts
                    or (prior_state is not None and not (retry_allowed or retry_v2_allowed or retry_v3_allowed
                                                         or dispatch_retry_allowed or dispatch_retry_v2_allowed
                                                         or dispatch_retry_v3_allowed or dispatch_retry_v4_allowed
                                                         or dispatch_retry_v5_allowed))
                    or (retry_budget_event is not None and index == retry_budget_event["event_index"] + 1
                        and not retry_allowed)
                    or (retry_budget_v2_event is not None
                        and index == retry_budget_v2_event["event_index"] + 1 and not retry_v2_allowed)
                    or (retry_budget_v3_event is not None
                        and index == retry_budget_v3_event["event_index"] + 1 and not retry_v3_allowed)
                    or data.get("cycle_id") != "g008-competitive-campaign-v2"
                    or data["registration_event_hash"] != events[ri]["event_hash"]
                    or data["registration_id"] != events[ri]["event_data"]["registration_id"]
                    or data["kernel_id"] != amendment["kernel_id"]
                    or data["expected_current_version"] != latest_version
                    or data["session_timeout_seconds"] != registration["record"]["session_timeout_seconds"]
                    or data["lock_head_hash"] != events[49]["event_hash"]):
                raise RegistryError("campaign v2 preflight changed registration, version, kernel, or benchmark lock")
            record, kind = registration["record"], registration["kind"]
            if kind in {"train", "validation"}:
                record = effective_records[ri]
                if retry_v2_allowed and data["expected_current_version"] != 19:
                    raise RegistryError("campaign v2 a3 retry must use exact version-19 cursor")
            if kind == "train":
                if data["implementation_variant_id"] != record["strategy_role"]:
                    raise RegistryError("campaign v2 training changed the paired strategy arm")
            elif kind == "validation":
                if (len(artifact_events) != 6 or len([state for state in attempts_by_registration.values()
                                                     if state["record"]["registration_kind"] == "train"
                                                     and state["phase"] == "terminal"
                                                     and state["terminal"]["terminal_status"] == "COMPLETE"]) != 6
                        or data["implementation_variant_id"] != "frozen-candidate"):
                    raise RegistryError("campaign v2 per-arm validation requires six attested COMPLETE checkpoints")
                train_ri = training_by_identity[(record["pair_id"], record["strategy_role"])]
                train_record = registrations[train_ri]["record"]
                train_row = amendment["records"][train_ri - 56]
                if (record["training_registration_id"] != events[train_ri]["event_data"]["registration_id"]
                        or record["training_run_uuid"] != train_record["run_uuid"]
                        or record["training_record_sha256"] != train_row["record_sha256"]):
                    raise RegistryError("campaign v2 validation registration changed exact training mapping")
                artifact_i = artifact_events.get(train_ri)
                if artifact_i is None:
                    raise RegistryError("campaign v2 validation lacks exact training artifact attestation")
            elif kind == "final":
                if (comparison_index is None or candidate_freeze_index is None
                        or data["implementation_variant_id"] != "frozen-candidate"):
                    raise RegistryError("campaign v2 final preflight requires completed comparison and candidate freeze")
            for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                        "backend_sha256", "trainer_sha256", "engine_sha256"):
                if data[key] != record[key]:
                    raise RegistryError("campaign v2 preflight changed frozen package/component hash")
            state = {"phase": "preflight", "last_index": index, "registration_index": ri,
                     "record": {**record, "registration_kind": kind}, "preflight": data,
                     "preflight_hash": event["event_hash"]}
            attempts[aid] = state
            attempts_by_registration[ri] = state
            if dispatch_retry_allowed:
                dispatch_retry_v1_used = True
            if dispatch_retry_v2_allowed:
                dispatch_retry_v2_used = True
            if dispatch_retry_v3_allowed:
                dispatch_retry_v3_used = True
            if dispatch_retry_v4_allowed:
                dispatch_retry_v4_used = True
            if dispatch_retry_v5_allowed:
                dispatch_retry_v5_used = True
            continue

        if typ in {"workload_push_dispatched", "workload_push_accepted", "workload_attempt"}:
            aid = data["attempt_id"]
            state = attempts.get(aid)
            if (state is None or state["phase"] == "terminal" or index != state["last_index"] + 1
                    or data.get("cycle_id") != "g008-competitive-campaign-v2"
                    or data["registration_event_index"] != state["registration_index"]
                    or data["registration_event_hash"] != events[state["registration_index"]]["event_hash"]):
                raise RegistryError("campaign v2 attempt lifecycle is not contiguous or changed registration")
            if typ == "workload_push_dispatched":
                if (state["phase"] != "preflight" or data["preflight_event_index"] != state["last_index"]
                        or data["preflight_event_hash"] != state["preflight_hash"]
                        or any(data[key] != state["preflight"][key] for key in (
                            "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                            "backend_sha256", "trainer_sha256", "engine_sha256"))):
                    raise RegistryError("campaign v2 dispatch must immediately bind exact preflight")
                state.update(phase="dispatched", last_index=index, dispatch_hash=event["event_hash"])
            elif typ == "workload_push_accepted":
                if (state["phase"] != "dispatched" or data["dispatch_event_hash"] != state["dispatch_hash"]
                        or data["preflight_event_index"] != index - 2
                        or data["implementation_variant_id"] != state["preflight"]["implementation_variant_id"]
                        or data["lock_head_hash"] != state["preflight"]["lock_head_hash"]
                        or data["assigned_version"] <= latest_version):
                    raise RegistryError("campaign v2 acceptance changed exact dispatch or version cursor")
                state.update(phase="accepted", last_index=index, version=data["assigned_version"])
            else:
                rejected = data["outcome"] == "pre_assignment_rejected"
                if rejected:
                    proof = data["no_assignment_metadata"]
                    if (state["phase"] != "dispatched" or data["assigned_version"] is not None
                            or data["actual_gpu_seconds"] != 0 or data["output_bytes"] != 0
                            or proof.get("no_version_assigned") is not True
                            or proof.get("pre_push_current_version") != state["preflight"]["expected_current_version"]
                            or proof.get("post_push_current_version") != state["preflight"]["expected_current_version"]
                            or proof.get("post_push_recheck_current_version") != state["preflight"]["expected_current_version"]
                            or proof.get("metadata_check_count") != 2
                            or proof.get("verification_source") != "pinned Kaggle SDK get_kernel"):
                        raise RegistryError("campaign v2 ambiguous dispatch lacks exact two-observation no-assignment proof")
                elif (state["phase"] != "accepted" or data["outcome"] != "terminal"
                        or data["accepted_event_index"] != state["last_index"]
                        or data["assigned_version"] != state["version"]
                        or any(data[key] != state["preflight"][key] for key in (
                            "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                            "backend_sha256", "trainer_sha256", "engine_sha256"))
                        or data["actual_gpu_seconds"] > state["record"]["workload_gpu_seconds_cap"]
                        or data["output_bytes"] > state["record"]["workload_output_bytes_cap"]):
                    raise RegistryError("campaign v2 terminal attempt exceeds reservation or accepted push")
                state.update(phase="terminal", last_index=index, terminal_index=index, terminal=data)
                if not rejected:
                    latest_version = data["assigned_version"]
                attempt_history_by_registration.setdefault(state["registration_index"], []).append(state)
            continue

        if typ == "competitive_training_artifact_v2":
            ri, attempt_i = data["training_registration_event_index"], data["training_attempt_event_index"]
            state = attempts_by_registration.get(ri)
            if (ri not in train_registration_indices or state is None or state["phase"] != "terminal"
                    or state["terminal"].get("terminal_status") != "COMPLETE"
                    or state["terminal_index"] != index - 1 or attempt_i != state["terminal_index"]
                    or data["training_registration_event_hash"] != events[ri]["event_hash"]
                    or data["training_attempt_event_hash"] != events[attempt_i]["event_hash"]
                    or data["step"] != state["record"]["update_budget"]
                    or data["package_sha256"] != state["record"]["package_sha256"]
                    or data["source_sha256"] != state["record"]["source_sha256"]
                    or data["config_sha256"] != state["record"]["config_sha256"]
                    or data["schedule_descriptor_sha256"] != state["record"]["schedule_descriptor_sha256"]
                    or ri in artifact_events):
                raise RegistryError("campaign v2 training artifact must immediately attest exact COMPLETE training output")
            artifact_events[ri] = index
            continue

        if typ == "competitive_validation_result_v2":
            vri, trai = data["validation_registration_event_index"], data["training_registration_event_index"]
            validation_state = attempts_by_registration.get(vri)
            training_state = attempts_by_registration.get(trai)
            identity = (registrations[vri]["record"]["pair_id"], registrations[vri]["record"]["strategy_role"])
            artifact_i = artifact_events.get(trai)
            if (vri not in validation_registration_indices or trai != training_by_identity.get(identity)
                    or validation_state is None or validation_state["phase"] != "terminal"
                    or validation_state["terminal"].get("terminal_status") != "COMPLETE"
                    or validation_state["terminal_index"] != index - 1
                    or training_state is None or training_state["phase"] != "terminal"
                    or training_state["terminal"].get("terminal_status") != "COMPLETE"
                    or artifact_i is None or vri in validation_results
                    or data["validation_registration_event_hash"] != events[vri]["event_hash"]
                    or data["validation_attempt_event_index"] != validation_state["terminal_index"]
                    or data["validation_attempt_event_hash"] != events[validation_state["terminal_index"]]["event_hash"]
                    or data["training_registration_event_hash"] != events[trai]["event_hash"]
                    or data["training_attempt_event_index"] != training_state["terminal_index"]
                    or data["training_attempt_event_hash"] != events[training_state["terminal_index"]]["event_hash"]
                    or data["training_artifact_event_index"] != artifact_i
                    or data["training_artifact_event_hash"] != events[artifact_i]["event_hash"]
                    or data["checkpoint_sha256"] != events[artifact_i]["event_data"]["checkpoint_sha256"]
                    or data["opponent_id"] != "g007-step-300"):
                raise RegistryError("campaign v2 validation result changed exact checkpoint/roster/run provenance")
            validation_results[vri] = index
            continue

        if typ == "competitive_strategy_comparison_v2":
            expected = _competitive_v2_comparison_payload(events[:index], created_at_utc=data["created_at_utc"])
            if comparison_index is not None or index == 0 or events[index - 1]["event_type"] != "competitive_validation_result_v2" or data != expected:
                raise RegistryError("campaign v2 strategy comparison must recompute all six paired validation results")
            comparison_index = index
            continue

        if typ == "competitive_candidate_freeze_v2":
            comparison = events[comparison_index]["event_data"] if comparison_index is not None else None
            selected = comparison.get("selected_candidate") if comparison else None
            if (candidate_freeze_index is not None or comparison_index != index - 1 or selected is None
                    or data["strategy_comparison_event_index"] != comparison_index
                    or data["strategy_comparison_event_hash"] != events[comparison_index]["event_hash"]):
                raise RegistryError("campaign v2 candidate freeze must immediately follow frozen paired comparison")
            selected_bindings = {
                "candidate_registration_event_index": selected["training_registration_event_index"],
                "candidate_registration_event_hash": selected["training_registration_event_hash"],
                "train_attempt_event_index": selected["training_attempt_event_index"],
                "train_attempt_event_hash": selected["training_attempt_event_hash"],
                "training_artifact_event_index": selected["training_artifact_event_index"],
                "training_artifact_event_hash": selected["training_artifact_event_hash"],
                "validation_registration_event_index": selected["validation_registration_event_index"],
                "validation_registration_event_hash": selected["validation_registration_event_hash"],
                "validation_attempt_event_index": selected["validation_attempt_event_index"],
                "validation_attempt_event_hash": selected["validation_attempt_event_hash"],
                "validation_result_event_index": selected["validation_result_event_index"],
                "validation_result_event_hash": selected["validation_result_event_hash"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "checkpoint_artifact_sha256": selected["checkpoint_sha256"],
                "output_manifest_sha256": selected["output_manifest_sha256"],
            }
            if any(data.get(key) != value for key, value in selected_bindings.items()):
                raise RegistryError("campaign v2 candidate freeze changed deterministic median checkpoint selection")
            candidate_freeze_index = index
            continue

        if typ == "competitive_final_result_v2":
            _validate_g008_v2_final_result_roster(
                registrations[final_registration_index]["record"], data)
            if (candidate_freeze_index is None or final_result_index is not None or index != len(events) - 1
                    or data["candidate_freeze_event_index"] != candidate_freeze_index
                    or data["candidate_freeze_event_hash"] != events[candidate_freeze_index]["event_hash"]
                    or data["final_registration_event_index"] != final_registration_index
                    or data["final_registration_event_hash"] != events[final_registration_index]["event_hash"]):
                raise RegistryError("campaign v2 final result must bind the pre-registered holdout after candidate freeze")
            final_state = attempts_by_registration.get(final_registration_index)
            if (final_state is None or final_state["phase"] != "terminal"
                    or final_state["terminal"].get("terminal_status") != "COMPLETE"
                    or data["final_attempt_event_index"] != final_state["terminal_index"]
                    or data["final_attempt_event_hash"] != events[final_state["terminal_index"]]["event_hash"]
                    or data["candidate_checkpoint_sha256"] != events[candidate_freeze_index]["event_data"]["checkpoint_sha256"]
                    or data["opponent_id"] != "open-space"):
                raise RegistryError("campaign v2 final result changed exact COMPLETE holdout or selected checkpoint")
            final_result_index = index
            continue

        if typ in {"competitive_validation_result", "competitive_candidate_freeze", "competitive_final_result"}:
            raise RegistryError("campaign v2 cannot use v1 lifecycle result events")
        raise RegistryError("event type is not permitted in campaign v2 lifecycle")

    if final_result_index is not None and final_result_index != len(events) - 1:
        raise RegistryError("campaign v2 final result is the terminal event")
    for state in attempts.values():
        if state["phase"] != "terminal" and state["last_index"] != len(events) - 1:
            raise RegistryError("pending campaign v2 workload blocks later events")
    if len(artifact_events) > 0 and len(artifact_events) < 6 and any(
            ri in attempts_by_registration and attempts_by_registration[ri]["phase"] == "terminal"
            and attempts_by_registration[ri]["record"]["registration_kind"] != "train"
            for ri in attempts_by_registration):
        raise RegistryError("campaign v2 validation cannot start before all six training artifacts")
    actual = {"train": 0, "validation": 0, "final": 0}
    reserved = {"train": 0, "validation": 0, "final": 0}
    for ri, registration in registrations.items():
        kind = registration["kind"]
        state = attempts_by_registration.get(ri)
        history = attempt_history_by_registration.get(ri, [])
        spent = sum(item["terminal"]["actual_gpu_seconds"] for item in history)
        actual[kind] += spent
        if state is not None and state["phase"] == "terminal":
            continue
        if state is not None:
            if (kind == "train" and retry_budget_v3_event is not None
                    and ri == retry_budget_v3_event["event_data"]["failed_registration_event_index"]
                    and len(history) == 1):
                reserved[kind] += retry_budget_v3_event["event_data"]["retry_timeout_seconds"]
            elif kind == "train" and retry_budget_v2_event is not None and len(history) == 2:
                reserved[kind] += retry_budget_v2_event["event_data"]["retry_timeout_seconds"]
            elif kind == "train" and retry_budget_event is not None and len(history) == 1:
                reserved[kind] += retry_budget_event["event_data"]["retry_timeout_seconds"]
            else:
                reserved[kind] += max(0, registration["record"]["workload_gpu_seconds_cap"] - spent)
        else:
            reserved[kind] += registration["record"]["session_timeout_seconds"]
    effective_budget = retry_budget_v3_event or retry_budget_v2_event or retry_budget_event
    training_cap = (effective_budget["event_data"]["effective_training_gpu_seconds_cap"]
                    if effective_budget is not None else amendment["training_gpu_seconds_cap"])
    evaluation_cap = (effective_budget["event_data"]["effective_evaluation_gpu_seconds_cap"]
                      if effective_budget is not None else amendment["evaluation_gpu_seconds_cap"])
    total_cap = (effective_budget["event_data"]["effective_total_gpu_seconds_cap"]
                 if effective_budget is not None else amendment["total_gpu_seconds_cap"])
    if (actual["train"] + reserved["train"] > training_cap
            or actual["validation"] + actual["final"] + reserved["validation"] + reserved["final"]
            > evaluation_cap
            or sum(actual.values()) + sum(reserved.values()) > total_cap):
        raise RegistryError("campaign v2 actual use/reservations exceed frozen cumulative budget")


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
            if (event_type in {"recovery_cycle_amendment_v2", "g008_competitive_campaign_amendment_v1",
                               "g008_competitive_campaign_amendment_v2"}
                    or (event_type == "registration" and event_data.get("cycle_id") in {
                        "g008-recovery-cycle-3", "g008-competitive-campaign-v1",
                        "g008-competitive-campaign-v2"})):
                raise RegistryError("amendments and campaign registrations must use their atomic batch API")
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
            if any(e["event_type"] == "g008_competitive_campaign_amendment_v2" for e in candidate_events):
                _validate_g008_competitive_campaign_v2(candidate_events)
            elif any(e["event_type"] == "g008_competitive_campaign_amendment_v1" for e in candidate_events):
                _validate_g008_competitive_campaign(candidate_events)
            elif any(e["event_type"] == "recovery_cycle_amendment_v2" for e in candidate_events):
                _validate_recovery_cycle3_suffix(candidate_events)
            elif any(e["event_type"] == "recovery_cycle_amendment" for e in candidate_events):
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

    def register_recovery_cycle3_batch(self, amendment: dict[str, Any],
                                       workloads: list[dict[str, Any]], *,
                                       expected_parent_hash: str | None = None,
                                       expected_parent_index: int | None = None) -> list[dict[str, Any]]:
        """Validate and atomically append Cycle-3 events 27-29 under one registry lock."""
        expected_parent_hash = _CYCLE3_PARENT_EVENT_HASH if expected_parent_hash is None else expected_parent_hash
        expected_parent_index = _CYCLE3_EVENT_INDEX - 1 if expected_parent_index is None else expected_parent_index
        if expected_parent_hash != _CYCLE3_PARENT_EVENT_HASH or expected_parent_index != _CYCLE3_EVENT_INDEX - 1:
            raise RegistryError("Cycle-3 batch requires the frozen terminal Cycle-2 parent anchor")
        if not isinstance(workloads, list) or len(workloads) != 2:
            raise RegistryError("Cycle-3 batch requires exactly benchmark and train workloads")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            events = _read(self.path)
            if (len(events) - 1 != expected_parent_index or not events
                    or events[-1]["event_hash"] != expected_parent_hash):
                raise RegistryError("Cycle-3 registry head changed from its frozen parent")
            if any(event["event_type"] == "recovery_cycle_amendment_v2" for event in events):
                raise RegistryError("Cycle-3 amendment is already present")
            _validate_data("recovery_cycle_amendment_v2", amendment)
            named = amendment["recovery_registrations"]
            if [row["registration_kind"] for row in named] != ["benchmark", "train"]:
                raise RegistryError("Cycle-3 workload order must be benchmark then train")
            workload_by_id: dict[str, dict[str, Any]] = {}
            for workload in workloads:
                if (not isinstance(workload, dict)
                        or set(workload) != {"registration_id", "registration_kind", "record"}
                        or not isinstance(workload["registration_id"], str)
                        or not isinstance(workload["registration_kind"], str)
                        or not isinstance(workload["record"], dict)):
                    raise RegistryError("Cycle-3 workload candidate shape is invalid")
                if workload["registration_id"] in workload_by_id:
                    raise RegistryError("duplicate Cycle-3 workload registration_id")
                workload_by_id[workload["registration_id"]] = workload
            if set(workload_by_id) != {row["registration_id"] for row in named}:
                raise RegistryError("Cycle-3 workload candidates differ from authorized IDs")

            candidate_events = list(events)
            candidate_payloads: list[dict[str, Any]] = []
            prev = events[-1]["event_hash"]
            amendment_payload = _payload(_CYCLE3_EVENT_INDEX, "recovery_cycle_amendment_v2", amendment, prev)
            amendment_event = {**amendment_payload, "event_hash": _event_hash(amendment_payload)}
            candidate_events.append(amendment_event)
            candidate_payloads.append(amendment_event)
            for row in named:
                workload = workload_by_id[row["registration_id"]]
                if workload["registration_kind"] != row["registration_kind"]:
                    raise RegistryError("Cycle-3 workload kind differs from authorized binding")
                record = dict(workload["record"])
                if _CYCLE3_DERIVED_RECORD_FIELDS & record.keys():
                    raise RegistryError("Cycle-3 workload record must omit derived amendment pointers before binding")
                if cycle3_registration_record_sha256(record) != row["record_sha256"]:
                    raise RegistryError("Cycle-3 workload record differs from user-authorized binding")
                record["recovery_cycle_amendment_event_index"] = amendment_event["event_index"]
                record["recovery_cycle_amendment_event_hash"] = amendment_event["event_hash"]
                data = {"registration_id": row["registration_id"],
                        "registration_kind": row["registration_kind"],
                        "record": record, "cycle_id": "g008-recovery-cycle-3"}
                _validate_data("registration", data)
                prev = candidate_events[-1]["event_hash"]
                payload = _payload(len(candidate_events), "registration", data, prev)
                event = {**payload, "event_hash": _event_hash(payload)}
                candidate_events.append(event)
                candidate_payloads.append(event)
            _validate_recovery_cycle3_suffix(candidate_events)
            encoded = self.path.read_bytes() + b"".join(
                canonical_bytes(event) + b"\n" for event in candidate_payloads)
            original_stat = self.path.stat()
            temp_fd = -1
            temp_path: str | None = None
            try:
                temp_fd, temp_path = tempfile.mkstemp(
                    prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
                os.fchown(temp_fd, original_stat.st_uid, original_stat.st_gid)
                os.fchmod(temp_fd, stat.S_IMODE(original_stat.st_mode))
                offset = 0
                while offset < len(encoded):
                    written = os.write(temp_fd, encoded[offset:])
                    if written <= 0:
                        raise RegistryError("short atomic Cycle-3 temp write")
                    offset += written
                os.fsync(temp_fd)
                os.close(temp_fd)
                temp_fd = -1
                os.replace(temp_path, self.path)
                temp_path = None
                directory_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temp_fd >= 0:
                    os.close(temp_fd)
                if temp_path is not None:
                    try:
                        os.unlink(temp_path)
                    except FileNotFoundError:
                        pass
            _read(self.path)
            return candidate_payloads

    def register_recovery_cycle3_retry_batch(self, amendment: dict[str, Any],
                                             workload: dict[str, Any]) -> list[dict[str, Any]]:
        """Atomically append the sole authorized Cycle-3 retry amendment and registration."""
        if (not isinstance(workload, dict)
                or set(workload) != {"registration_id", "registration_kind", "record"}
                or workload.get("registration_kind") != "benchmark"
                or not isinstance(workload.get("record"), dict)):
            raise RegistryError("Cycle-3 retry workload candidate shape is invalid")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            events = _read(self.path)
            if (len(events) - 1 != _CYCLE3_RETRY_PARENT_EVENT_INDEX or not events
                    or events[-1]["event_hash"] != _CYCLE3_RETRY_PARENT_EVENT_HASH):
                raise RegistryError("Cycle-3 retry requires exact terminal event 33 registry head")
            if any(event["event_type"] == "recovery_cycle_retry_amendment_v1" for event in events):
                raise RegistryError("Cycle-3 benchmark retry is already registered")
            _validate_data("recovery_cycle_retry_amendment_v1", amendment)
            if (workload["registration_id"] != amendment["retry_registration_id"]
                    or workload["registration_kind"] != amendment["registration_kind"]
                    or cycle3_retry_registration_record_sha256(workload["record"])
                    != amendment["retry_record_sha256"]):
                raise RegistryError("Cycle-3 retry workload differs from authorized identity or record")
            if (amendment["prior_head_hash"] != events[-1]["event_hash"]
                    or amendment["cycle3_amendment_event_hash"] != events[27]["event_hash"]):
                raise RegistryError("Cycle-3 retry amendment changed immutable parent binding")
            candidate = list(events)
            prev = events[-1]["event_hash"]
            amendment_payload = _payload(34, "recovery_cycle_retry_amendment_v1", amendment, prev)
            amendment_event = {**amendment_payload, "event_hash": _event_hash(amendment_payload)}
            candidate.append(amendment_event)
            record = dict(workload["record"])
            if _CYCLE3_RETRY_RECORD_POINTER_FIELDS & record.keys():
                raise RegistryError("Cycle-3 retry record must omit derived event pointers before binding")
            record.update({
                "cycle3_retry_amendment_event_index": 34,
                "cycle3_retry_amendment_event_hash": amendment_event["event_hash"],
                "recovery_cycle_amendment_event_index": 27,
                "recovery_cycle_amendment_event_hash": events[27]["event_hash"],
            })
            registration_data = {
                "registration_id": workload["registration_id"],
                "registration_kind": workload["registration_kind"],
                "record": record,
                "cycle_id": "g008-recovery-cycle-3",
            }
            _validate_data("registration", registration_data)
            registration_payload = _payload(35, "registration", registration_data, amendment_event["event_hash"])
            registration_event = {**registration_payload, "event_hash": _event_hash(registration_payload)}
            candidate.append(registration_event)
            _validate_recovery_cycle3_suffix(candidate)
            encoded = self.path.read_bytes() + b"".join(
                canonical_bytes(event) + b"\n" for event in candidate[len(events):])
            original_stat = self.path.stat()
            temp_fd = -1
            temp_path: str | None = None
            try:
                temp_fd, temp_path = tempfile.mkstemp(
                    prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
                os.fchown(temp_fd, original_stat.st_uid, original_stat.st_gid)
                os.fchmod(temp_fd, stat.S_IMODE(original_stat.st_mode))
                offset = 0
                while offset < len(encoded):
                    written = os.write(temp_fd, encoded[offset:])
                    if written <= 0:
                        raise RegistryError("short atomic Cycle-3 retry temp write")
                    offset += written
                os.fsync(temp_fd)
                os.close(temp_fd)
                temp_fd = -1
                os.replace(temp_path, self.path)
                temp_path = None
                directory_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temp_fd >= 0:
                    os.close(temp_fd)
                if temp_path is not None:
                    try:
                        os.unlink(temp_path)
                    except FileNotFoundError:
                        pass
            _read(self.path)
            return [amendment_event, registration_event]

    def register_g008_competitive_campaign_batch(self, amendment: dict[str, Any]) -> list[dict[str, Any]]:
        """Atomically append a frozen campaign amendment followed by all seven registrations."""
        _validate_data("g008_competitive_campaign_amendment_v1", amendment)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            events = _read(self.path)
            if (not events or events[-1]["event_type"] != "benchmark_artifact_validation_rejected"
                    or amendment["prior_final_event_index"] != len(events) - 1
                    or amendment["prior_head_hash"] != events[-1]["event_hash"]):
                raise RegistryError("competitive campaign requires exact current rejected-artifact head")
            if any(e["event_type"] == "g008_competitive_campaign_amendment_v1" for e in events):
                raise RegistryError("competitive campaign amendment already exists")
            candidate = list(events)
            payloads = []
            amendment_payload = _payload(len(candidate), "g008_competitive_campaign_amendment_v1", amendment,
                                         events[-1]["event_hash"])
            amendment_payload["event_hash"] = _event_hash(amendment_payload)
            candidate.append(amendment_payload)
            payloads.append(amendment_payload)
            for row in amendment["records"]:
                record = dict(row["record"])
                record.update({"competitive_campaign_event_index": amendment_payload["event_index"],
                               "competitive_campaign_event_hash": amendment_payload["event_hash"]})
                data = {"registration_id": row["registration_id"], "registration_kind": row["registration_kind"],
                        "record": record, "cycle_id": "g008-competitive-campaign-v1"}
                _validate_data("registration", data)
                payload = _payload(len(candidate), "registration", data, candidate[-1]["event_hash"])
                payload["event_hash"] = _event_hash(payload)
                candidate.append(payload)
                payloads.append(payload)
            _validate_g008_competitive_campaign(candidate)
            encoded = self.path.read_bytes() + b"".join(canonical_bytes(event) + b"\n" for event in payloads)
            original_stat = self.path.stat()
            temp_fd = -1
            temp_path: str | None = None
            try:
                temp_fd, temp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp",
                                                     dir=self.path.parent)
                os.fchown(temp_fd, original_stat.st_uid, original_stat.st_gid)
                os.fchmod(temp_fd, stat.S_IMODE(original_stat.st_mode))
                offset = 0
                while offset < len(encoded):
                    written = os.write(temp_fd, encoded[offset:])
                    if written <= 0:
                        raise RegistryError("short atomic campaign temp write")
                    offset += written
                os.fsync(temp_fd)
                os.close(temp_fd)
                temp_fd = -1
                os.replace(temp_path, self.path)
                temp_path = None
                directory_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temp_fd >= 0:
                    os.close(temp_fd)
                if temp_path is not None:
                    try:
                        os.unlink(temp_path)
                    except FileNotFoundError:
                        pass
            _read(self.path)
            return payloads

    def revise_g008_competitive_campaign_v2_packages(self, revision: dict[str, Any]) -> dict[str, Any]:
        """Append a one-time, exact-head package correction after a registered train ERROR."""
        return self._append("g008_competitive_campaign_v2_package_revision_v1", revision)

    def authorize_g008_competitive_campaign_v2_retry_budget(self, authorization: dict[str, Any]) -> dict[str, Any]:
        """Append the single exact retry authorization after package revision event 73."""
        return self._append("g008_competitive_campaign_v2_retry_budget_v1", authorization)

    def revise_g008_competitive_campaign_v2_package_v2(self, revision: dict[str, Any]) -> dict[str, Any]:
        """Append the exact row-56 correction following the failed a2 retry."""
        return self._append("g008_competitive_campaign_v2_package_revision_v2", revision)

    def authorize_g008_competitive_campaign_v2_retry_budget_v2(
            self, authorization: dict[str, Any]) -> dict[str, Any]:
        """Append the one-shot a3 authorization following the second package correction."""
        return self._append("g008_competitive_campaign_v2_retry_budget_v2", authorization)

    def revise_g008_competitive_campaign_v2_packages_v3(self, revision: dict[str, Any]) -> dict[str, Any]:
        """Append the exact row-57 repair and bind corrected packages for remaining train rows."""
        return self._append("g008_competitive_campaign_v2_package_revision_v3", revision)

    def authorize_g008_competitive_campaign_v2_retry_budget_v3(
            self, authorization: dict[str, Any]) -> dict[str, Any]:
        """Append the one-shot row-57 retry while retaining the existing campaign caps."""
        return self._append("g008_competitive_campaign_v2_retry_budget_v3", authorization)

    def authorize_g008_competitive_campaign_v2_dispatch_retry(self, authorization: dict[str, Any]) -> dict[str, Any]:
        """Authorize the one fresh row-62 attempt after exact no-assignment reconciliation."""
        return self._append("g008_competitive_campaign_v2_dispatch_retry_v1", authorization)

    def revise_g008_competitive_campaign_v2_validation_packages(self, revision: dict[str, Any]) -> dict[str, Any]:
        """Bind one consistent corrected source package set for all six validations."""
        return self._append("g008_competitive_campaign_v2_package_revision_v4", revision)

    def authorize_g008_competitive_campaign_v2_dispatch_retry_v2(
            self, authorization: dict[str, Any]) -> dict[str, Any]:
        """Authorize one row-62 reattempt after the revised validation packages are frozen."""
        return self._append("g008_competitive_campaign_v2_dispatch_retry_v2", authorization)

    def authorize_g008_competitive_campaign_v2_dispatch_retry_v3(
            self, authorization: dict[str, Any]) -> dict[str, Any]:
        """Authorize one diagnostic row-62 attempt after an exact server HTTP 400 rejection."""
        return self._append("g008_competitive_campaign_v2_dispatch_retry_v3", authorization)

    def authorize_g008_competitive_campaign_v2_dispatch_retry_v4(
            self, authorization: dict[str, Any]) -> dict[str, Any]:
        """Authorize one row-62 attempt with bounded Kaggle response-body diagnostics."""
        return self._append("g008_competitive_campaign_v2_dispatch_retry_v4", authorization)

    def revise_g008_competitive_campaign_v2_validation_packages_v5(
            self, revision: dict[str, Any]) -> dict[str, Any]:
        """Bind sub-1-MiB validation packages after the exact source-size rejection."""
        return self._append("g008_competitive_campaign_v2_package_revision_v5", revision)

    def authorize_g008_competitive_campaign_v2_dispatch_retry_v5(
            self, authorization: dict[str, Any]) -> dict[str, Any]:
        """Authorize one row-62 submission of the corrected validation source."""
        return self._append("g008_competitive_campaign_v2_dispatch_retry_v5", authorization)

    def register_g008_competitive_campaign_v2_batch(self, amendment: dict[str, Any]) -> list[dict[str, Any]]:
        """Atomically freeze the six paired training jobs and two evaluations after Cycle 3."""
        _validate_data("g008_competitive_campaign_amendment_v2", amendment)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            events = _read(self.path)
            if (len(events) != 55 or events[-1]["event_index"] != 54
                    or events[-1]["event_type"] != "workload_attempt"
                    or amendment["prior_final_event_index"] != 54
                    or amendment["prior_head_hash"] != events[-1]["event_hash"]):
                raise RegistryError("campaign v2 requires exact current parent head at event 54")
            if any(e["event_type"] in {"g008_competitive_campaign_amendment_v1",
                                        "g008_competitive_campaign_amendment_v2"} for e in events):
                raise RegistryError("a competitive campaign amendment already exists")
            candidate = list(events)
            payloads = []
            amendment_payload = _payload(55, "g008_competitive_campaign_amendment_v2", amendment,
                                         events[-1]["event_hash"])
            amendment_payload["event_hash"] = _event_hash(amendment_payload)
            candidate.append(amendment_payload)
            payloads.append(amendment_payload)
            for row in amendment["records"]:
                record = dict(row["record"])
                record.update({"competitive_campaign_v2_event_index": amendment_payload["event_index"],
                               "competitive_campaign_v2_event_hash": amendment_payload["event_hash"]})
                data = {"registration_id": row["registration_id"],
                        "registration_kind": row["registration_kind"], "record": record,
                        "cycle_id": "g008-competitive-campaign-v2"}
                _validate_data("registration", data)
                payload = _payload(len(candidate), "registration", data, candidate[-1]["event_hash"])
                payload["event_hash"] = _event_hash(payload)
                candidate.append(payload)
                payloads.append(payload)
            _validate_g008_competitive_campaign_v2(candidate)
            encoded = self.path.read_bytes() + b"".join(canonical_bytes(event) + b"\n" for event in payloads)
            original_stat = self.path.stat()
            temp_fd = -1
            temp_path: str | None = None
            try:
                temp_fd, temp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp",
                                                     dir=self.path.parent)
                os.fchown(temp_fd, original_stat.st_uid, original_stat.st_gid)
                os.fchmod(temp_fd, stat.S_IMODE(original_stat.st_mode))
                offset = 0
                while offset < len(encoded):
                    written = os.write(temp_fd, encoded[offset:])
                    if written <= 0:
                        raise RegistryError("short atomic campaign v2 temp write")
                    offset += written
                os.fsync(temp_fd)
                os.close(temp_fd)
                temp_fd = -1
                os.replace(temp_path, self.path)
                temp_path = None
                directory_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temp_fd >= 0:
                    os.close(temp_fd)
                if temp_path is not None:
                    try:
                        os.unlink(temp_path)
                    except FileNotFoundError:
                        pass
            _read(self.path)
            return payloads

    def freeze_competitive_candidate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Append exact candidate/checkpoint identity after all paired training jobs finish."""
        return self._append("competitive_candidate_freeze", data)

    def freeze_competitive_candidate_v2(self, data: dict[str, Any]) -> dict[str, Any]:
        """Append a v2 candidate checkpoint only after all six training runs complete."""
        return self._append("competitive_candidate_freeze_v2", data)

    def record_competitive_training_artifact_v2(self, data: dict[str, Any]) -> dict[str, Any]:
        """Bind one verified checkpoint bundle to an exact COMPLETE v2 training attempt."""
        return self._append("competitive_training_artifact_v2", data)

    def record_competitive_validation_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """Append the reserved, non-final validation outcome for a frozen candidate."""
        return self._append("competitive_validation_result", data)

    def record_competitive_validation_result_v2(self, data: dict[str, Any]) -> dict[str, Any]:
        """Append the unique v2 validation result for the frozen candidate."""
        return self._append("competitive_validation_result_v2", data)

    def record_competitive_strategy_comparison_v2(self, created_at_utc: str) -> dict[str, Any]:
        """Compute the preregistered paired decision from all six v2 validation results."""
        data = _competitive_v2_comparison_payload(self.events(), created_at_utc=created_at_utc)
        return self._append("competitive_strategy_comparison_v2", data)

    def record_competitive_final_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """Append the final-only open-space result after a COMPLETE registered evaluation."""
        return self._append("competitive_final_result", data)

    def record_competitive_final_result_v2(self, data: dict[str, Any]) -> dict[str, Any]:
        """Append final open-space evidence only after the v2 validation passes."""
        return self._append("competitive_final_result_v2", data)

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

    def reject_benchmark_artifact_validation(self, attempt_event_index: int, attempt_event_hash: str, *,
                                             artifact_sha256: str | None,
                                             output_sha256: str | None,
                                             validation_codes: list[str], validator_id: str,
                                             created_at_utc: str) -> dict[str, Any]:
        """Append a structured rejection for an invalid COMPLETE Cycle-3 benchmark output."""
        events = self.events()
        if (not isinstance(attempt_event_index, int) or isinstance(attempt_event_index, bool)
                or attempt_event_index < 0 or attempt_event_index >= len(events)
                or events[attempt_event_index]["event_type"] != "workload_attempt"):
            raise RegistryError("artifact rejection requires an exact prior terminal attempt")
        attempt = events[attempt_event_index]["event_data"]
        matches = [event for event in events if event["event_type"] == "registration"
                   and event["event_index"] == attempt["registration_event_index"]]
        if (len(matches) != 1 or matches[0]["event_data"]["registration_kind"] != "benchmark"
                or attempt.get("outcome") != "terminal" or attempt.get("terminal_status") != "COMPLETE"):
            raise RegistryError("artifact rejection requires an exact COMPLETE registered benchmark")
        registered_cycle = matches[0]["event_data"].get("cycle_id")
        cycle_id = (registered_cycle if registered_cycle in {
            "g008-competitive-campaign-v1", "g008-competitive-campaign-v2"}
                    else "g008-recovery-cycle-3")
        data = {
            "schema_id": "g008-benchmark-artifact-validation-rejection", "schema_version": 1,
            "cycle_id": cycle_id, "created_at_utc": created_at_utc,
            "attempt_event_index": attempt_event_index, "attempt_event_hash": attempt_event_hash,
            "registration_event_index": matches[0]["event_index"], "registration_event_hash": matches[0]["event_hash"],
            "artifact_sha256": artifact_sha256, "output_sha256": output_sha256,
            "validator_id": validator_id, "validation_codes": validation_codes,
        }
        return self._append("benchmark_artifact_validation_rejected", data)

    def authorize_cycle3_retry_continuation(self, prior_attempt_id: str, *,
                                            created_at_utc: str) -> dict[str, Any]:
        """Append a head- and registration-bound authorization for one budgeted retry."""
        events = self.events()
        if len(events) < 36 or events[35]["event_type"] != "registration":
            raise RegistryError("Cycle-3 continuation requires the existing retry registration")
        terminal = next((event for event in events if event["event_type"] == "workload_attempt"
                         and event["event_data"].get("attempt_id") == prior_attempt_id), None)
        if terminal is None or terminal["event_data"].get("registration_event_index") != 35:
            raise RegistryError("Cycle-3 continuation requires the exact prior benchmark attempt")
        terminal_data = terminal["event_data"]
        status = terminal_data.get("terminal_status")
        rejection = next((event for event in events if event["event_type"] == "benchmark_artifact_validation_rejected"
                          and event["event_data"].get("attempt_event_index") == terminal["event_index"]), None)
        if status == "COMPLETE":
            if rejection is None or rejection["event_index"] != len(events) - 1:
                raise RegistryError("COMPLETE attempt needs explicit latest artifact-validation rejection")
        elif status not in {"ERROR", "CANCELLED", "TIMEOUT"} or rejection is not None:
            raise RegistryError("Cycle-3 continuation requires terminal failure or rejected COMPLETE output")
        head = events[-1]
        data = {
            "schema_id": "g008-cycle3-retry-policy-amendment", "schema_version": 1,
            "cycle_id": "g008-recovery-cycle-3-retry", "created_at_utc": created_at_utc,
            "policy_sha256": _CYCLE3_RETRY_POLICY_SHA256,
            "prior_head_hash": head["event_hash"], "prior_final_event_index": head["event_index"],
            "registration_event_index": 35, "registration_event_hash": events[35]["event_hash"],
            "registration_id": events[35]["event_data"]["registration_id"],
            "prior_attempt_id": prior_attempt_id, "prior_attempt_event_index": terminal["event_index"],
            "prior_attempt_event_hash": terminal["event_hash"], "prior_terminal_status": status,
            "validation_rejection_event_index": rejection["event_index"] if rejection else None,
            "validation_rejection_event_hash": rejection["event_hash"] if rejection else None,
        }
        return self._append("cycle3_retry_policy_amendment_v1", data)

    def record_exploratory_kernel_observation(self, *, version_number: int, run_uuid: str,
                                             terminal_status: str, artifact_sha256: dict[str, str],
                                             actual_gpu_seconds: int, output_bytes: int,
                                             created_at_utc: str) -> dict[str, Any]:
        """Record an already-existing, non-promotable Cycle-3 kernel version.

        The observation must immediately supersede the pending one-shot retry policy.
        It advances the factual Kaggle version cursor but creates no workload attempt,
        benchmark artifact, or authority to push another run.
        """
        events = self.events()
        if not events or events[-1]["event_type"] != "cycle3_retry_policy_amendment_v1":
            raise RegistryError("exploratory observation requires the exact pending retry continuation head")
        policy = events[-1]
        registration = events[policy["event_data"]["registration_event_index"]]
        data = {
            "schema_id": "g008-exploratory-kernel-observation", "schema_version": 1,
            "cycle_id": "g008-recovery-cycle-3-retry", "created_at_utc": created_at_utc,
            "kernel_id": registration["event_data"]["record"]["kernel_id"],
            "version_number": version_number, "run_uuid": run_uuid,
            "terminal_status": terminal_status, "exploratory": True, "promotable": False,
            "continuation_event_index": policy["event_index"],
            "continuation_event_hash": policy["event_hash"], "artifact_sha256": artifact_sha256,
            "actual_gpu_seconds": actual_gpu_seconds, "output_bytes": output_bytes,
        }
        return self._append("exploratory_kernel_observation_v1", data)

    def authorize_cycle3_retry_rebased(self, attempt_id: str, *, created_at_utc: str) -> dict[str, Any]:
        """Authorize exactly one retry from the observed kernel cursor, with a fresh attempt ID."""
        events = self.events()
        if not events or events[-1]["event_type"] != "exploratory_kernel_observation_v1":
            raise RegistryError("Cycle-3 retry rebase requires the exact exploratory-observation head")
        observation = events[-1]
        policy = events[observation["event_data"]["continuation_event_index"]]
        registration = events[policy["event_data"]["registration_event_index"]]
        record = registration["event_data"]["record"]
        prior_head = events[-1]
        if (not isinstance(attempt_id, str) or not attempt_id
                or any(event["event_type"] == "workload_preflight"
                       and event["event_data"].get("attempt_id") == attempt_id for event in events)):
            raise RegistryError("rebased retry requires a fresh non-empty attempt ID")
        data = {
            "schema_id": "g008-cycle3-retry-rebased-authorization", "schema_version": 1,
            "cycle_id": "g008-recovery-cycle-3-retry", "created_at_utc": created_at_utc,
            "prior_head_hash": prior_head["event_hash"], "prior_final_event_index": prior_head["event_index"],
            "observation_event_index": observation["event_index"],
            "observation_event_hash": observation["event_hash"],
            "registration_event_index": registration["event_index"],
            "registration_event_hash": registration["event_hash"],
            "registration_id": registration["event_data"]["registration_id"],
            "kernel_id": observation["event_data"]["kernel_id"],
            "expected_current_version": observation["event_data"]["version_number"],
            "observed_run_uuid": observation["event_data"]["run_uuid"],
            "observed_terminal_status": observation["event_data"]["terminal_status"],
            "attempt_id": attempt_id,
            "session_timeout_seconds": record["session_timeout_seconds"],
            "reserved_gpu_seconds": record["session_timeout_seconds"],
        }
        for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                    "backend_sha256", "trainer_sha256", "engine_sha256"):
            data[key] = record[key]
        return self._append("cycle3_retry_rebased_authorization_v1", data)

    def observed_kernel_cursor(self, kernel_id: str) -> dict[str, Any] | None:
        """Return the latest hash-chained exploratory observation for a kernel, if any."""
        for event in reversed(self.events()):
            if (event["event_type"] == "exploratory_kernel_observation_v1"
                    and event["event_data"]["kernel_id"] == kernel_id):
                data = event["event_data"]
                return {
                    "kernel_id": kernel_id,
                    "current_version_number": data["version_number"],
                    "observation_event_index": event["event_index"],
                    "observation_event_hash": event["event_hash"],
                    "run_uuid": data["run_uuid"],
                    "promotable": False,
                }
        return None

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
