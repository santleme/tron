"""Fail-closed validation for frozen G008 split manifests.

This module is deliberately separate from ``lc-eval/1``: historical G007
reports retain their original fixed contract and are not interpreted as split
manifests.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "lc-split-manifest-3.schema.json"
_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = Draft202012Validator(_SCHEMA)
SEED_PLAN_PATH = Path(__file__).resolve().parents[1] / "configs" / "g008-split-seed-plan.json"
_SPLITS = ("train", "imitation_demo", "benchmark", "validation", "final")
_HASH_FIELDS = (
    "opponent_sha256",
    "opponent_code_sha256",
    "opponent_config_sha256",
    "opponent_model_sha256",
)
_ALL_PROVENANCE_HASH_FIELDS = _HASH_FIELDS + (
    "target_checkpoint_sha256",
    "training_config_sha256",
    "source_data_sha256",
)
_SPLIT_USAGES = {
    "train": {"train", "reward_design"},
    "imitation_demo": {"imitation"},
    "benchmark": {"benchmark"},
    "validation": {"validation", "checkpoint_selection"},
    "final": {"final_eval"},
}
# Locked in training/configs/g008-split-seed-plan.json. These component
# fingerprints identify the protected open-space policy even if a row renames
# the opponent ID. Re-freeze this lock if the policy implementation changes.
OPEN_SPACE_CODE_SHA256 = "9e5103176218ca092206ba50899a35c43ad488ce825a95af487e76798ce60123"
OPEN_SPACE_CONFIG_SHA256 = "4043dda35bb8cc2adc43112cb5b00eff50ac721eb062b15d0d83b7ae41510b7f"
FROZEN_SEED_PLAN_SHA256 = "0173875c2f13fc3ffe5dbae76c2b6d6e46bc8398da7f8a748988a51b345d4b24"


class ProvenanceError(ValueError):
    """Raised when a split manifest is incomplete, inconsistent, or leaky."""


SEED_CODEC_PREFIX = b"CIRCUIT-AI/G008/ENV-SEED/v1\0"
SEED_TUPLE_FIELDS = ("split", "domain_label", "env_root_seed", "run_key",
                     "update_index", "rollout_index", "arena_index", "game_index", "reset_index")
SEED_DOMAIN_LABELS = (
    "g008/train/initial", "g008/train/reset",
    "g008/benchmark/measured/initial", "g008/benchmark/measured/reset",
    "g008/benchmark/warmup/initial", "g008/benchmark/warmup/reset",
    "g008/eval/validation/initial", "g008/eval/validation/reset",
    "g008/eval/final/initial", "g008/eval/final/reset",
)
UINT64_MAX = (1 << 64) - 1
_SCHEDULE_SUITES = ("train", "imitation_demo", "benchmark", "validation", "final")
_SLOT_KEYS = frozenset(("slot_id", "suite_ordinal", "root_ordinal", *SEED_TUPLE_FIELDS,
                        "derived_seed_uint64", "status", "protocol_exclusion_reason"))
_OUTCOME_KEYS = frozenset(("initial_slot_id", "seat0_agent_id", "seat1_agent_id", "result",
                           "terminal_reason", "ticks_completed"))


def _canonical_json(value: Any) -> bytes:
    """Canonical UTF-8 JSON used by all G008 digest contracts."""
    def normalize(item: Any) -> Any:
        if isinstance(item, str):
            return unicodedata.normalize("NFC", item)
        if isinstance(item, list):
            return [normalize(value) for value in item]
        if isinstance(item, tuple):
            return [normalize(value) for value in item]
        if isinstance(item, dict):
            return {normalize(key): normalize(value) for key, value in item.items()}
        if isinstance(item, float):
            raise ProvenanceError("canonical G008 JSON forbids floats")
        return item
    try:
        return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"value is not canonical-JSON serializable: {exc}") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode_optional_uint64(value: int | None) -> bytes:
    """Encode optional uint64; N/A has the frozen all-ones representation."""
    if value is None:
        return b"\xff" * 8
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT64_MAX:
        raise ProvenanceError("optional integer must be uint64 or None")
    return value.to_bytes(8, "big")


def _encode_string(value: str) -> bytes:
    if not isinstance(value, str):
        raise ProvenanceError("seed codec string fields must be strings")
    encoded = unicodedata.normalize("NFC", value).encode("utf-8")
    if len(encoded) > 0xFFFFFFFF:
        raise ProvenanceError("seed codec string exceeds uint32 framing")
    return len(encoded).to_bytes(4, "big") + encoded


def _encode_required_uint64(value: Any, field: str) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT64_MAX:
        raise ProvenanceError(f"{field} must be a required uint64 (N/A is forbidden)")
    return value.to_bytes(8, "big")


def _validate_tuple_protocol(seed_tuple: Mapping[str, Any]) -> None:
    split = seed_tuple["split"]
    domain = seed_tuple["domain_label"]
    root = seed_tuple["env_root_seed"]
    run_key = seed_tuple["run_key"]
    domain_ok = {
        "train": domain.startswith("g008/train/"),
        "benchmark": domain.startswith("g008/benchmark/"),
        "validation": domain.startswith("g008/eval/validation/"),
        "final": domain.startswith("g008/eval/final/"),
    }
    if split == "imitation_demo":
        raise ProvenanceError("G008 plan needs explicit imitation_demo environment domain labels")
    if not domain_ok[split]:
        raise ProvenanceError("seed tuple domain label does not match its split")
    if split == "train" and not re.fullmatch(r"train-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", run_key):
        raise ProvenanceError("train run_key must contain a lowercase canonical UUID")
    if split == "validation" and run_key != "validation-suite-v1":
        raise ProvenanceError("validation run_key must be validation-suite-v1")
    if split == "final" and run_key != "final-suite-v1":
        raise ProvenanceError("final run_key must be final-suite-v1")
    if split != "benchmark":
        plan, _ = _read_seed_plan()
        lower, upper = plan["partitions"][split]["root_seed_range_inclusive"]
        if not lower <= root <= upper:
            raise ProvenanceError(f"environment root is outside frozen {split} partition")
        return
    plan, _ = _read_seed_plan()
    lower, upper = plan["partitions"]["benchmark"]["root_seed_range_inclusive"]
    if not lower <= root <= upper:
        raise ProvenanceError("environment root is outside frozen benchmark partition")
    if domain.startswith("g008/benchmark/measured/"):
        repeat = root - 5_000_000
        if not 0 <= repeat <= 4 or run_key != f"pair-{repeat:02d}":
            raise ProvenanceError("measured benchmark root/run_key assignment is not frozen")
    else:
        match = re.fullmatch(r"warmup-(python-oracle|low-risk|numpy-control|numba-candidate)-(incumbent|candidate)-(00|01)", run_key)
        if not match or not 5_000_256 <= root <= 5_000_271:
            raise ProvenanceError("warmup benchmark root/run_key assignment is not frozen")
        _, role, rollout_text = match.groups()
        # The backend token is part of the run key, so match its exact role's
        # frozen assignment rather than accepting a root by range alone.
        backend = match.group(1)
        roots_for_role = next((entry["root_seeds"] for entry in plan["partitions"]["benchmark"]["warmup_assignments"]
                               if entry["backend_id"] == backend and entry["role"] == role), None)
        rollout = int(rollout_text)
        if roots_for_role is None or roots_for_role[rollout] != root or seed_tuple["rollout_index"] != rollout:
            raise ProvenanceError("warmup root, run_key, and rollout index assignment mismatch")


def encode_seed_tuple(seed_tuple: Mapping[str, Any]) -> bytes:
    """Serialize the exact nine-field G008 environment seed tuple."""
    if not isinstance(seed_tuple, Mapping) or set(seed_tuple) != set(SEED_TUPLE_FIELDS):
        raise ProvenanceError("seed tuple must contain exactly the nine codec fields")
    split, domain, root, run_key, *indices = (seed_tuple[name] for name in SEED_TUPLE_FIELDS)
    if not isinstance(split, str) or split not in _SCHEDULE_SUITES:
        raise ProvenanceError("seed tuple has unknown split")
    if not isinstance(domain, str) or domain not in SEED_DOMAIN_LABELS:
        raise ProvenanceError("seed tuple has unknown domain label")
    if not isinstance(run_key, str):
        raise ProvenanceError("run_key must be a string")
    _encode_required_uint64(root, "env_root_seed")
    for name, value in zip(SEED_TUPLE_FIELDS[4:], indices):
        _encode_required_uint64(value, name)
    _validate_tuple_protocol(seed_tuple)
    payload = bytearray(SEED_CODEC_PREFIX)
    payload.extend(_encode_string(split))
    payload.extend(_encode_string(domain))
    payload.extend(_encode_required_uint64(root, "env_root_seed"))
    payload.extend(_encode_string(run_key))
    for name, value in zip(SEED_TUPLE_FIELDS[4:], indices):
        payload.extend(_encode_required_uint64(value, name))
    return bytes(payload)


def derive_environment_seed(seed_tuple: Mapping[str, Any]) -> tuple[str, int, bytes]:
    payload = encode_seed_tuple(seed_tuple)
    digest = hashlib.sha256(payload).digest()
    return digest.hex(), int.from_bytes(digest[:8], "big"), payload


def schedule_spec_payload(seed_plan_sha: str | None = None) -> dict[str, Any]:
    """Return the Phase-A-frozen global schedule payload for its digest."""
    plan, actual_sha = _read_seed_plan()
    if seed_plan_sha is not None and seed_plan_sha != actual_sha:
        raise ProvenanceError("stale seed-plan binding while building schedule spec")
    if plan.get("format") != "circuit-ai-g008-split-seed-plan/3" or actual_sha != FROZEN_SEED_PLAN_SHA256:
        raise ProvenanceError("G008 requires the exact approved seed-plan /3 bytes")
    return {
        "format": plan["format"],
        "seed_plan_sha256": actual_sha,
        "suite_order": plan["schedule"]["suite_order"],
        "allowed_duplicate": "Only identical environment tuple reused by incumbent and candidate sides of same measured pair; all other derived-seed collisions fail closed.",
        "codec": plan["environment_seed_codec"],
        "partitions": plan["partitions"],
        "benchmark_schedule": plan["partitions"]["benchmark"],
        "seed_semantics": plan["seed_semantics"],
        "evaluation_opponents": plan["frozen_evaluation_opponents"],
        "schedule_digest_contract": plan["schedule_digest_contract"],
        "registry_lock_lifecycle": plan["registry_lock_lifecycle"],
    }


def schedule_spec_digest(seed_plan_sha: str | None = None) -> str:
    return _sha256(_canonical_json(schedule_spec_payload(seed_plan_sha)))


def _tuple_from_slot(slot: Mapping[str, Any]) -> dict[str, Any]:
    return {name: slot[name] for name in SEED_TUPLE_FIELDS}


def _slot_id(suite_id: str, run_id: str, backend_id: str, root_ordinal: int,
             seed_tuple: Mapping[str, Any]) -> str:
    identity = {"suite_id": suite_id, "run_id": run_id, "backend_id": backend_id,
                "root_ordinal": root_ordinal, **seed_tuple}
    return _sha256(_canonical_json(identity))


def collision_scan(slots: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Scan the complete reserved roster; only paired identical tuple reuse passes."""
    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    tuple_count = 0
    for slot in slots:
        tuple_count += 1
        digest, derived, _ = derive_environment_seed(_tuple_from_slot(slot))
        if slot.get("derived_seed_uint64") != derived:
            raise ProvenanceError("slot derived seed does not match its canonical tuple")
        by_seed.setdefault(derived, []).append(slot)
    duplicates = []
    passed = True
    for derived in sorted(key for key, members in by_seed.items() if len(members) > 1):
        members = sorted(by_seed[derived], key=lambda slot: slot["slot_id"].encode("utf-8"))
        tuples = [_tuple_from_slot(slot) for slot in members]
        backends = {slot.get("backend_id") for slot in members}
        legal_pair = (len(members) == 2 and tuples[0] == tuples[1]
                      and backends == {"incumbent", "candidate"}
                      and tuples[0]["split"] == "benchmark"
                      and tuples[0]["domain_label"].startswith("g008/benchmark/measured/")
                      and members[0].get("run_key", members[0]["run_key"]) == members[1].get("run_key", members[1]["run_key"]))
        if not legal_pair:
            passed = False
        duplicates.append({"derived_seed_uint64": derived, "members": [
            {"slot_id": slot["slot_id"], "backend_id": slot.get("backend_id", "shared"),
             "tuple": _tuple_from_slot(slot)} for slot in members]})
    return {"algorithm": "sha256-derived-seed-collision-scan/v1", "total_slots": len(slots),
            "total_tuples": tuple_count, "unique_seed_count": len(by_seed),
            "duplicate_groups": duplicates, "passed": passed}


def build_expanded_schedule(*, suite_id: str, run_id: str, run_key: str,
                            slots: list[Mapping[str, Any]], seed_plan_sha: str | None = None) -> dict[str, Any]:
    """Canonicalize a preregistered full reservation roster and digest it."""
    if not all(isinstance(value, str) and value for value in (suite_id, run_id, run_key)):
        raise ProvenanceError("suite_id, run_id, and run_key must be nonempty strings")
    suite_id, run_id, run_key = (unicodedata.normalize("NFC", value)
                                 for value in (suite_id, run_id, run_key))
    spec_digest = schedule_spec_digest(seed_plan_sha)
    normalized_slots = []
    for source in slots:
        slot = dict(source)
        extra = set(slot) - (_SLOT_KEYS | {"backend_id"})
        if extra:
            raise ProvenanceError(f"expanded slot has unexpected fields: {sorted(extra)}")
        backend_id = slot.pop("backend_id", "shared")
        if backend_id not in {"incumbent", "candidate", "shared"}:
            raise ProvenanceError("backend_id must be incumbent, candidate, or shared")
        seed_tuple = _tuple_from_slot(slot)
        seed_tuple = {key: unicodedata.normalize("NFC", value) if isinstance(value, str) else value
                      for key, value in seed_tuple.items()}
        digest, derived, _ = derive_environment_seed(seed_tuple)
        if seed_tuple["run_key"] != run_key:
            raise ProvenanceError("slot run_key differs from schedule run_key")
        status = slot.get("status")
        reason = slot.get("protocol_exclusion_reason")
        if status not in {"reserved", "protocol-excluded"}:
            raise ProvenanceError("slot status must be preregistered reserved or protocol-excluded")
        if not isinstance(reason, str) or (status == "protocol-excluded") != bool(reason):
            raise ProvenanceError("protocol exclusion reason must be present only for excluded slots")
        for index_name in ("suite_ordinal", "root_ordinal"):
            _encode_required_uint64(slot.get(index_name), index_name)
        normalized = {"slot_id": _slot_id(suite_id, run_id, backend_id, slot["root_ordinal"], seed_tuple),
                      "suite_ordinal": slot["suite_ordinal"], "root_ordinal": slot["root_ordinal"],
                      **seed_tuple, "derived_seed_uint64": derived, "status": status,
                      "protocol_exclusion_reason": unicodedata.normalize("NFC", reason)}
        normalized["backend_id"] = backend_id
        normalized_slots.append(normalized)
    order = lambda slot: tuple(slot[name] for name in ("suite_ordinal", "root_ordinal", "arena_index",
        "update_index", "rollout_index", "game_index", "reset_index")) + (slot["slot_id"],)
    normalized_slots.sort(key=order)
    if len({slot["slot_id"] for slot in normalized_slots}) != len(normalized_slots):
        raise ProvenanceError("slot_id collision in expanded schedule")
    scan = collision_scan(normalized_slots)
    body_slots = [{key: value for key, value in slot.items() if key != "backend_id"} for slot in normalized_slots]
    body = {"schedule_spec_digest": spec_digest, "suite_id": suite_id, "run_id": run_id,
            "run_key": run_key, "slots": body_slots, "collision_scan": scan}
    return {**body, "expanded_schedule_digest": _sha256(_canonical_json(body))}


def build_training_run_schedule(*, run_id: str, root_seed: int, arena_count: int,
                                total_updates: int, rollout_steps: int,
                                seed_plan_sha: str | None = None) -> dict[str, Any]:
    """Freeze every initial and maximum per-update reset slot for one G008 train run."""
    if not isinstance(run_id, str) or not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", run_id):
        raise ProvenanceError("G008 training run_id must be a canonical lowercase RFC 4122 UUID")
    for name, value in (("arena_count", arena_count), ("total_updates", total_updates),
                        ("rollout_steps", rollout_steps)):
        if type(value) is not int or value < 1:
            raise ProvenanceError(f"{name} must be a positive integer")
    if arena_count * total_updates * (rollout_steps + 1) > 1_000_000:
        raise ProvenanceError("G008 training schedule exceeds the one-million-slot preparation cap")
    plan, actual_plan_sha = _read_seed_plan()
    if seed_plan_sha is not None and seed_plan_sha != actual_plan_sha:
        raise ProvenanceError("G008 training schedule has a stale seed-plan hash")
    lower, upper = plan["partitions"]["train"]["root_seed_range_inclusive"]
    if type(root_seed) is not int or not lower <= root_seed <= upper:
        raise ProvenanceError("G008 training root_seed is outside the frozen train partition")
    run_key = f"train-{run_id}"
    slots = []
    for update in range(total_updates):
        for arena in range(arena_count):
            for reset_index in range(rollout_steps + 1):
                slots.append({
                    "split": "train",
                    "domain_label": "g008/train/initial" if reset_index == 0 else "g008/train/reset",
                    "env_root_seed": root_seed,
                    "run_key": run_key,
                    "update_index": update,
                    "rollout_index": 0,
                    "arena_index": arena,
                    "game_index": 0,
                    "reset_index": reset_index,
                    "suite_ordinal": 0,
                    "root_ordinal": 0,
                    "status": "reserved",
                    "protocol_exclusion_reason": "",
                })
    return build_expanded_schedule(suite_id="g008-train", run_id=run_id,
                                   run_key=run_key, slots=slots,
                                   seed_plan_sha=actual_plan_sha)


def benchmark_run_keys() -> tuple[str, ...]:
    """Return benchmark run_key order exactly as measured roots then warmup roots."""
    plan, _ = _read_seed_plan()
    if plan.get("format") != "circuit-ai-g008-split-seed-plan/3":
        raise ProvenanceError("G008 requires the approved seed-plan /3 before benchmark schedule expansion")
    benchmark = plan["partitions"]["benchmark"]
    keys = [f"pair-{entry['repeat_id']:02d}" for entry in benchmark["measured_roots"]]
    for assignment in benchmark["warmup_assignments"]:
        keys.extend(f"warmup-{assignment['backend_id']}-{assignment['role']}-{index:02d}"
                    for index, _ in enumerate(assignment["root_seeds"]))
    return tuple(keys)


def build_benchmark_reservation_roster(*, suite_id: str,
                                      run_ids: Mapping[str, str]) -> list[dict[str, Any]]:
    """Materialize the frozen G008 benchmark upper-bound roster (all reset slots)."""
    plan, _ = _read_seed_plan()
    if plan.get("format") != "circuit-ai-g008-split-seed-plan/3":
        raise ProvenanceError("G008 requires the approved seed-plan /3 before benchmark schedule expansion")
    expected_keys = benchmark_run_keys()
    if not suite_id or not isinstance(run_ids, Mapping) or tuple(run_ids) != expected_keys:
        raise ProvenanceError("benchmark roster requires run IDs in the exact frozen run-key order")
    if any(not isinstance(run_ids[key], str) or not run_ids[key] for key in expected_keys):
        raise ProvenanceError("benchmark run IDs must be nonempty strings")
    if len(set(run_ids.values())) != len(run_ids):
        raise ProvenanceError("each benchmark run_key requires a unique run_id")
    benchmark = plan["partitions"]["benchmark"]
    roster: list[dict[str, Any]] = []

    def append_root(root: int, root_ordinal: int, run_key: str, initial_domain: str,
                    reset_domain: str, update_count: int, backend_ids: tuple[str, ...],
                    rollout_index: int = 0) -> None:
        for arena in range(128):
            entries = [(0, rollout_index, 0, 0, initial_domain)]
            if update_count == 3:
                entries.extend(((g - 1) // 32, rollout_index, g, ((g - 1) % 32) + 1, reset_domain)
                               for g in range(1, 97))
            else:
                entries.extend((0, rollout_index, g, g, reset_domain) for g in range(1, 33))
            for update, rollout, game, reset, domain in entries:
                seed_tuple = {"split": "benchmark", "domain_label": domain,
                              "env_root_seed": root, "run_key": run_key,
                              "update_index": update, "rollout_index": rollout,
                              "arena_index": arena, "game_index": game, "reset_index": reset}
                _, derived, _ = derive_environment_seed(seed_tuple)
                for backend_id in backend_ids:
                    roster.append({"suite_ordinal": 2, "root_ordinal": root_ordinal,
                                   **seed_tuple, "derived_seed_uint64": derived,
                                   "status": "reserved", "protocol_exclusion_reason": "",
                                   "slot_id": _slot_id(suite_id, run_ids[run_key], backend_id,
                                                       root_ordinal, seed_tuple),
                                   "backend_id": backend_id})

    measured = benchmark["measured_roots"]
    for item in measured:
        repeat_id = item["repeat_id"]
        for backend_id in ("incumbent", "candidate"):
            append_root(item["root_seed"], repeat_id, f"pair-{repeat_id:02d}",
                        "g008/benchmark/measured/initial", "g008/benchmark/measured/reset",
                        3, (backend_id,))
    # Warm-up assignments are already in the frozen plan's root order. A
    # rollout's index is encoded in the run key and tuple; never reuse a root.
    for assignment in benchmark["warmup_assignments"]:
        for rollout_index, root in enumerate(assignment["root_seeds"]):
            warmup_key = f"warmup-{assignment['backend_id']}-{assignment['role']}-{rollout_index:02d}"
            append_root(root, root - 5_000_000, warmup_key,
                        "g008/benchmark/warmup/initial", "g008/benchmark/warmup/reset",
                        1, ("shared",), rollout_index)
    return roster


def build_benchmark_run_schedules(*, suite_id: str,
                                  run_ids: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Expand one canonical schedule per frozen benchmark run_key/run_id."""
    roster = build_benchmark_reservation_roster(suite_id=suite_id, run_ids=run_ids)
    slots_by_key: dict[str, list[dict[str, Any]]] = {key: [] for key in benchmark_run_keys()}
    for slot in roster:
        slots_by_key[slot["run_key"]].append(slot)
    schedules = {}
    for run_key in benchmark_run_keys():
        schedules[run_key] = build_expanded_schedule(
            suite_id=suite_id, run_id=run_ids[run_key], run_key=run_key,
            slots=slots_by_key[run_key])
    return schedules


def build_schedule_aggregate(*, expanded_schedules: Mapping[str, Mapping[str, Any]],
                              realized_schedules: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Bind an ordered set of run IDs to common spec and expanded/realized digests."""
    run_ids = list(expanded_schedules)
    if not run_ids or run_ids != list(realized_schedules):
        raise ProvenanceError("aggregate run-ID order/membership differs between expanded and realized inputs")
    spec_digest = schedule_spec_digest()
    runs = []
    for run_id in run_ids:
        expanded = expanded_schedules[run_id]
        realized = realized_schedules[run_id]
        if expanded.get("run_id") != run_id:
            raise ProvenanceError("aggregate mapping key differs from expanded run_id")
        binding = {"seed_plan_sha256": seed_plan_sha256(),
                   "schedule_spec_digest": spec_digest,
                   "expanded_schedule_digest": expanded.get("expanded_schedule_digest"),
                   "realized_schedule_digest": realized.get("realized_schedule_digest")}
        validate_schedule_binding(binding, expanded=expanded, realized=realized)
        runs.append({"run_id": run_id,
                     "expanded_schedule_digest": binding["expanded_schedule_digest"],
                     "realized_schedule_digest": binding["realized_schedule_digest"]})
    return {"schedule_spec_digest": spec_digest, "runs": runs}


def validate_expanded_schedule(expanded: Mapping[str, Any], seed_plan_sha: str | None = None) -> None:
    """Recompute expanded digest, tuple encodings, slot IDs, order, and full collision scan."""
    required = {"schedule_spec_digest", "suite_id", "run_id", "run_key", "slots",
                "collision_scan", "expanded_schedule_digest"}
    if not isinstance(expanded, Mapping) or set(expanded) != required:
        raise ProvenanceError("expanded schedule is missing or has unexpected top-level fields")
    for field in ("suite_id", "run_id", "run_key"):
        if not isinstance(expanded[field], str) or expanded[field] != unicodedata.normalize("NFC", expanded[field]):
            raise ProvenanceError(f"expanded schedule {field} must be NFC UTF-8")
    spec = schedule_spec_digest(seed_plan_sha)
    if expanded["schedule_spec_digest"] != spec:
        raise ProvenanceError("expanded schedule has missing or stale schedule-spec binding")
    body = {key: expanded[key] for key in required if key != "expanded_schedule_digest"}
    if _sha256(_canonical_json(body)) != expanded["expanded_schedule_digest"]:
        raise ProvenanceError("expanded schedule digest is stale or forged")
    if not isinstance(expanded["slots"], list):
        raise ProvenanceError("expanded schedule slots must be an array")
    duplicate_members: dict[str, str] = {}
    scan = expanded["collision_scan"]
    if not isinstance(scan, Mapping) or not scan.get("passed"):
        raise ProvenanceError("expanded schedule collision scan failed")
    for group in scan.get("duplicate_groups", []):
        for member in group.get("members", []):
            slot_id = member.get("slot_id")
            backend = member.get("backend_id")
            if slot_id in duplicate_members or backend not in {"incumbent", "candidate"}:
                raise ProvenanceError("collision scan has malformed/duplicate paired membership")
            duplicate_members[slot_id] = backend
    reconstructed = []
    seen_ids = set()
    for slot in expanded["slots"]:
        if not isinstance(slot, Mapping) or set(slot) != _SLOT_KEYS:
            raise ProvenanceError("expanded slot has unexpected or missing fields")
        seed_tuple = _tuple_from_slot(slot)
        if any(isinstance(value, str) and value != unicodedata.normalize("NFC", value)
               for value in seed_tuple.values()):
            raise ProvenanceError("expanded tuple strings must be NFC-normalized")
        _, derived, _ = derive_environment_seed(seed_tuple)
        if slot["derived_seed_uint64"] != derived:
            raise ProvenanceError("expanded slot derived seed does not match tuple")
        backend = duplicate_members.get(slot["slot_id"], "shared")
        expected_id = _slot_id(expanded["suite_id"], expanded["run_id"], backend,
                               slot["root_ordinal"], seed_tuple)
        if slot["slot_id"] != expected_id or slot["slot_id"] in seen_ids:
            raise ProvenanceError("expanded slot_id is invalid or duplicated")
        seen_ids.add(slot["slot_id"])
        reconstructed.append({**slot, "backend_id": backend})
    ordering = lambda slot: tuple(slot[name] for name in ("suite_ordinal", "root_ordinal", "arena_index",
        "update_index", "rollout_index", "game_index", "reset_index")) + (slot["slot_id"],)
    if reconstructed != sorted(reconstructed, key=ordering):
        raise ProvenanceError("expanded slots are not in frozen canonical order")
    recomputed_scan = collision_scan(reconstructed)
    if recomputed_scan != scan:
        raise ProvenanceError("collision scan is incomplete, stale, or malformed")


def validate_realization(expanded: Mapping[str, Any], *, realized_slots: list[Mapping[str, Any]],
                         realized_tuple_list: list[Mapping[str, Any]],
                         episode_outcomes: list[Mapping[str, Any]], completed_run: bool = True) -> dict[str, Any]:
    """Validate one-to-one runtime realization and return canonical digest payload."""
    validate_expanded_schedule(expanded)
    expected_digest = expanded["expanded_schedule_digest"]
    frozen_slots = expanded["slots"]
    if len(realized_slots) != len(frozen_slots):
        raise ProvenanceError("realization must classify every reserved slot exactly once")
    valid_statuses = {"consumed", "terminal-unreachable", "protocol-excluded", "run-aborted-before-reach"}
    by_id = {slot["slot_id"]: slot for slot in frozen_slots}
    seen = set()
    normalized_realized = []
    for record in realized_slots:
        if set(record) != {"slot_id", "status"} or record["slot_id"] not in by_id or record["slot_id"] in seen:
            raise ProvenanceError("realized slot is malformed, unknown, or duplicated")
        if record["status"] not in valid_statuses:
            raise ProvenanceError("unknown realized slot status")
        if completed_run and record["status"] == "run-aborted-before-reach":
            raise ProvenanceError("completed run cannot contain aborted-before-reach slots")
        if (record["status"] == "protocol-excluded") != (by_id[record["slot_id"]]["status"] == "protocol-excluded"):
            raise ProvenanceError("realized exclusion differs from preregistered status")
        seen.add(record["slot_id"])
        normalized_realized.append({"slot_id": record["slot_id"], "status": record["status"]})
    status_by_id = {entry["slot_id"]: entry["status"] for entry in normalized_realized}
    normalized_realized = [{"slot_id": slot["slot_id"], "status": status_by_id[slot["slot_id"]]}
                          for slot in frozen_slots]
    consumed_tuples = []
    for item in realized_tuple_list:
        if set(item) != set(SEED_TUPLE_FIELDS):
            raise ProvenanceError("realized tuple has unexpected or missing fields")
        encode_seed_tuple(item)
        consumed_tuples.append({name: unicodedata.normalize("NFC", item[name]) if isinstance(item[name], str)
                                else item[name] for name in SEED_TUPLE_FIELDS})
    status_by_id = {entry["slot_id"]: entry["status"] for entry in normalized_realized}
    expected_consumed = [_tuple_from_slot(slot) for slot in frozen_slots
                         if status_by_id[slot["slot_id"]] == "consumed"]
    if consumed_tuples != expected_consumed:
        raise ProvenanceError("realized tuple list does not match consumed slots in frozen order")
    outcomes = []
    initial_ids = set()
    for outcome in episode_outcomes:
        if set(outcome) != _OUTCOME_KEYS:
            raise ProvenanceError("episode outcome has unexpected or missing fields")
        item = dict(outcome)
        for key in ("initial_slot_id", "seat0_agent_id", "seat1_agent_id"):
            if not isinstance(item[key], str) or not item[key]:
                raise ProvenanceError(f"outcome {key} must be a nonempty string")
            item[key] = unicodedata.normalize("NFC", item[key])
        slot = by_id.get(item["initial_slot_id"])
        if slot is None or item["initial_slot_id"] in initial_ids:
            raise ProvenanceError("outcome references an unknown or duplicate initial slot")
        if slot["game_index"] != 0 or slot["reset_index"] != 0:
            raise ProvenanceError("outcome initial_slot_id must reference an initial game slot")
        if status_by_id[item["initial_slot_id"]] != "consumed":
            raise ProvenanceError("outcome initial slot was not consumed")
        initial_ids.add(item["initial_slot_id"])
        if item["result"] not in {"seat0_win", "seat1_win", "draw", "invalid"}:
            raise ProvenanceError("invalid outcome result")
        if item["terminal_reason"] not in {"collision", "wall", "timeout", "other"}:
            raise ProvenanceError("invalid terminal reason")
        if item["result"] == "invalid" and item["terminal_reason"] != "other":
            raise ProvenanceError("invalid result requires terminal_reason=other")
        _encode_required_uint64(item["ticks_completed"], "ticks_completed")
        outcomes.append(item)
    outcomes.sort(key=lambda item: item["initial_slot_id"].encode("utf-8"))
    consumed_initial_ids = {slot["slot_id"] for slot in frozen_slots
                            if status_by_id[slot["slot_id"]] == "consumed"
                            and slot["game_index"] == 0 and slot["reset_index"] == 0}
    if completed_run and initial_ids != consumed_initial_ids:
        raise ProvenanceError("completed-game outcomes must cover every consumed initial game exactly once")
    payload = {"expanded_schedule_digest": expected_digest, "realized_slots": normalized_realized,
               "realized_tuple_list": consumed_tuples, "episode_outcomes": outcomes}
    return {**payload, "realized_schedule_digest": _sha256(_canonical_json(payload))}


def validate_schedule_binding(binding: Mapping[str, Any], *, expanded: Mapping[str, Any],
                              realized: Mapping[str, Any], seed_plan_sha: str | None = None) -> None:
    """Fail closed when a consumer artifact omits or misbinds schedule ancestry."""
    actual_seed_sha = seed_plan_sha or seed_plan_sha256()
    required = {"seed_plan_sha256", "schedule_spec_digest", "expanded_schedule_digest",
                "realized_schedule_digest"}
    if not isinstance(binding, Mapping) or set(binding) != required:
        raise ProvenanceError("schedule binding must contain all four exact digest fields")
    expected_spec = schedule_spec_digest(actual_seed_sha)
    if binding["seed_plan_sha256"] != actual_seed_sha:
        raise ProvenanceError("schedule binding has a missing or stale seed-plan hash")
    if binding["schedule_spec_digest"] != expected_spec or expanded.get("schedule_spec_digest") != expected_spec:
        raise ProvenanceError("schedule binding has a missing or stale schedule-spec digest")
    if expanded.get("expanded_schedule_digest") != binding["expanded_schedule_digest"]:
        raise ProvenanceError("schedule binding has a missing or wrong-parent expanded digest")
    if realized.get("expanded_schedule_digest") != binding["expanded_schedule_digest"]:
        raise ProvenanceError("realized schedule has the wrong expanded parent")
    if realized.get("realized_schedule_digest") != binding["realized_schedule_digest"]:
        raise ProvenanceError("schedule binding has a missing or stale realized digest")
    realized_fields = {"expanded_schedule_digest", "realized_slots", "realized_tuple_list",
                       "episode_outcomes", "realized_schedule_digest"}
    if set(realized) != realized_fields:
        raise ProvenanceError("realized schedule has missing or unexpected fields")
    rebuilt = validate_realization(expanded,
                                   realized_slots=realized["realized_slots"],
                                   realized_tuple_list=realized["realized_tuple_list"],
                                   episode_outcomes=realized["episode_outcomes"],
                                   completed_run=True)
    if rebuilt != dict(realized):
        raise ProvenanceError("realized schedule is stale, malformed, or forged")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the canonical manifest digest, excluding its digest field."""
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seed_plan_sha256() -> str:
    """Digest the exact frozen seed-plan artifact bytes used by this validator."""
    try:
        return hashlib.sha256(SEED_PLAN_PATH.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProvenanceError(f"cannot read frozen G008 seed plan: {exc}") from exc


def _read_seed_plan() -> tuple[dict[str, Any], str]:
    try:
        stat = SEED_PLAN_PATH.stat()
        return _read_seed_plan_cached(str(SEED_PLAN_PATH.resolve()), stat.st_mtime_ns, stat.st_size)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read frozen G008 seed plan: {exc}") from exc


@lru_cache(maxsize=8)
def _read_seed_plan_cached(path: str, _mtime_ns: int, _size: int) -> tuple[dict[str, Any], str]:
    contents = Path(path).read_bytes()
    plan = json.loads(contents)
    digest = hashlib.sha256(contents).hexdigest()
    if plan.get("format") != "circuit-ai-g008-split-seed-plan/3" or digest != FROZEN_SEED_PLAN_SHA256:
        raise ProvenanceError("G008 requires the exact approved seed-plan /3 bytes")
    return plan, digest


def _contains_open_space_token(source_id: str) -> bool:
    """Recognize the protected policy name amid ordinary ID separators/case."""
    normalized = unicodedata.normalize("NFKC", source_id).casefold()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return any(tokens[index:index + 2] == ["open", "space"] for index in range(len(tokens) - 1))


def validate_manifest(manifest: Mapping[str, Any], *,
                      expanded_schedules: Mapping[str, Mapping[str, Any]] | None = None,
                      realized_schedules: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    """Validate schema, frozen digest, full coverage, and split isolation.

    The function returns ``None`` on success and raises :class:`ProvenanceError`
    on every invalid or incomplete manifest. Callers must invoke it before
    loading training, promotion, or evaluation rows.
    """
    if not isinstance(manifest, Mapping):
        raise ProvenanceError("manifest must be an object")
    if manifest.get("format") == "lc-split-manifest/1":
        raise ProvenanceError("G008 consumer rejects split-manifest v1; /3 is required")
    errors = sorted(_VALIDATOR.iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise ProvenanceError(f"manifest schema violation at {location}: {first.message}")
    if manifest["frozen"] is not True:
        raise ProvenanceError("manifest is not frozen")
    if manifest["manifest_sha256"] != manifest_sha256(manifest):
        raise ProvenanceError("manifest SHA-256 does not match canonical contents")
    seed_plan, actual_seed_plan_sha256 = _read_seed_plan()
    if seed_plan.get("format") != "circuit-ai-g008-split-seed-plan/3" or actual_seed_plan_sha256 != FROZEN_SEED_PLAN_SHA256:
        raise ProvenanceError("G008 consumer requires the exact approved seed-plan /3 bytes")
    if manifest["seed_plan_sha256"] != actual_seed_plan_sha256:
        raise ProvenanceError("manifest seed_plan_sha256 does not match the frozen seed-plan artifact")
    expected_spec_digest = schedule_spec_digest(actual_seed_plan_sha256)
    if manifest["schedule_spec_digest"] != expected_spec_digest:
        raise ProvenanceError("manifest schedule_spec_digest is missing or stale")
    bindings = manifest["schedule_bindings"]
    if not bindings:
        raise ProvenanceError("manifest schedule_bindings must include every consumed run")
    if expanded_schedules is None or realized_schedules is None:
        raise ProvenanceError("manifest schedule artifacts are required before rows can be consumed")
    if (not isinstance(expanded_schedules, Mapping) or not isinstance(realized_schedules, Mapping)
            or any(not isinstance(item, Mapping) for item in expanded_schedules.values())
            or any(not isinstance(item, Mapping) for item in realized_schedules.values())):
        raise ProvenanceError("manifest schedule artifacts must be run-ID mappings of schedule objects")
    seen_run_ids = set()
    for binding in bindings:
        run_id = binding["run_id"]
        if run_id in seen_run_ids:
            raise ProvenanceError("manifest schedule_bindings contain a duplicate run_id")
        seen_run_ids.add(run_id)
        if binding["seed_plan_sha256"] != actual_seed_plan_sha256:
            raise ProvenanceError("manifest schedule binding has a wrong seed-plan hash")
        if binding["schedule_spec_digest"] != expected_spec_digest:
            raise ProvenanceError("manifest schedule binding has a missing or stale schedule-spec digest")
        if run_id not in expanded_schedules or run_id not in realized_schedules:
            raise ProvenanceError(f"manifest schedule artifacts are missing for run_id {run_id}")
        if expanded_schedules[run_id].get("run_id") != run_id:
            raise ProvenanceError("expanded schedule run_id does not match manifest schedule binding")
        validate_schedule_binding(
            {key: binding[key] for key in ("seed_plan_sha256", "schedule_spec_digest",
                                            "expanded_schedule_digest", "realized_schedule_digest")},
            expanded=expanded_schedules[run_id], realized=realized_schedules[run_id],
            seed_plan_sha=actual_seed_plan_sha256)
    if seen_run_ids != set(expanded_schedules) or seen_run_ids != set(realized_schedules):
        raise ProvenanceError("manifest schedule artifacts have unreferenced or missing runs")

    rows = manifest["rows"]
    covered_splits = {row["split"] for row in rows}
    missing_splits = set(_SPLITS) - covered_splits
    if missing_splits:
        raise ProvenanceError(f"missing split coverage: {', '.join(sorted(missing_splits))}")
    if not any(row["split"] == "final" and row["opponent_id"] == "open-space"
               and row["usages"] == ["final_eval"]
               and row["opponent_code_sha256"] == OPEN_SPACE_CODE_SHA256
               and row["opponent_config_sha256"] == OPEN_SPACE_CONFIG_SHA256
               for row in rows):
        raise ProvenanceError("final split must contain the exact open-space policy identity")

    # The same opponent identity, ID, or component hash must never cross a split.
    opponent_owners: dict[str, str] = {}
    source_id_owners: dict[str, str] = {}
    source_data_owners: dict[str, str] = {}
    identity_by_id: dict[str, tuple[str, ...]] = {}
    id_by_identity: dict[tuple[str, ...], str] = {}
    seeds_by_split: dict[str, set[int]] = {split: set() for split in _SPLITS}
    game_keys: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        split = row["split"]
        if not _SPLIT_USAGES[split].issuperset(row["usages"]):
            raise ProvenanceError(f"row {index} has usage incompatible with split {split}")
        # Schema enforces all hashes, while this explicit check documents that
        # no optional/null provenance is accepted at the execution boundary.
        for field in _ALL_PROVENANCE_HASH_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ProvenanceError(f"row {index} is missing valid SHA-256 provenance: {field}")

        opponent_id = row["opponent_id"]
        identity = tuple(row[field] for field in _HASH_FIELDS)
        prior = identity_by_id.setdefault(opponent_id, identity)
        if prior != identity:
            raise ProvenanceError(f"opponent ID {opponent_id!r} has inconsistent hashes")
        prior_id = id_by_identity.setdefault(identity, opponent_id)
        if prior_id != opponent_id:
            raise ProvenanceError(f"opponent hash identity is aliased by {prior_id!r} and {opponent_id!r}")
        for key in (opponent_id, *identity):
            owner = opponent_owners.setdefault(key, split)
            if owner != split:
                raise ProvenanceError(f"opponent identity/hash overlaps {owner} and {split}: {key}")

        source_id = row["source_id"]
        owner = source_id_owners.setdefault(source_id, split)
        if owner != split:
            raise ProvenanceError(f"source ID overlaps {owner} and {split}: {source_id}")
        source_hash = row["source_data_sha256"]
        owner = source_data_owners.setdefault(source_hash, split)
        if owner != split:
            raise ProvenanceError(f"source data hash overlaps {owner} and {split}: {source_hash}")

        is_open_space = opponent_id == "open-space"
        has_open_space_fingerprint = (
            row["opponent_code_sha256"] == OPEN_SPACE_CODE_SHA256
            or row["opponent_config_sha256"] == OPEN_SPACE_CONFIG_SHA256
        )
        if has_open_space_fingerprint and not (
            split == "final"
            and opponent_id == "open-space"
            and row["usages"] == ["final_eval"]
            and row["opponent_code_sha256"] == OPEN_SPACE_CODE_SHA256
            and row["opponent_config_sha256"] == OPEN_SPACE_CONFIG_SHA256
        ):
            raise ProvenanceError("open-space fingerprint must use its exact final-evaluation identity")
        if is_open_space and (split != "final" or row["usages"] != ["final_eval"]):
            raise ProvenanceError("exact open-space policy/data is final-evaluation-only")
        if split != "final" and (is_open_space or _contains_open_space_token(source_id)
                                  or has_open_space_fingerprint):
            raise ProvenanceError("open-space policy/data cannot appear outside final evaluation")

        seed = row["seed"]
        try:
            lower, upper = seed_plan["partitions"][split]["root_seed_range_inclusive"]
        except (KeyError, TypeError) as exc:
            raise ProvenanceError(f"frozen seed plan has no valid range for {split}") from exc
        if not lower <= seed <= upper:
            raise ProvenanceError(f"row {index} seed {seed} is outside frozen {split} range [{lower}, {upper}]")
        seeds_by_split[split].add(seed)
        key = (split, opponent_id, *identity, row["target_checkpoint_sha256"],
               row["training_config_sha256"], row["source_data_sha256"], seed,
               row["arena_index"], row["game_index"], row["target_seat"])
        if key in game_keys:
            raise ProvenanceError(f"duplicate game key at row {index}")
        game_keys.add(key)

    for left_index, left in enumerate(_SPLITS):
        for right in _SPLITS[left_index + 1:]:
            overlap = seeds_by_split[left] & seeds_by_split[right]
            if overlap:
                raise ProvenanceError(f"seed overlap between {left} and {right}: {min(overlap)}")

