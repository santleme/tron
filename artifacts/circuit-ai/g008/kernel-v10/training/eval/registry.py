"""Append-only, hash-chained G008 experiment registry.

This is a local integrity mechanism, not a signature. It detects inconsistent
partial edits while independently captured head hashes remain available.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
_HASH_CHARS = frozenset("0123456789abcdef")
_REGISTRATION_KINDS = frozenset({"train", "imitation", "ablation", "validation", "final", "benchmark"})


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


def _keys(data: Any, expected: set[str], event_type: str) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != expected:
        raise RegistryError(f"{event_type} event_data must contain exactly {sorted(expected)}")
    _normal(data)
    return data


def _validate_data(event_type: str, data: Any) -> None:
    if event_type == "registration":
        d = _keys(data, {"registration_id", "registration_kind", "record"}, event_type)
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
        d = _keys(data, {"artifact_sha256", "pre_lock_head_hash", "pre_lock_final_event_index",
                         "registration_event_index", "registration_event_hash"}, event_type)
        _validate_benchmark_binding(d)
    elif event_type == "benchmark_attestation":
        d = _keys(data, {"artifact_sha256", "pre_lock_head_hash", "pre_lock_final_event_index",
                         "registration_event_index", "registration_event_hash", "lock_head_hash"}, event_type)
        _validate_benchmark_binding(d)
        if not _is_hash(d["lock_head_hash"]):
            raise RegistryError("invalid lock_head_hash")
    elif event_type in {"workload_completion", "evidence"}:
        d = _keys(data, {"registration_event_index", "registration_event_hash", "lock_head_hash", "artifact_sha256"}, event_type)
        if (not isinstance(d["registration_event_index"], int) or isinstance(d["registration_event_index"], bool)
                or d["registration_event_index"] < 0):
            raise RegistryError("invalid registration_event_index")
        if not all(_is_hash(d[k]) for k in ("registration_event_hash", "lock_head_hash", "artifact_sha256")):
            raise RegistryError("invalid hash in post-lock evidence")
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
    return events


def _verify_benchmark_registration(data: dict[str, Any], events: list[dict[str, Any]], registrations: dict[str, int]) -> None:
    i = data["registration_event_index"]
    if i < 0 or i >= len(events):
        raise RegistryError("benchmark registration reference is not prior")
    event = events[i]
    if (event["event_type"] != "registration" or event["event_data"]["registration_kind"] != "benchmark"
            or event["event_hash"] != data["registration_event_hash"]):
        raise RegistryError("benchmark artifact registration reference mismatch")


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

    def register_benchmark_artifact(self, artifact_sha256: str, registration_event_index: int,
                                    registration_event_hash: str) -> dict[str, Any]:
        events = self.events()
        head = events[-1]["event_hash"] if events else None
        data = {"artifact_sha256": artifact_sha256, "pre_lock_head_hash": head,
                "pre_lock_final_event_index": len(events)-1,
                "registration_event_index": registration_event_index,
                "registration_event_hash": registration_event_hash}
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
