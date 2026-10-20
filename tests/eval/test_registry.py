import hashlib
import json

import pytest

from training.eval.registry import GENESIS, Registry, RegistryError, canonical_bytes


def test_canonical_event_contract_rejects_float_and_non_nfc():
    assert canonical_bytes({"z": "ok", "a": 1}) == b'{"a":1,"z":"ok"}'
    with pytest.raises(RegistryError, match="floats"):
        canonical_bytes({"x": 1.0})
    with pytest.raises(RegistryError, match="NFC"):
        canonical_bytes({"x": "e\u0301"})


def test_lock_and_benchmark_attestation_lifecycle(tmp_path):
    registry = Registry(tmp_path / "registry.jsonl")
    assert registry.head == (None, -1)
    registration = registry.register("bench-1", "benchmark", {"config_sha256": "a" * 64})
    artifact_hash = hashlib.sha256(b"timings and parity only").hexdigest()
    artifact = registry.register_benchmark_artifact(artifact_hash, 0, registration["event_hash"])
    prefix = registration["event_hash"]
    assert artifact["event_data"]["pre_lock_head_hash"] == prefix
    assert "lock_head_hash" not in artifact["event_data"]
    lock = registry.close_registrations()
    assert lock["event_data"] == {"pre_lock_head_hash": artifact["event_hash"],
                                  "pre_lock_final_event_index": 1}
    assert registry.lock_head_hash == lock["event_hash"]
    with pytest.raises(RegistryError, match="after lock"):
        registry.register("new-run", "train", {})
    attestation = registry.attest_benchmark(1)
    assert attestation["event_data"]["lock_head_hash"] == lock["event_hash"]
    assert registry.benchmark_is_attested(1, artifact_hash)
    assert not registry.benchmark_is_attested(1, "0" * 64)
    assert registry.events()[-1]["prev_event_hash"] == attestation["prev_event_hash"]


def test_empty_registry_lock_uses_null_and_minus_one(tmp_path):
    registry = Registry(tmp_path / "empty.jsonl")
    lock = registry.close_registrations()
    assert lock["prev_event_hash"] == GENESIS
    assert lock["event_data"] == {"pre_lock_head_hash": None, "pre_lock_final_event_index": -1}
    assert registry.lock_head_hash == lock["event_hash"]


def test_duplicate_registration_and_lock_are_rejected_without_corrupting_file(tmp_path):
    registry = Registry(tmp_path / "registry.jsonl")
    registry.register("run-1", "train", {"config": "x"})
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="duplicate"):
        registry.register("run-1", "train", {"config": "x"})
    registry.close_registrations()
    locked = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="irreversible"):
        registry.close_registrations()
    assert registry.path.read_bytes() == locked
    assert len(registry.events()) == 2


def test_post_lock_evidence_must_pin_registered_run_and_lock(tmp_path):
    registry = Registry(tmp_path / "registry.jsonl")
    registration = registry.register("train-1", "train", {"run": 1})
    lock = registry.close_registrations()
    valid = {"registration_event_index": 0, "registration_event_hash": registration["event_hash"],
             "lock_head_hash": lock["event_hash"], "artifact_sha256": "b" * 64}
    with pytest.raises(RegistryError, match="registration hash"):
        registry._append("evidence", {**valid, "registration_event_hash": "c" * 64})
    with pytest.raises(RegistryError, match="lock hash"):
        registry._append("evidence", {**valid, "lock_head_hash": "c" * 64})
    registry._append("evidence", valid)
    assert len(registry.events()) == 3


@pytest.mark.parametrize("mutation", ["payload", "reorder", "hash"])
def test_replay_fails_closed_on_registry_mutation(tmp_path, mutation):
    registry = Registry(tmp_path / "registry.jsonl")
    registry.register("run-1", "train", {"x": 1})
    registry.register("run-2", "imitation", {"x": 2})
    lines = registry.path.read_bytes().splitlines()
    if mutation == "payload":
        event = json.loads(lines[0]); event["event_data"]["record"]["x"] = 99
        lines[0] = canonical_bytes(event)
    elif mutation == "reorder":
        lines.reverse()
    else:
        event = json.loads(lines[0]); event["event_hash"] = "f" * 64
        lines[0] = canonical_bytes(event)
    registry.path.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(RegistryError):
        registry.events()


def test_independent_anchor_detects_truncation(tmp_path):
    registry = Registry(tmp_path / "registry.jsonl")
    registry.register("run-1", "train", {"x": 1})
    second = registry.register("run-2", "imitation", {"x": 2})
    registry.path.write_bytes(registry.path.read_bytes().splitlines()[0] + b"\n")
    with pytest.raises(RegistryError, match="anchor"):
        registry.events(expected_head_hash=second["event_hash"], expected_final_event_index=1)


def test_stale_or_duplicate_benchmark_attestation_rejected(tmp_path):
    registry = Registry(tmp_path / "registry.jsonl")
    reg = registry.register("bench", "benchmark", {})
    digest = hashlib.sha256(b"artifact").hexdigest()
    registry.register_benchmark_artifact(digest, 0, reg["event_hash"])
    registry.close_registrations()
    registry.attest_benchmark(1)
    with pytest.raises(RegistryError, match="duplicate"):
        registry.attest_benchmark(1)
    assert registry.benchmark_is_attested(1, digest)


def test_bad_lock_anchor_is_rejected_on_replay(tmp_path):
    registry = Registry(tmp_path / "registry.jsonl")
    registry.register("run", "train", {})
    registry.close_registrations()
    lines = registry.path.read_bytes().splitlines()
    event = json.loads(lines[1])
    event["event_data"]["pre_lock_final_event_index"] = 2
    lines[1] = canonical_bytes(event)
    registry.path.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(RegistryError, match="hash mismatch|pin exact"):
        registry.events()

