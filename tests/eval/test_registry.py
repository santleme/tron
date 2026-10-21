import hashlib
import json
import stat
import uuid
import copy

import pytest

from training.eval.registry import GENESIS, Registry, RegistryError, canonical_bytes
import training.eval.registry as registry_module


def _selection_evidence(selection="incumbent", *, parity=True, ratios=None, median=None):
    ratios = [1.0] * 5 if ratios is None else ratios
    median = sorted(ratios)[2] if median is None else median
    return {
        "parity": {"passed": parity},
        "pairs": [{"candidate_speedup_ratio": ratio} for ratio in ratios],
        "summary": {"candidate_speedup_ratio": {"median": median}},
        "selection": selection,
        "selected_variant_id": selection,
    }


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


def _legacy_registry_with_recovery_amendment(tmp_path, monkeypatch):
    registry = Registry(tmp_path / "registry.jsonl")
    legacy_benchmark = registry.register("legacy-0", "benchmark", {"legacy_index": 0})
    for index in range(1, 10):
        registry.register(f"legacy-{index}", "train", {"legacy_index": index})
    # The immutable 0–12 prefix includes its historical benchmark artifact at index 10.
    registry.register_benchmark_artifact("a" * 64, 0, legacy_benchmark["event_hash"])
    for index in range(11, 13):
        registry.register(f"legacy-{index}", "train", {"legacy_index": index})
    monkeypatch.setattr(registry_module, "_LEGACY_HEAD_INDEX", 12)
    monkeypatch.setattr(registry_module, "_LEGACY_HEAD_HASH", registry.head[0])
    benchmark_uuid, training_uuid = str(uuid.uuid4()), str(uuid.uuid4())
    amendment = {
        "schema_id": "g008-kaggle-recovery-amendment", "schema_version": 1,
        "created_at_utc": "2026-10-20T18:00:00Z", "legacy_head_hash": registry.head[0],
        "legacy_final_event_index": 12, "recovery_plan_sha256": registry_module._RECOVERY_PLAN_SHA256,
        "superseded_plan_sha256": "0a9e1ebccae2a793fadf531fb99013a9d19ff832a5880de9ee15204b477ca81a",
        "superseded_test_spec_sha256": "e3b8efef4b441d694883a856164ed3d25f2b278f2bd7ca9ee30dfb8097b3ba6c",
        "historical_overrun_evidence": {
            "overrun": True, "kernel_id": "owner/kernel", "current_version_number": 10,
            "all_prior_attempts_terminal": True,
            "prior_attempts_terminal_evidence": [{"kernel_id": "owner/kernel", "version": 10,
                                                  "status": "CANCELLED", "evidence": {"captured": True}}],
        }, "user_request_basis": {"authorized": True},
        "recovery_registrations": [
            {"registration_id": "recovery-bench", "registration_kind": "benchmark", "run_uuid": benchmark_uuid},
            {"registration_id": "recovery-train", "registration_kind": "train", "run_uuid": training_uuid},
        ],
        "submission_attempt_limit": None,
        "compute_caps_gpu_seconds": {"benchmark": 3600, "training": 7200,
                                      "project_experiment": 21600, "project_total": 28800},
        "output_caps_bytes": {"aggregate": 5_000_000_000, "per_run": 1_000_000_000},
        "protected_final_reserve_gpu_seconds": 7200,
        "required_preflight": {"one_t4": True},
        "reservation_metadata": {
            "approved_corrected_cycle5_seed_plan_sha256": registry_module._RECOVERY_SEED_PLAN_SHA256,
            "schedule_spec_digest": "ca13f87ee6b45f7b4d340f6c62715d4980f9966468e5511f7a06953f056c8ec5",
            "validation_suite_sha256": "ee9db0e48dc055ed3e744a92166f274003f0d435cb58014acbd3e8d250d151b0",
            "final_suite_sha256": "1178831d3373f596affe032b88fc8a0768eec406d038e6c35bd9adfb84afefa3",
            "validation_opponent_roster_sha256": "974adf788d5519633ad1fe7130a6deb634a0cc55d6af2d549e06d64c0f5496c0",
            "final_opponent_roster_sha256": "0fa20d714149054782dd5a83ab9ae99ebca6582b6c5702bdcc206bf7e3b61c27",
            "validation_opponent_checkpoint_sha256": "f79ed460afc6d4419e2e1b0bd084a4f1cf77058fcb8cdcebb7122a2864304412",
            "final_opponent_code_sha256": "9e5103176218ca092206ba50899a35c43ad488ce825a95af487e76798ce60123",
            "final_opponent_config_sha256": "4043dda35bb8cc2adc43112cb5b00eff50ac721eb062b15d0d83b7ae41510b7f",
            "validation_opponent_id": "g007-step-300", "final_opponent_id": "open-space", "reservation_only": True,
            "pre_lock_execution": False, "campaign_gpu_seconds_before": 8424,
            "campaign_output_bytes_before": 26403, "quota_gpu_seconds_remaining_before": 100000,
            "validation_game_count": 512, "final_game_count": 512, "target_seat_order": [0, 1],
            "validation_roots": list(range(1_000_000, 1_000_256)),
            "final_roots": list(range(4_000_000, 4_000_256)),
            "validation_root_range": [1_000_000, 1_000_255], "final_root_range": [4_000_000, 4_000_255],
            "protocol_selection_policy": {
                "protocol": "G008 frozen principal protocol / schedule seed plan /3",
                "selection": "candidate requires parity and median paired improvement >=10% with at least 4/5 pairs >=10%; otherwise use incumbent"},
            "reserved_gpu_seconds": {"benchmark": 3600, "training": 7200, "validation": 0, "final": 7200},
            "reserved_runtime_seconds": {"benchmark": 3600, "training": 7200, "validation": 0, "final": 7200},
            "reserved_output_bytes": {"benchmark": 1000, "training": 1000, "validation": 1000, "final": 1000},
        },
    }
    amendment_event = registry.register_recovery_amendment(amendment)
    variants = [
        {"variant_id": variant, "package_sha256": format(int(digit), "064x"),
         "source_sha256": format(int(digit) + 1, "064x"),
         "trainer_sha256": format(int(digit) + 2, "064x"), "backend_sha256": format(int(digit) + 3, "064x"),
         "engine_sha256": format(int(digit) + 4, "064x"), "config_sha256": "5" * 64,
         "seed_plan_sha256": registry_module._RECOVERY_SEED_PLAN_SHA256}
        for variant, digit in (("incumbent", "3"), ("candidate", "7"))
    ]
    def register(kind, rid, run_uuid, timeout, cap):
        return registry.register_recovery_workload(rid, kind, {
            "run_uuid": run_uuid, "recovery_amendment_event_index": amendment_event["event_index"],
            "recovery_amendment_event_hash": amendment_event["event_hash"],
            "submission_policy": "cumulative_compute_only", "session_timeout_seconds": timeout,
            "workload_gpu_seconds_cap": cap, "workload_output_bytes_cap": 1_000_000_000,
            "accelerator_count": 1, "machine_shape": "NvidiaTeslaT4", "kernel_id": "owner/kernel", "package_sha256": "3" * 64,
            "source_sha256": "4" * 64, "config_sha256": "5" * 64,
            "seed_plan_sha256": registry_module._RECOVERY_SEED_PLAN_SHA256,
            "backend_sha256": "6" * 64, "trainer_sha256": "7" * 64, "engine_sha256": "8" * 64,
            **({"allowed_implementation_variants": variants} if kind == "train" else {"implementation_variants": variants}),
        })
    # Construct the fully authorized pair independently of the validator under regression. The
    # suffix validator currently rejects the second append because it mistakes legacy artifact 10
    # for a recovery artifact while reading the otherwise valid one-registration prefix.
    with monkeypatch.context() as setup:
        setup.setattr(registry_module, "_validate_recovery_suffix", lambda _events: None)
        bench = register("benchmark", "recovery-bench", benchmark_uuid, 3600, 3600)
        train = register("train", "recovery-train", training_uuid, 7200, 7200)
    return registry, bench, train, amendment_event


def test_recovery_registration_prefix_allows_one_of_two_before_execution(tmp_path, monkeypatch):
    registry, benchmark, train, amendment = _legacy_registry_with_recovery_amendment(tmp_path, monkeypatch)

    # The historical prefix remains byte-for-byte intact and contains its old artifact.
    prefix = registry.path.read_bytes().splitlines()
    assert len(prefix) == 16  # 13 legacy events, amendment, benchmark registration, train registration
    legacy_events = [json.loads(line) for line in prefix[:13]]
    assert legacy_events[10]["event_type"] == "benchmark_artifact"
    assert legacy_events[12]["event_hash"] == amendment["event_data"]["legacy_head_hash"]

    # Start from the valid amended prefix with no recovery workload registrations.
    registry.path.write_bytes(b"\n".join(prefix[:14]) + b"\n")
    events = registry.events()
    assert [event["event_type"] for event in events[13:]] == ["recovery_amendment"]

    first = registry.register_recovery_workload(
        benchmark["event_data"]["registration_id"], benchmark["event_data"]["registration_kind"],
        benchmark["event_data"]["record"],
    )
    assert first["event_hash"] == benchmark["event_hash"]
    assert [event["event_type"] for event in registry.events()[13:]] == [
        "recovery_amendment", "registration",
    ]

    # Neither workload execution nor locking may begin with only one frozen registration.
    with pytest.raises(RegistryError, match="both recovery registrations"):
        registry.preflight_workload(_preflight(benchmark))
    with pytest.raises(RegistryError, match="both recovery registrations"):
        registry.close_registrations()
    before_artifact_attempt = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="both recovery registrations"):
        registry.register_benchmark_artifact(
            "8" * 64, first["event_index"], first["event_hash"],
            attempt_event_index=first["event_index"], attempt_event_hash="9" * 64,
            report_sha256="a" * 64, selection_evidence=_selection_evidence(),
        )
    assert registry.path.read_bytes() == before_artifact_attempt

    # Appending the second named workload completes the valid registration prefix.
    second = registry.register_recovery_workload(
        train["event_data"]["registration_id"], train["event_data"]["registration_kind"],
        train["event_data"]["record"],
    )
    completed_prefix = registry.events()
    assert [event["event_type"] for event in completed_prefix[13:]] == [
        "recovery_amendment", "registration", "registration",
    ]
    assert second["event_hash"] == train["event_hash"]
    assert completed_prefix[-1]["event_hash"] == train["event_hash"]


def _preflight(registration, attempt_id="attempt-1", *, lock_hash=None, timeout=3600, variant_id=None):
    record = registration["event_data"]["record"]
    kind = registration["event_data"]["registration_kind"]
    variant_id = variant_id or ("incumbent" if kind == "train" else "paired")
    variants = record.get("allowed_implementation_variants", record.get("implementation_variants", []))
    variant = next((v for v in variants if v["variant_id"] == variant_id), None)
    return {
        "attempt_id": attempt_id, "registration_event_index": registration["event_index"],
        "registration_event_hash": registration["event_hash"],
        "registration_id": registration["event_data"]["registration_id"],
        "kernel_id": record["kernel_id"], "package_sha256": variant["package_sha256"] if variant else record["package_sha256"],
        "source_sha256": variant["source_sha256"] if variant else record["source_sha256"],
        "config_sha256": record["config_sha256"], "seed_plan_sha256": record["seed_plan_sha256"],
        "backend_sha256": variant["backend_sha256"] if variant else record["backend_sha256"],
        "trainer_sha256": variant["trainer_sha256"] if variant else record["trainer_sha256"],
        "engine_sha256": variant["engine_sha256"] if variant else record["engine_sha256"],
        "implementation_variant_id": variant_id,
        "session_timeout_seconds": timeout, "accelerator_count": 1, "reserved_gpu_seconds": timeout,
        "expected_current_version": 10, "quota_evidence": {"remaining_gpu_seconds": 30000},
        "policy_evidence": {"paid_services": False, "billing_changes": False, "machine_shape": "NvidiaTeslaT4",
                            "prior_runs_terminal": True, "no_queued_or_running": True}, "prior_attempts": [],
        "no_outside_writer_confirmed": True, "exclusive_lock_id": "lock-1",
        "reconciliation_slack_seconds": 600, "lock_head_hash": lock_hash,
    }


def _dispatch(preflight, *, attempt_id="attempt-1", lock_hash=None):
    return {
        "attempt_id": attempt_id, "registration_event_index": preflight["registration_event_index"],
        "registration_event_hash": preflight["registration_event_hash"],
        "preflight_event_index": preflight["registration_event_index"] + 1,
        "preflight_event_hash": "0" * 64, "kernel_id": preflight["kernel_id"],
        "package_sha256": preflight["package_sha256"], "source_sha256": preflight["source_sha256"],
        "config_sha256": preflight["config_sha256"], "seed_plan_sha256": preflight["seed_plan_sha256"],
        "backend_sha256": preflight["backend_sha256"], "trainer_sha256": preflight["trainer_sha256"],
        "engine_sha256": preflight["engine_sha256"], "implementation_variant_id": preflight["implementation_variant_id"],
        "session_timeout_seconds": preflight["session_timeout_seconds"],
        "accelerator_count": 1, "reserved_gpu_seconds": preflight["reserved_gpu_seconds"],
        "dispatch_timestamp_utc": "2026-10-20T18:01:00Z", "request_identity": "push-attempt-1",
        "lock_head_hash": lock_hash,
    }


def _attempt(preflight_event, dispatch_event, *, accepted_event=None, outcome="terminal", lock_hash=None):
    return {
        "attempt_id": preflight_event["event_data"]["attempt_id"], "outcome": outcome,
        "registration_event_index": preflight_event["event_data"]["registration_event_index"],
        "registration_event_hash": preflight_event["event_data"]["registration_event_hash"],
        "preflight_event_index": preflight_event["event_index"], "preflight_event_hash": preflight_event["event_hash"],
        "dispatch_event_index": dispatch_event["event_index"], "dispatch_event_hash": dispatch_event["event_hash"],
        "accepted_event_index": accepted_event["event_index"] if accepted_event else None,
        "accepted_event_hash": accepted_event["event_hash"] if accepted_event else None,
        "assigned_version": accepted_event["event_data"]["assigned_version"] if accepted_event else None,
        "implementation_variant_id": preflight_event["event_data"]["implementation_variant_id"],
        "package_sha256": preflight_event["event_data"]["package_sha256"],
        "source_sha256": preflight_event["event_data"]["source_sha256"],
        "config_sha256": preflight_event["event_data"]["config_sha256"],
        "seed_plan_sha256": preflight_event["event_data"]["seed_plan_sha256"],
        "backend_sha256": preflight_event["event_data"]["backend_sha256"],
        "trainer_sha256": preflight_event["event_data"]["trainer_sha256"],
        "engine_sha256": preflight_event["event_data"]["engine_sha256"],
        "terminal_status": "COMPLETE" if accepted_event else None,
        "terminal_metadata": {"current_version_number": accepted_event["event_data"]["assigned_version"],
                              "kernel_id": preflight_event["event_data"]["kernel_id"], "status": "COMPLETE"} if accepted_event else None,
        "status_evidence": [{"status": "COMPLETE",
                             "current_version_number": accepted_event["event_data"]["assigned_version"],
                             "kernel_id": preflight_event["event_data"]["kernel_id"]}] if accepted_event else None,
        "actual_gpu_seconds": 120 if accepted_event else 0, "output_bytes": 100,
        "result": {"ok": True}, "artifact_sha256": "7" * 64 if accepted_event else None,
        "rejection_evidence": None if accepted_event else {"server_rejected_before_assignment": True},
        "no_assignment_metadata": None if accepted_event else {
            "pre_push_current_version": 10, "post_push_current_version": 10, "no_version_assigned": True},
        "lock_head_hash": lock_hash,
    }


def test_recovery_append_only_suffix_pairs_dispatch_receipt_terminal_and_lock(tmp_path, monkeypatch):
    registry, benchmark, _, _ = _legacy_registry_with_recovery_amendment(tmp_path, monkeypatch)
    legacy_prefix = b"\n".join(registry.path.read_bytes().splitlines()[:13]) + b"\n"
    preflight_data = _preflight(benchmark)
    preflight = registry.preflight_workload(preflight_data)
    dispatch_data = _dispatch(preflight_data)
    dispatch_data.update(preflight_event_index=preflight["event_index"], preflight_event_hash=preflight["event_hash"])
    dispatch = registry.dispatch_workload_push(dispatch_data)
    accepted_data = {
        "attempt_id": "attempt-1", "registration_event_index": benchmark["event_index"],
        "registration_event_hash": benchmark["event_hash"], "preflight_event_index": preflight["event_index"],
        "preflight_event_hash": preflight["event_hash"], "dispatch_event_index": dispatch["event_index"],
        "dispatch_event_hash": dispatch["event_hash"], "assigned_version": 11,
        "implementation_variant_id": "paired",
        "push_response": {"assigned_version": 11},
        "confirming_metadata": {"current_version_number": 11, "kernel_id": "owner/kernel"},
        "lock_head_hash": None,
    }
    accepted = registry.accept_workload_push(accepted_data)
    attempt = registry.resolve_workload_attempt(_attempt(preflight, dispatch, accepted_event=accepted))
    artifact = registry.register_benchmark_artifact(
        "8" * 64, benchmark["event_index"], benchmark["event_hash"],
        attempt_event_index=attempt["event_index"], attempt_event_hash=attempt["event_hash"],
        report_sha256="9" * 64, selection_evidence=_selection_evidence())
    lock = registry.close_registrations()
    assert lock["event_index"] == artifact["event_index"] + 1
    assert registry.events()[-2]["event_data"]["attempt_event_hash"] == attempt["event_hash"]
    assert registry.path.read_bytes().startswith(legacy_prefix)


def test_pending_dispatch_keeps_reservation_and_blocks_any_following_event(tmp_path, monkeypatch):
    registry, benchmark, _, _ = _legacy_registry_with_recovery_amendment(tmp_path, monkeypatch)
    preflight_data = _preflight(benchmark)
    preflight = registry.preflight_workload(preflight_data)
    dispatch_data = _dispatch(preflight_data)
    dispatch_data.update(preflight_event_index=preflight["event_index"], preflight_event_hash=preflight["event_hash"])
    registry.dispatch_workload_push(dispatch_data)
    with pytest.raises(RegistryError, match="pending|contiguous|unknown|prior attempt"):
        registry.preflight_workload(_preflight(benchmark, attempt_id="retry"))


def test_preassignment_rejection_is_terminal_and_releases_reservation(tmp_path, monkeypatch):
    registry, benchmark, _, _ = _legacy_registry_with_recovery_amendment(tmp_path, monkeypatch)
    pf_data = _preflight(benchmark)
    pf = registry.preflight_workload(pf_data)
    ds_data = _dispatch(pf_data)
    ds_data.update(preflight_event_index=pf["event_index"], preflight_event_hash=pf["event_hash"])
    ds = registry.dispatch_workload_push(ds_data)
    terminal = registry.resolve_workload_attempt(_attempt(pf, ds, outcome="pre_assignment_rejected"))
    assert terminal["event_data"]["actual_gpu_seconds"] == 0
    assert registry.events()[-1]["event_hash"] == terminal["event_hash"]


def test_training_requires_attested_prelock_benchmark_and_binds_lock_and_variant(tmp_path, monkeypatch):
    registry, benchmark, training, _ = _legacy_registry_with_recovery_amendment(tmp_path, monkeypatch)
    with pytest.raises(RegistryError, match="require registration lock"):
        registry.preflight_workload(_preflight(training, lock_hash=None, timeout=7200))

    pf_data = _preflight(benchmark)
    pf = registry.preflight_workload(pf_data)
    ds_data = _dispatch(pf_data)
    ds_data.update(preflight_event_index=pf["event_index"], preflight_event_hash=pf["event_hash"])
    ds = registry.dispatch_workload_push(ds_data)
    accepted_data = {
        "attempt_id": "attempt-1", "registration_event_index": benchmark["event_index"],
        "registration_event_hash": benchmark["event_hash"], "preflight_event_index": pf["event_index"],
        "preflight_event_hash": pf["event_hash"], "dispatch_event_index": ds["event_index"],
        "dispatch_event_hash": ds["event_hash"], "assigned_version": 11,
        "implementation_variant_id": "paired", "push_response": {"assigned_version": 11},
        "confirming_metadata": {"current_version_number": 11, "kernel_id": "owner/kernel"}, "lock_head_hash": None,
    }
    accepted = registry.accept_workload_push(accepted_data)
    attempt = registry.resolve_workload_attempt(_attempt(pf, ds, accepted_event=accepted))
    artifact = registry.register_benchmark_artifact(
        "8" * 64, benchmark["event_index"], benchmark["event_hash"],
        attempt_event_index=attempt["event_index"], attempt_event_hash=attempt["event_hash"],
        report_sha256="9" * 64, selection_evidence=_selection_evidence())
    lock = registry.close_registrations()
    registry.attest_benchmark(artifact["event_index"])
    wrong_variant = _preflight(
        training, attempt_id="wrong-variant", lock_hash=lock["event_hash"], timeout=7200,
        variant_id="candidate")
    wrong_variant["expected_current_version"] = 11
    with pytest.raises(RegistryError, match="benchmark-selected"):
        registry.preflight_workload(wrong_variant)
    train_data = _preflight(training, attempt_id="train-attempt", lock_hash=lock["event_hash"], timeout=7200)
    train_data["expected_current_version"] = 11
    train_pf = registry.preflight_workload(train_data)
    train_dispatch_data = _dispatch(train_data, attempt_id="train-attempt", lock_hash=lock["event_hash"])
    train_dispatch_data.update(preflight_event_index=train_pf["event_index"], preflight_event_hash=train_pf["event_hash"])
    train_dispatch = registry.dispatch_workload_push(train_dispatch_data)
    train_accepted_data = {
        "attempt_id": "train-attempt", "registration_event_index": training["event_index"],
        "registration_event_hash": training["event_hash"], "preflight_event_index": train_pf["event_index"],
        "preflight_event_hash": train_pf["event_hash"], "dispatch_event_index": train_dispatch["event_index"],
        "dispatch_event_hash": train_dispatch["event_hash"], "assigned_version": 12,
        "implementation_variant_id": "incumbent", "push_response": {"assigned_version": 12},
        "confirming_metadata": {"current_version_number": 12, "kernel_id": "owner/kernel"}, "lock_head_hash": lock["event_hash"],
    }
    train_accepted = registry.accept_workload_push(train_accepted_data)
    terminal_data = _attempt(train_pf, train_dispatch, accepted_event=train_accepted, lock_hash=lock["event_hash"])
    terminal_data["attempt_id"] = "train-attempt"
    terminal = registry.resolve_workload_attempt(terminal_data)
    assert terminal["event_data"]["implementation_variant_id"] == "incumbent"
    assert terminal["event_data"]["lock_head_hash"] == lock["event_hash"]


def test_recovery_rejects_third_registration_and_wrong_seed_or_cap(tmp_path, monkeypatch):
    registry, benchmark, train, _ = _legacy_registry_with_recovery_amendment(tmp_path, monkeypatch)
    with pytest.raises(RegistryError, match="authorizes exactly"):
        registry.register_recovery_workload("third", "benchmark", benchmark["event_data"]["record"])
    # Tampering with a frozen registration cannot be hidden by recomputing later event hashes.
    lines = registry.path.read_bytes().splitlines()
    event = json.loads(lines[14])
    event["event_data"]["record"]["session_timeout_seconds"] = 3601
    payload = {k: event[k] for k in ("event_index", "event_type", "event_data", "prev_event_hash")}
    event["event_hash"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    lines[14] = canonical_bytes(event)
    registry.path.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(RegistryError):
        registry.events()


def _cycle2_amendment(parent_event, *, benchmark_uuid=None, training_uuid=None, **overrides):
    data = {
        "schema_id": "g008-kaggle-recovery-cycle-amendment", "schema_version": 1,
        "cycle_id": "g008-recovery-cycle-2", "created_at_utc": "2026-10-20T20:00:00Z",
        "prior_head_hash": parent_event["event_hash"], "prior_final_event_index": 19,
        "proposal_sha256": registry_module._CYCLE2_PROPOSAL_SHA256,
        "seed_plan_sha256": registry_module._RECOVERY_SEED_PLAN_SHA256,
        "retired_registration_event_indices": [14, 15],
        "recovery_registrations": [
            {"registration_id": "cycle2-benchmark", "registration_kind": "benchmark",
             "run_uuid": benchmark_uuid or str(uuid.uuid4())},
            {"registration_id": "cycle2-train", "registration_kind": "train",
             "run_uuid": training_uuid or str(uuid.uuid4())},
        ],
        "compute_caps_gpu_seconds": {"benchmark": 5600, "train": 7200,
                                      "project_experiment": 21600, "project_total": 28800},
        "project_gpu_seconds_before": 8549, "project_output_bytes_before": 30922,
        "quota_gpu_seconds_remaining_before": 99468,
        "protected_final_reserve_gpu_seconds": 7200,
        "output_caps_bytes": {"aggregate": 5_000_000_000, "per_run": 1_000_000_000},
        "output_category_reservations_bytes": {"benchmark": 1_000_000_000, "training": 1_000_000_000,
                                                "validation": 1_000_000_000, "final": 1_000_000_000},
        "kernel_id": "owner/kernel", "current_version_number": 11,
        "current_terminal_status": "ERROR",
        "current_status_evidence": {"status": "ERROR", "current_version_number": 11,
                                     "kernel_id": "owner/kernel"},
        "quota_evidence": {"remaining_gpu_seconds": 99468, "used_gpu_seconds": 8532,
                           "source": "pinned Kaggle SDK quota_view"},
        "user_authorization": {"proposal_sha256": registry_module._CYCLE2_PROPOSAL_SHA256,
                               "statement": "Authorized cycle 2 for the approved proposal.",
                               "confirmed_at_utc": "2026-10-20T20:00:00Z"},
    }
    data.update(overrides)
    return data


def _cycle2_parent(tmp_path, monkeypatch):
    registry, benchmark, _, _ = _legacy_registry_with_recovery_amendment(tmp_path, monkeypatch)
    pf_data = _preflight(benchmark)
    pf = registry.preflight_workload(pf_data)
    dispatch_data = _dispatch(pf_data)
    dispatch_data.update(preflight_event_index=pf["event_index"], preflight_event_hash=pf["event_hash"])
    dispatch = registry.dispatch_workload_push(dispatch_data)
    accepted = registry.accept_workload_push({
        "attempt_id": "attempt-1", "registration_event_index": benchmark["event_index"],
        "registration_event_hash": benchmark["event_hash"], "preflight_event_index": pf["event_index"],
        "preflight_event_hash": pf["event_hash"], "dispatch_event_index": dispatch["event_index"],
        "dispatch_event_hash": dispatch["event_hash"], "assigned_version": 11,
        "implementation_variant_id": "paired", "push_response": {"assigned_version": 11},
        "confirming_metadata": {"current_version_number": 11, "kernel_id": "owner/kernel"},
        "lock_head_hash": None,
    })
    terminal_data = _attempt(pf, dispatch, accepted_event=accepted)
    terminal_data.update(actual_gpu_seconds=125, output_bytes=4519, terminal_status="ERROR",
                         terminal_metadata={"current_version_number": 11, "kernel_id": "owner/kernel", "status": "ERROR"},
                         status_evidence=[{"status": "ERROR", "current_version_number": 11,
                                           "kernel_id": "owner/kernel"}])
    terminal = registry.resolve_workload_attempt(terminal_data)
    monkeypatch.setattr(registry_module, "_CYCLE1_FINAL_EVENT_HASH", terminal["event_hash"])
    return registry, terminal


def _append_cycle2_registrations(registry, amendment):
    amendment_event = registry.register_recovery_cycle_amendment(amendment)
    hashes = {"package_sha256": "1" * 64, "source_sha256": "2" * 64,
              "config_sha256": "3" * 64, "seed_plan_sha256": registry_module._RECOVERY_SEED_PLAN_SHA256,
              "backend_sha256": "4" * 64, "trainer_sha256": "5" * 64, "engine_sha256": "6" * 64}
    variants = [
        {"variant_id": name, "package_sha256": package * 64,
         "source_sha256": source * 64, "trainer_sha256": trainer * 64,
         "backend_sha256": backend * 64, "engine_sha256": engine * 64,
         "config_sha256": "3" * 64, "seed_plan_sha256": registry_module._RECOVERY_SEED_PLAN_SHA256}
        for name, package, source, trainer, backend, engine in (
            ("incumbent", "1", "2", "3", "4", "5"), ("candidate", "7", "8", "9", "a", "b"))
    ]
    registrations = []
    for kind, timeout in (("benchmark", 2800), ("train", 7200)):
        row = next(r for r in amendment["recovery_registrations"] if r["registration_kind"] == kind)
        record = {**hashes, "cycle_id": "g008-recovery-cycle-2", "run_uuid": row["run_uuid"],
                  "recovery_cycle_amendment_event_index": amendment_event["event_index"],
                  "recovery_cycle_amendment_event_hash": amendment_event["event_hash"],
                  "session_timeout_seconds": timeout,
                  "workload_gpu_seconds_cap": 5600 if kind == "benchmark" else 7200,
                  "workload_output_bytes_cap": 100_000_000 if kind == "benchmark" else 1_000_000_000,
                  "accelerator_count": 1, "machine_shape": "NvidiaTeslaT4", "kernel_id": "owner/kernel",
                  **({"implementation_variants": copy.deepcopy(variants)} if kind == "benchmark"
                     else {"allowed_implementation_variants": copy.deepcopy(variants)})}
        registrations.append(registry._append("registration", {
            "registration_id": row["registration_id"], "registration_kind": kind,
            "record": record, "cycle_id": "g008-recovery-cycle-2"}))
    return amendment_event, registrations


def test_cycle2_amendment_counts_event19_gpu_and_output_once(tmp_path, monkeypatch):
    registry, parent = _cycle2_parent(tmp_path, monkeypatch)
    amendment = _cycle2_amendment(parent)
    amendment_event = registry.register_recovery_cycle_amendment(amendment)
    assert amendment_event["event_index"] == 20
    assert amendment["project_gpu_seconds_before"] == 8424 + parent["event_data"]["actual_gpu_seconds"]
    assert amendment["project_output_bytes_before"] == 26403 + parent["event_data"]["output_bytes"]
    assert registry.events()[-1] == amendment_event


@pytest.mark.parametrize("field,value", [
    ("project_gpu_seconds_before", 8674),
    ("project_output_bytes_before", 35441),
    ("prior_final_event_index", 18),
    ("retired_registration_event_indices", [14]),
])
def test_cycle2_amendment_rejects_double_count_or_mutated_parent(tmp_path, monkeypatch, field, value):
    registry, parent = _cycle2_parent(tmp_path, monkeypatch)
    with pytest.raises(RegistryError):
        registry.register_recovery_cycle_amendment(_cycle2_amendment(parent, **{field: value}))
    assert registry.head[1] == 19


def test_cycle2_rejects_reused_registration_and_cycle_mutation(tmp_path, monkeypatch):
    registry, parent = _cycle2_parent(tmp_path, monkeypatch)
    amendment = _cycle2_amendment(parent)
    _, registrations = _append_cycle2_registrations(registry, amendment)
    with pytest.raises(RegistryError, match="exactly two|cycle-2"):
        registry.register("extra-cycle2", "train", registrations[1]["event_data"]["record"])

    prefix = registry.path.read_bytes().splitlines()
    event = json.loads(prefix[-1])
    event["event_data"]["cycle_id"] = "g008-recovery-cycle-1"
    payload = {k: event[k] for k in ("event_index", "event_type", "event_data", "prev_event_hash")}
    event["event_hash"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    prefix[-1] = canonical_bytes(event)
    registry.path.write_bytes(b"\n".join(prefix) + b"\n")
    with pytest.raises(RegistryError, match="cycle_id|cycle"):
        registry.events()


def test_cycle2_attempt_cannot_reference_retired_cycle1_registration(tmp_path, monkeypatch):
    registry, parent = _cycle2_parent(tmp_path, monkeypatch)
    amendment = _cycle2_amendment(parent)
    _, registrations = _append_cycle2_registrations(registry, amendment)
    data = _preflight(registrations[0], attempt_id="cycle2-attempt", timeout=2800)
    data.update(cycle_id="g008-recovery-cycle-2", expected_current_version=11,
                registration_event_index=14,
                registration_event_hash=registry.events()[14]["event_hash"])
    with pytest.raises(RegistryError, match="current-cycle|retired|registration"):
        registry.preflight_workload(data)


def _cycle3_parent(tmp_path, monkeypatch):
    registry, parent = _cycle2_parent(tmp_path, monkeypatch)
    amendment = _cycle2_amendment(parent)
    _, registrations = _append_cycle2_registrations(registry, amendment)
    benchmark = registrations[0]
    preflight_data = _preflight(benchmark, timeout=2800)
    preflight_data.update(cycle_id="g008-recovery-cycle-2", expected_current_version=11,
                          attempt_id="cycle2-benchmark-attempt")
    preflight = registry.preflight_workload(preflight_data)
    dispatch_data = _dispatch(preflight_data, attempt_id="cycle2-benchmark-attempt")
    dispatch_data.update(cycle_id="g008-recovery-cycle-2", preflight_event_index=preflight["event_index"],
                         preflight_event_hash=preflight["event_hash"])
    dispatch = registry.dispatch_workload_push(dispatch_data)
    accepted = registry.accept_workload_push({
        "attempt_id": "cycle2-benchmark-attempt", "cycle_id": "g008-recovery-cycle-2",
        "registration_event_index": benchmark["event_index"],
        "registration_event_hash": benchmark["event_hash"],
        "preflight_event_index": preflight["event_index"], "preflight_event_hash": preflight["event_hash"],
        "dispatch_event_index": dispatch["event_index"], "dispatch_event_hash": dispatch["event_hash"],
        "assigned_version": 12, "implementation_variant_id": "paired",
        "push_response": {"assigned_version": 12},
        "confirming_metadata": {"current_version_number": 12, "kernel_id": "owner/kernel"},
        "lock_head_hash": None,
    })
    terminal_data = _attempt(preflight, dispatch, accepted_event=accepted)
    terminal_data.update(cycle_id="g008-recovery-cycle-2", attempt_id="cycle2-benchmark-attempt",
                         actual_gpu_seconds=115, output_bytes=3763, terminal_status="ERROR",
                         terminal_metadata={"current_version_number": 12, "kernel_id": "owner/kernel", "status": "ERROR"},
                         status_evidence=[{"status": "ERROR", "current_version_number": 12,
                                           "kernel_id": "owner/kernel"}])
    terminal = registry.resolve_workload_attempt(terminal_data)
    assert terminal["event_index"] == 26
    monkeypatch.setattr(registry_module, "_CYCLE3_PARENT_EVENT_HASH", terminal["event_hash"])
    return registry, terminal


def _cycle3_amendment(parent, **overrides):
    proposal = registry_module._CYCLE3_PROPOSAL_SHA256
    identities = [
        {"registration_id": "cycle3-benchmark", "registration_kind": "benchmark", "run_uuid": str(uuid.uuid4())},
        {"registration_id": "cycle3-train", "registration_kind": "train", "run_uuid": str(uuid.uuid4())},
    ]
    record_templates = _cycle3_record_templates(identities)
    registrations = [{**identity, "record_sha256": registry_module.cycle3_registration_record_sha256(
        record_templates[identity["registration_kind"]])} for identity in identities]
    binding_digest = registry_module.cycle3_registration_bindings_sha256(registrations)
    context = {
        "kernel_id": "owner/kernel", "current_version_number": 12, "current_terminal_status": "ERROR",
        "current_status_evidence": {"status": "ERROR", "current_version_number": 12, "kernel_id": "owner/kernel"},
        "quota_gpu_seconds_remaining_before": 99468,
        "quota_evidence": {"remaining_gpu_seconds": 99468, "used_gpu_seconds": 8532,
                           "source": "pinned Kaggle SDK quota_view"},
    }
    context_digest = registry_module.cycle3_preflight_context_sha256(context)
    data = {
        "schema_id": "g008-kaggle-recovery-cycle-amendment", "schema_version": 2,
        "cycle_id": "g008-recovery-cycle-3", "created_at_utc": "2026-10-20T22:00:00Z",
        "prior_head_hash": parent["event_hash"], "prior_final_event_index": 26,
        "proposal_sha256": proposal, "retired_registration_event_indices": [21, 22],
        "recovery_registrations": registrations,
        "registration_bindings_sha256": binding_digest,
        "preflight_context_sha256": context_digest,
        "compute_caps_gpu_seconds": {"benchmark": 5600, "train": 7200,
                                      "project_experiment": 21600, "project_total": 28800},
        "project_gpu_seconds_before": 8664, "project_output_bytes_before": 34685,
        "quota_gpu_seconds_remaining_before": 99468, "protected_final_reserve_gpu_seconds": 7200,
        "output_caps_bytes": {"aggregate": 5_000_000_000, "per_run": 1_000_000_000},
        "output_category_reservations_bytes": {"benchmark": 1_000_000_000, "training": 1_000_000_000,
                                                "validation": 1_000_000_000, "final": 1_000_000_000},
        "kernel_id": context["kernel_id"], "current_version_number": context["current_version_number"],
        "current_terminal_status": context["current_terminal_status"],
        "current_status_evidence": context["current_status_evidence"],
        "quota_evidence": context["quota_evidence"],
        "quota_gpu_seconds_remaining_before": context["quota_gpu_seconds_remaining_before"],
        "user_authorization": {"proposal_sha256": proposal,
                               "registration_bindings_sha256": binding_digest,
                               "preflight_context_sha256": context_digest,
                               "statement": registry_module.cycle3_authorization_statement(
                                   proposal, binding_digest, context_digest),
                               "confirmed_at_utc": "2026-10-20T22:00:00Z"},
    }
    data.update(overrides)
    return data


def _cycle3_record_templates(registration_rows):
    hashes = {"package_sha256": "1" * 64, "source_sha256": "2" * 64,
              "config_sha256": "3" * 64, "seed_plan_sha256": registry_module._RECOVERY_SEED_PLAN_SHA256,
              "backend_sha256": "4" * 64, "trainer_sha256": "5" * 64, "engine_sha256": "6" * 64}
    variants = [
        {"variant_id": name, "package_sha256": package * 64, "source_sha256": source * 64,
         "trainer_sha256": trainer * 64, "backend_sha256": backend * 64,
         "engine_sha256": engine * 64, "config_sha256": "3" * 64,
         "seed_plan_sha256": registry_module._RECOVERY_SEED_PLAN_SHA256}
        for name, package, source, trainer, backend, engine in (
            ("incumbent", "1", "2", "3", "4", "5"), ("candidate", "7", "8", "9", "a", "b"))
    ]
    records = {}
    for kind, timeout in (("benchmark", 2800), ("train", 7200)):
        row = next(r for r in registration_rows if r["registration_kind"] == kind)
        record = {**hashes, "cycle_id": "g008-recovery-cycle-3", "run_uuid": row["run_uuid"],
                  "session_timeout_seconds": timeout,
                  "workload_gpu_seconds_cap": 5600 if kind == "benchmark" else 7200,
                  "workload_output_bytes_cap": 100_000_000 if kind == "benchmark" else 1_000_000_000,
                  "accelerator_count": 1, "machine_shape": "NvidiaTeslaT4", "kernel_id": "owner/kernel",
                  **({"implementation_variants": copy.deepcopy(variants)} if kind == "benchmark"
                     else {"allowed_implementation_variants": copy.deepcopy(variants)})}
        records[kind] = record
    return records


def _cycle3_workload_candidates(amendment):
    templates = _cycle3_record_templates(amendment["recovery_registrations"])
    workloads = []
    for row in amendment["recovery_registrations"]:
        kind = row["registration_kind"]
        workloads.append({"registration_id": row["registration_id"], "registration_kind": kind,
                          "record": templates[kind]})
    return workloads


def _refresh_cycle3_bindings(amendment, workloads):
    by_id = {workload["registration_id"]: workload for workload in workloads}
    for row in amendment["recovery_registrations"]:
        row["record_sha256"] = registry_module.cycle3_registration_record_sha256(
            by_id[row["registration_id"]]["record"])
    digest = registry_module.cycle3_registration_bindings_sha256(amendment["recovery_registrations"])
    amendment["registration_bindings_sha256"] = digest
    authorization = amendment["user_authorization"]
    authorization["registration_bindings_sha256"] = digest
    authorization["statement"] = registry_module.cycle3_authorization_statement(
        amendment["proposal_sha256"], digest, amendment["preflight_context_sha256"])


def _append_cycle3_registrations(registry, amendment):
    events = registry.register_recovery_cycle3_batch(amendment, _cycle3_workload_candidates(amendment))
    return events[0], events[1:]


def _cycle3_successful_benchmark(tmp_path, monkeypatch, *, attest=True, record_artifact=True, close_lock=True):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    amendment = _cycle3_amendment(parent)
    _, registrations = _append_cycle3_registrations(registry, amendment)
    benchmark = registrations[0]
    attempt_id = "cycle3-benchmark-attempt"
    preflight_data = _preflight(benchmark, attempt_id=attempt_id, timeout=2800)
    preflight_data.update(cycle_id="g008-recovery-cycle-3", expected_current_version=12)
    preflight = registry.preflight_workload(preflight_data)
    dispatch_data = _dispatch(preflight_data, attempt_id=attempt_id)
    dispatch_data.update(cycle_id="g008-recovery-cycle-3", preflight_event_index=preflight["event_index"],
                         preflight_event_hash=preflight["event_hash"])
    dispatch = registry.dispatch_workload_push(dispatch_data)
    accepted = registry.accept_workload_push({
        "attempt_id": attempt_id, "cycle_id": "g008-recovery-cycle-3",
        "registration_event_index": benchmark["event_index"], "registration_event_hash": benchmark["event_hash"],
        "preflight_event_index": preflight["event_index"], "preflight_event_hash": preflight["event_hash"],
        "dispatch_event_index": dispatch["event_index"], "dispatch_event_hash": dispatch["event_hash"],
        "assigned_version": 13, "implementation_variant_id": "paired",
        "push_response": {"assigned_version": 13},
        "confirming_metadata": {"current_version_number": 13, "kernel_id": "owner/kernel"},
        "lock_head_hash": None,
    })
    terminal_data = _attempt(preflight, dispatch, accepted_event=accepted)
    terminal_data.update(cycle_id="g008-recovery-cycle-3", actual_gpu_seconds=10, output_bytes=5,
                         terminal_status="COMPLETE",
                         terminal_metadata={"current_version_number": 13, "kernel_id": "owner/kernel", "status": "COMPLETE"},
                         status_evidence=[{"status": "COMPLETE", "current_version_number": 13,
                                           "kernel_id": "owner/kernel"}])
    terminal = registry.resolve_workload_attempt(terminal_data)
    artifact = None
    if record_artifact:
        artifact = registry.register_benchmark_artifact(
            "a" * 64, benchmark["event_index"], benchmark["event_hash"],
            attempt_event_index=terminal["event_index"], attempt_event_hash=terminal["event_hash"],
            report_sha256="b" * 64, selection_evidence=_selection_evidence())
    lock = registry.close_registrations() if record_artifact and close_lock else None
    attestation = registry.attest_benchmark(artifact["event_index"]) if attest and lock is not None else None
    return registry, registrations, terminal, artifact, lock, attestation


def _cycle3_failed_attempt_registry(tmp_path, monkeypatch, *, actual_gpu_seconds=10):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    _, registrations = _append_cycle3_registrations(registry, _cycle3_amendment(parent))
    benchmark = registrations[0]
    attempt_id = "cycle3-benchmark-attempt"
    preflight_data = _preflight(benchmark, attempt_id=attempt_id, timeout=2800)
    preflight_data.update(cycle_id="g008-recovery-cycle-3", expected_current_version=12)
    preflight = registry.preflight_workload(preflight_data)
    dispatch_data = _dispatch(preflight_data, attempt_id=attempt_id)
    dispatch_data.update(cycle_id="g008-recovery-cycle-3", preflight_event_index=preflight["event_index"],
                         preflight_event_hash=preflight["event_hash"])
    dispatch = registry.dispatch_workload_push(dispatch_data)
    accepted = registry.accept_workload_push({
        "attempt_id": attempt_id, "cycle_id": "g008-recovery-cycle-3",
        "registration_event_index": benchmark["event_index"], "registration_event_hash": benchmark["event_hash"],
        "preflight_event_index": preflight["event_index"], "preflight_event_hash": preflight["event_hash"],
        "dispatch_event_index": dispatch["event_index"], "dispatch_event_hash": dispatch["event_hash"],
        "assigned_version": 13, "implementation_variant_id": "paired",
        "push_response": {"assigned_version": 13},
        "confirming_metadata": {"current_version_number": 13, "kernel_id": "owner/kernel"},
        "lock_head_hash": None,
    })
    terminal_data = _attempt(preflight, dispatch, accepted_event=accepted)
    terminal_data.update(cycle_id="g008-recovery-cycle-3", actual_gpu_seconds=actual_gpu_seconds, output_bytes=5,
                         terminal_status="ERROR",
                         terminal_metadata={"current_version_number": 13, "kernel_id": "owner/kernel", "status": "ERROR"},
                         status_evidence=[{"status": "ERROR", "current_version_number": 13,
                                           "kernel_id": "owner/kernel"}])
    terminal = registry.resolve_workload_attempt(terminal_data)
    monkeypatch.setattr(registry_module, "_CYCLE3_RETRY_PARENT_EVENT_HASH", terminal["event_hash"])
    monkeypatch.setattr(registry_module, "_CYCLE3_RETRY_PROPOSAL_SHA256", "c" * 64)
    return registry, registrations, terminal


def _cycle3_retry_workload(registry, *, registration_id="cycle3-benchmark-retry", run_uuid=None,
                           package_sha256="d" * 64, source_sha256="e" * 64):
    run_uuid = run_uuid or str(uuid.uuid4())
    rows = [{"registration_id": registration_id, "registration_kind": "benchmark", "run_uuid": run_uuid},
            {"registration_id": "cycle3-train-template", "registration_kind": "train",
             "run_uuid": str(uuid.uuid4())}]
    record = _cycle3_record_templates(rows)["benchmark"]
    record.update(package_sha256=package_sha256, source_sha256=source_sha256)
    record["run_uuid"] = run_uuid
    return {"registration_id": registration_id, "registration_kind": "benchmark", "record": record}


def _cycle3_retry_amendment(registry, workload):
    events = registry.events()
    record = workload["record"]
    record_sha = registry_module.cycle3_retry_registration_record_sha256(record)
    context = {
        "kernel_id": "owner/kernel", "current_version_number": 13, "current_terminal_status": "ERROR",
        "current_status_evidence": {"status": "ERROR", "current_version_number": 13, "kernel_id": "owner/kernel"},
        "quota_gpu_seconds_remaining_before": 99468,
        "quota_evidence": {"remaining_gpu_seconds": 99468, "used_gpu_seconds": 8532,
                           "source": "pinned Kaggle SDK quota_view"},
    }
    context_sha = registry_module.cycle3_retry_context_sha256(context)
    proposal = registry_module._CYCLE3_RETRY_PROPOSAL_SHA256
    amendment = {
        "schema_id": "g008-kaggle-recovery-cycle-retry-amendment", "schema_version": 1,
        "cycle_id": "g008-recovery-cycle-3-retry", "created_at_utc": "2026-10-21T00:00:00Z",
        "prior_head_hash": events[33]["event_hash"], "prior_final_event_index": 33,
        "cycle3_amendment_event_index": 27, "cycle3_amendment_event_hash": events[27]["event_hash"],
        "proposal_sha256": proposal, "retry_registration_id": workload["registration_id"],
        "registration_kind": "benchmark", "retry_run_uuid": record["run_uuid"],
        "retry_record_sha256": record_sha,
        "registration_binding_sha256": registry_module.cycle3_retry_binding_sha256(
            workload["registration_id"], "benchmark", record["run_uuid"], record_sha),
        **context, "preflight_context_sha256": context_sha,
        "user_authorization": {
            "proposal_sha256": proposal, "record_sha256": record_sha,
            "preflight_context_sha256": context_sha,
            "statement": registry_module.cycle3_retry_authorization_statement(proposal, record_sha, context_sha),
            "confirmed_at_utc": "2026-10-21T00:00:00Z",
        },
    }
    return amendment


def _cycle3_complete_retry(registry, retry_registration, *, status="COMPLETE", actual=10,
                           attempt_id="cycle3-benchmark-retry-attempt", version=14):
    pf_data = _preflight(retry_registration, attempt_id=attempt_id, timeout=2800)
    pf_data.update(cycle_id="g008-recovery-cycle-3", expected_current_version=version - 1)
    pf = registry.preflight_workload(pf_data)
    dispatch_data = _dispatch(pf_data, attempt_id=attempt_id)
    dispatch_data.update(cycle_id="g008-recovery-cycle-3", preflight_event_index=pf["event_index"],
                         preflight_event_hash=pf["event_hash"])
    dispatch = registry.dispatch_workload_push(dispatch_data)
    accepted = registry.accept_workload_push({
        "attempt_id": attempt_id, "cycle_id": "g008-recovery-cycle-3",
        "registration_event_index": retry_registration["event_index"],
        "registration_event_hash": retry_registration["event_hash"],
        "preflight_event_index": pf["event_index"], "preflight_event_hash": pf["event_hash"],
        "dispatch_event_index": dispatch["event_index"], "dispatch_event_hash": dispatch["event_hash"],
        "assigned_version": version, "implementation_variant_id": "paired",
        "push_response": {"assigned_version": version},
        "confirming_metadata": {"current_version_number": version, "kernel_id": "owner/kernel"},
        "lock_head_hash": None,
    })
    terminal_data = _attempt(pf, dispatch, accepted_event=accepted)
    terminal_data.update(cycle_id="g008-recovery-cycle-3", actual_gpu_seconds=actual, output_bytes=5,
                         terminal_status=status,
                         terminal_metadata={"current_version_number": version, "kernel_id": "owner/kernel", "status": status},
                         status_evidence=[{"status": status, "current_version_number": version,
                                           "kernel_id": "owner/kernel"}])
    terminal = registry.resolve_workload_attempt(terminal_data)
    return terminal


def _append_cycle3_retry(registry, workload):
    amendment = _cycle3_retry_amendment(registry, workload)
    events = registry.register_recovery_cycle3_retry_batch(amendment, workload)
    return events[0], events[1]


def test_cycle3_appends_only_after_reconciled_cycle2_and_replays(tmp_path, monkeypatch):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    amendment = _cycle3_amendment(parent)
    registry.path.chmod(0o640)
    before_stat = registry.path.stat()
    event, registrations = _append_cycle3_registrations(registry, amendment)
    assert event["event_index"] == 27
    assert [row["event_index"] for row in registrations] == [28, 29]
    assert registry.events()[-1] == registrations[-1]
    after_stat = registry.path.stat()
    assert stat.S_IMODE(after_stat.st_mode) == stat.S_IMODE(before_stat.st_mode)
    assert (after_stat.st_uid, after_stat.st_gid) == (before_stat.st_uid, before_stat.st_gid)
    assert amendment["project_gpu_seconds_before"] == 8549 + 115
    assert amendment["project_output_bytes_before"] == 30922 + 3763


@pytest.mark.parametrize("field,value", [
    ("prior_final_event_index", 25),
    ("retired_registration_event_indices", [21]),
    ("project_gpu_seconds_before", 8665),
    ("project_output_bytes_before", 34686),
    ("current_version_number", 11),
    ("current_terminal_status", "COMPLETE"),
    ("compute_caps_gpu_seconds", {"benchmark": 5600, "train": 7200, "project_experiment": 21600, "project_total": 30000}),
])
def test_cycle3_rejects_parent_budget_status_and_retirement_mutations(tmp_path, monkeypatch, field, value):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    with pytest.raises(RegistryError):
        amendment = _cycle3_amendment(parent, **{field: value})
        registry.register_recovery_cycle3_batch(amendment, _cycle3_workload_candidates(amendment))
    assert registry.head[1] == 26


def test_cycle3_requires_proposal_bound_user_authorization(tmp_path, monkeypatch):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    amendment = _cycle3_amendment(parent)
    amendment["user_authorization"]["proposal_sha256"] = "1" * 64
    with pytest.raises(RegistryError, match="authorization"):
        registry.register_recovery_cycle3_batch(amendment, _cycle3_workload_candidates(amendment))


def test_cycle3_authorization_rejects_registration_binding_digest_drift(tmp_path, monkeypatch):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    amendment = _cycle3_amendment(parent)
    amendment["user_authorization"]["registration_bindings_sha256"] = "1" * 64
    with pytest.raises(RegistryError, match="authorization"):
        registry.register_recovery_cycle3_batch(amendment, _cycle3_workload_candidates(amendment))


@pytest.mark.parametrize("field,value", [
    ("current_version_number", 13),
    ("quota_gpu_seconds_remaining_before", 12345),
])
def test_cycle3_authorization_rejects_live_kernel_or_quota_context_drift(tmp_path, monkeypatch, field, value):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    amendment = _cycle3_amendment(parent)
    amendment[field] = value
    with pytest.raises(RegistryError, match="preflight context digest"):
        registry.register_recovery_cycle3_batch(amendment, _cycle3_workload_candidates(amendment))


def test_cycle3_authorization_rejects_context_digest_mismatch(tmp_path, monkeypatch):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    amendment = _cycle3_amendment(parent)
    amendment["user_authorization"]["preflight_context_sha256"] = "1" * 64
    with pytest.raises(RegistryError, match="authorization"):
        registry.register_recovery_cycle3_batch(amendment, _cycle3_workload_candidates(amendment))


def test_cycle3_registration_rejects_package_record_drift(tmp_path, monkeypatch):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    amendment = _cycle3_amendment(parent)
    workloads = _cycle3_workload_candidates(amendment)
    record = workloads[0]["record"]
    record["package_sha256"] = "f" * 64
    with pytest.raises(RegistryError, match="user-authorized binding"):
        registry.register_recovery_cycle3_batch(amendment, workloads)
    assert registry.head[1] == 26


def test_cycle3_batch_rejects_crossed_variant_hashes_without_partial_append(tmp_path, monkeypatch):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    amendment = _cycle3_amendment(parent)
    workloads = _cycle3_workload_candidates(amendment)
    train = next(workload for workload in workloads if workload["registration_kind"] == "train")
    candidate = next(v for v in train["record"]["allowed_implementation_variants"]
                     if v["variant_id"] == "candidate")
    candidate["source_sha256"] = "f" * 64
    _refresh_cycle3_bindings(amendment, workloads)
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="shared variant hashes"):
        registry.register_recovery_cycle3_batch(amendment, workloads)
    assert registry.path.read_bytes() == before
    assert registry.head[1] == 26


@pytest.mark.parametrize("reuse", ["registration_id", "run_uuid"])
def test_cycle3_batch_rejects_historical_identity_reuse_without_partial_append(tmp_path, monkeypatch, reuse):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    amendment = _cycle3_amendment(parent)
    workloads = _cycle3_workload_candidates(amendment)
    first_row, first_workload = amendment["recovery_registrations"][1], workloads[1]
    if reuse == "registration_id":
        first_row["registration_id"] = registry.events()[0]["event_data"]["registration_id"]
        first_workload["registration_id"] = first_row["registration_id"]
    else:
        historical_uuid = registry.events()[14]["event_data"]["record"]["run_uuid"]
        first_row["run_uuid"] = historical_uuid
        first_workload["record"]["run_uuid"] = historical_uuid
    _refresh_cycle3_bindings(amendment, workloads)
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="reuses a retired identity"):
        registry.register_recovery_cycle3_batch(amendment, workloads)
    assert registry.path.read_bytes() == before
    assert registry.head[1] == 26


@pytest.mark.parametrize("failure_point", ["write", "replace"])
def test_cycle3_batch_failure_preserves_old_registry_and_cleans_temp(tmp_path, monkeypatch, failure_point):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    amendment = _cycle3_amendment(parent)
    workloads = _cycle3_workload_candidates(amendment)
    old_bytes = registry.path.read_bytes()
    if failure_point == "write":
        def fail_write(_fd, _data):
            raise OSError("injected temp write failure")
        monkeypatch.setattr(registry_module.os, "write", fail_write)
    else:
        def fail_replace(_source, _destination):
            raise OSError("injected atomic replace failure")
        monkeypatch.setattr(registry_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected"):
        registry.register_recovery_cycle3_batch(amendment, workloads)
    assert registry.path.read_bytes() == old_bytes
    assert registry.head[1] == 26
    assert list(registry.path.parent.glob(f".{registry.path.name}.*.tmp")) == []


def test_cycle3_preassignment_rejection_requires_exact_no_assignment_proof(tmp_path, monkeypatch):
    registry, parent = _cycle3_parent(tmp_path, monkeypatch)
    _, registrations = _append_cycle3_registrations(registry, _cycle3_amendment(parent))
    benchmark = registrations[0]
    pf_data = _preflight(benchmark, timeout=2800)
    pf_data.update(cycle_id="g008-recovery-cycle-3", expected_current_version=12)
    pf = registry.preflight_workload(pf_data)
    ds_data = _dispatch(pf_data)
    ds_data.update(cycle_id="g008-recovery-cycle-3", preflight_event_index=pf["event_index"],
                   preflight_event_hash=pf["event_hash"])
    ds = registry.dispatch_workload_push(ds_data)
    rejected = _attempt(pf, ds, outcome="pre_assignment_rejected")
    rejected.update(cycle_id="g008-recovery-cycle-3", rejection_evidence={"server_rejected_before_assignment": False},
                    no_assignment_metadata={"pre_push_current_version": 12, "post_push_current_version": 12,
                                            "no_version_assigned": True})
    with pytest.raises(RegistryError, match="no-assignment proof"):
        registry.resolve_workload_attempt(rejected)


def test_cycle3_training_requires_benchmark_attestation(tmp_path, monkeypatch):
    registry, registrations, _, _, lock, _ = _cycle3_successful_benchmark(tmp_path, monkeypatch, attest=False)
    train = registrations[1]
    pf_data = _preflight(train, attempt_id="cycle3-train-attempt", timeout=7200, variant_id="incumbent",
                         lock_hash=lock["event_hash"])
    pf_data.update(cycle_id="g008-recovery-cycle-3", expected_current_version=13,
                   lock_head_hash=lock["event_hash"])
    with pytest.raises(RegistryError, match="attestation"):
        registry.preflight_workload(pf_data)


def test_cycle3_artifact_must_bind_successful_benchmark_registration(tmp_path, monkeypatch):
    registry, registrations, _, _, _, _ = _cycle3_successful_benchmark(tmp_path, monkeypatch, record_artifact=False)
    with pytest.raises(RegistryError, match="successful terminal benchmark"):
        registry.register_benchmark_artifact(
            "c" * 64, 21, registry.events()[21]["event_hash"],
            attempt_event_index=26, attempt_event_hash=registry.events()[26]["event_hash"],
            report_sha256="d" * 64, selection_evidence=_selection_evidence())


def test_cycle3_registration_lock_must_pin_artifact_exactly(tmp_path, monkeypatch):
    registry, _, _, artifact, _, _ = _cycle3_successful_benchmark(
        tmp_path, monkeypatch, attest=False, close_lock=False)
    with pytest.raises(RegistryError, match="pin exact benchmark artifact"):
        registry._append("registration_closed", {"pre_lock_head_hash": "0" * 64,
                                                  "pre_lock_final_event_index": artifact["event_index"]})


def test_cycle3_retry_preserves_original_prefix_and_binds_one_fresh_benchmark(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    original = registry.path.read_bytes()
    workload = _cycle3_retry_workload(registry)
    amendment, registration = _append_cycle3_retry(registry, workload)
    events = registry.events()
    assert events[:34] == [json.loads(line) for line in original.splitlines()]
    assert (amendment["event_index"], registration["event_index"]) == (34, 35)
    assert amendment["event_type"] == "recovery_cycle_retry_amendment_v1"
    assert registration["event_data"]["registration_kind"] == "benchmark"
    assert registration["event_data"]["record"]["cycle3_retry_amendment_event_hash"] == amendment["event_hash"]
    assert registration["event_data"]["record"]["recovery_cycle_amendment_event_hash"] == events[27]["event_hash"]
    assert registration["event_data"]["record"]["package_sha256"] == "d" * 64
    assert registration["event_data"]["record"]["source_sha256"] == "e" * 64
    assert registry_module.cycle3_retry_binding_sha256(
        workload["registration_id"], "benchmark", workload["record"]["run_uuid"],
        registry_module.cycle3_retry_registration_record_sha256(workload["record"])) == amendment["event_data"]["registration_binding_sha256"]


def test_cycle3_retry_rejects_changed_parent_and_duplicate_continuation(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    workload = _cycle3_retry_workload(registry)
    amendment = _cycle3_retry_amendment(registry, workload)
    amendment["prior_head_hash"] = "f" * 64
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="exact terminal event 33|parent|failed attempt"):
        registry.register_recovery_cycle3_retry_batch(amendment, workload)
    assert registry.path.read_bytes() == before
    amendment = _cycle3_retry_amendment(registry, workload)
    registry.register_recovery_cycle3_retry_batch(amendment, workload)
    with pytest.raises(RegistryError, match="already registered|exact terminal event 33"):
        registry.register_recovery_cycle3_retry_batch(amendment, workload)


def test_cycle3_retry_variant_source_must_match_registered_training_source(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    workload = _cycle3_retry_workload(registry)
    candidate = next(row for row in workload["record"]["implementation_variants"]
                     if row["variant_id"] == "candidate")
    candidate["source_sha256"] = "f" * 64
    amendment = _cycle3_retry_amendment(registry, workload)
    with pytest.raises(RegistryError, match="training source/component hashes"):
        registry.register_recovery_cycle3_retry_batch(amendment, workload)
    assert registry.head[1] == 33


def test_cycle3_retry_error_cannot_create_artifact_or_training(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    terminal = _cycle3_complete_retry(registry, registration, status="ERROR")
    with pytest.raises(RegistryError, match="COMPLETE benchmark attempt"):
        registry.register_benchmark_artifact(
            "a" * 64, registration["event_index"], registration["event_hash"],
            attempt_event_index=terminal["event_index"], attempt_event_hash=terminal["event_hash"],
            report_sha256="b" * 64, selection_evidence=_selection_evidence())
    train = registry.events()[29]
    train_pf = _preflight(train, attempt_id="cycle3-train-after-error", timeout=7200,
                          variant_id="incumbent")
    train_pf.update(cycle_id="g008-recovery-cycle-3", expected_current_version=14)
    with pytest.raises(RegistryError, match="attestation|retry benchmark attestation"):
        registry.preflight_workload(train_pf)


def test_cycle3_retry_reconciles_proven_preassignment_rejection_without_version_change(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    attempt_id = "cycle3-benchmark-retry-rejected"
    pf_data = _preflight(registration, attempt_id=attempt_id, timeout=2800)
    pf_data.update(cycle_id="g008-recovery-cycle-3", expected_current_version=13)
    pf = registry.preflight_workload(pf_data)
    dispatch_data = _dispatch(pf_data, attempt_id=attempt_id)
    dispatch_data.update(cycle_id="g008-recovery-cycle-3", preflight_event_index=pf["event_index"],
                         preflight_event_hash=pf["event_hash"])
    dispatch = registry.dispatch_workload_push(dispatch_data)
    rejected = _attempt(pf, dispatch, outcome="pre_assignment_rejected")
    rejected.update(cycle_id="g008-recovery-cycle-3", actual_gpu_seconds=0, output_bytes=0,
                    rejection_evidence={"server_rejected_before_assignment": True},
                    no_assignment_metadata={"pre_push_current_version": 13, "post_push_current_version": 13,
                                            "no_version_assigned": True})
    terminal = registry.resolve_workload_attempt(rejected)
    assert terminal["event_data"]["outcome"] == "pre_assignment_rejected"
    assert registry.events()[-1]["event_hash"] == terminal["event_hash"]


@pytest.mark.parametrize(("phase", "field"), [
    ("dispatch", "preflight_event_hash"),
    ("dispatch", "package_sha256"),
    ("accept", "dispatch_event_hash"),
    ("accept", "registration_event_hash"),
    ("terminal", "accepted_event_hash"),
    ("terminal", "dispatch_event_hash"),
])
def test_cycle3_retry_rejects_lifecycle_receipt_drift(tmp_path, monkeypatch, phase, field):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    attempt_id = f"cycle3-benchmark-retry-{phase}-{field}"
    pf_data = _preflight(registration, attempt_id=attempt_id, timeout=2800)
    pf_data.update(cycle_id="g008-recovery-cycle-3", expected_current_version=13)
    pf = registry.preflight_workload(pf_data)
    dispatch_data = _dispatch(pf_data, attempt_id=attempt_id)
    dispatch_data.update(cycle_id="g008-recovery-cycle-3", preflight_event_index=pf["event_index"],
                         preflight_event_hash=pf["event_hash"])
    if phase == "dispatch":
        dispatch_data[field] = "f" * 64 if field.endswith("hash") else "f" * 64
        with pytest.raises(RegistryError, match="dispatch"):
            registry.dispatch_workload_push(dispatch_data)
        return
    dispatch = registry.dispatch_workload_push(dispatch_data)
    accepted_data = {
        "attempt_id": attempt_id, "cycle_id": "g008-recovery-cycle-3",
        "registration_event_index": registration["event_index"],
        "registration_event_hash": registration["event_hash"],
        "preflight_event_index": pf["event_index"], "preflight_event_hash": pf["event_hash"],
        "dispatch_event_index": dispatch["event_index"], "dispatch_event_hash": dispatch["event_hash"],
        "assigned_version": 14, "implementation_variant_id": "paired",
        "push_response": {"assigned_version": 14},
        "confirming_metadata": {"current_version_number": 14, "kernel_id": "owner/kernel"},
        "lock_head_hash": None,
    }
    if phase == "accept":
        accepted_data[field] = "f" * 64
        with pytest.raises(RegistryError, match="acceptance"):
            registry.accept_workload_push(accepted_data)
        return
    accepted = registry.accept_workload_push(accepted_data)
    terminal_data = _attempt(pf, dispatch, accepted_event=accepted)
    terminal_data.update(cycle_id="g008-recovery-cycle-3", actual_gpu_seconds=10, output_bytes=5,
                         terminal_status="COMPLETE",
                         terminal_metadata={"current_version_number": 14, "kernel_id": "owner/kernel", "status": "COMPLETE"},
                         status_evidence=[{"status": "COMPLETE", "current_version_number": 14,
                                           "kernel_id": "owner/kernel"}])
    terminal_data[field] = "f" * 64
    with pytest.raises(RegistryError, match="terminal event|terminal result"):
        registry.resolve_workload_attempt(terminal_data)


def test_cycle3_retry_complete_artifact_lock_attestation_then_original_training(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    terminal = _cycle3_complete_retry(registry, registration)
    artifact = registry.register_benchmark_artifact(
        "a" * 64, registration["event_index"], registration["event_hash"],
        attempt_event_index=terminal["event_index"], attempt_event_hash=terminal["event_hash"],
        report_sha256="b" * 64, selection_evidence=_selection_evidence())
    lock = registry.close_registrations()
    attestation = registry.attest_benchmark(artifact["event_index"])
    train = registry.events()[29]
    train_pf = _preflight(train, attempt_id="cycle3-train-after-retry", timeout=7200,
                          variant_id="incumbent", lock_hash=lock["event_hash"])
    train_pf.update(cycle_id="g008-recovery-cycle-3", expected_current_version=14,
                    lock_head_hash=lock["event_hash"])
    recorded = registry.preflight_workload(train_pf)
    assert (artifact["event_index"], lock["event_index"], attestation["event_index"]) == (40, 41, 42)
    assert recorded["event_index"] == 43


def _reject_cycle3_artifact(registry, terminal):
    return registry.reject_benchmark_artifact_validation(
        terminal["event_index"], terminal["event_hash"], artifact_sha256=None, output_sha256="f" * 64,
        validation_codes=["visible_gpu_count_mismatch"], validator_id="test-validator/1",
        created_at_utc="2026-10-21T00:00:00Z")


def test_cycle3_retry_allows_multiple_budgeted_attempts_and_keeps_valid_final_adjacency(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    prefix = [json.loads(line) for line in registry.path.read_bytes().splitlines()]
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))

    first = _cycle3_complete_retry(registry, registration, attempt_id="retry-one", version=14)
    rejection = _reject_cycle3_artifact(registry, first)
    assert rejection["event_data"]["validation_codes"] == ["visible_gpu_count_mismatch"]
    registry.authorize_cycle3_retry_continuation("retry-one", created_at_utc="2026-10-21T00:01:00Z")

    second = _cycle3_complete_retry(registry, registration, status="ERROR", attempt_id="retry-two", version=15)
    registry.authorize_cycle3_retry_continuation("retry-two", created_at_utc="2026-10-21T00:02:00Z")

    final = _cycle3_complete_retry(registry, registration, attempt_id="retry-three", version=16)
    artifact = registry.register_benchmark_artifact(
        "a" * 64, registration["event_index"], registration["event_hash"],
        attempt_event_index=final["event_index"], attempt_event_hash=final["event_hash"],
        report_sha256="b" * 64, selection_evidence=_selection_evidence())
    lock = registry.close_registrations()
    attestation = registry.attest_benchmark(artifact["event_index"])
    events = registry.events()
    assert events[:34] == prefix
    assert [events[index]["event_type"] for index in (artifact["event_index"], lock["event_index"],
                                                        attestation["event_index"])] == [
        "benchmark_artifact", "registration_closed", "benchmark_attestation"]
    assert (lock["event_index"], attestation["event_index"]) == (artifact["event_index"] + 1,
                                                                  artifact["event_index"] + 2)


def test_cycle3_retry_pending_attempt_blocks_continuation_and_new_events(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    pf_data = _preflight(registration, attempt_id="pending-retry", timeout=2800)
    pf_data.update(cycle_id="g008-recovery-cycle-3", expected_current_version=13)
    registry.preflight_workload(pf_data)
    with pytest.raises(RegistryError, match="terminal|exact prior|current head"):
        registry.authorize_cycle3_retry_continuation("pending-retry", created_at_utc="2026-10-21T00:00:00Z")
    with pytest.raises(RegistryError, match="exact prior benchmark attempt"):
        registry.authorize_cycle3_retry_continuation("cycle3-benchmark-retry-attempt",
                                                      created_at_utc="2026-10-21T00:01:00Z")


def test_cycle3_exploratory_observation_advances_cursor_without_promoting_run(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    terminal = _cycle3_complete_retry(registry, registration, attempt_id="registered-retry", version=14)
    _reject_cycle3_artifact(registry, terminal)
    policy = registry.authorize_cycle3_retry_continuation(
        "registered-retry", created_at_utc="2026-10-21T00:01:00Z")

    observation = registry.record_exploratory_kernel_observation(
        version_number=15, run_uuid="b9941b1a-de48-4274-bafa-0d750ca40520",
        terminal_status="COMPLETE", artifact_sha256={"output_manifest": "a" * 64, "report": "b" * 64},
        actual_gpu_seconds=10, output_bytes=5,
        created_at_utc="2026-10-21T00:02:00Z")
    cursor = registry.observed_kernel_cursor("owner/kernel")

    assert observation["event_type"] == "exploratory_kernel_observation_v1"
    assert observation["event_data"]["continuation_event_hash"] == policy["event_hash"]
    assert observation["event_data"]["exploratory"] is True
    assert observation["event_data"]["promotable"] is False
    assert cursor == {
        "kernel_id": "owner/kernel", "current_version_number": 15,
        "observation_event_index": observation["event_index"],
        "observation_event_hash": observation["event_hash"],
        "run_uuid": "b9941b1a-de48-4274-bafa-0d750ca40520", "promotable": False,
    }
    assert registry.observed_kernel_cursor("other/kernel") is None
    assert not any(event["event_type"] == "workload_attempt"
                   and event["event_data"].get("run_uuid") == cursor["run_uuid"]
                   for event in registry.events())


def test_cycle3_exploratory_observation_consumes_one_shot_continuation_and_validates_hashes(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    terminal = _cycle3_complete_retry(registry, registration, attempt_id="registered-retry", version=14)
    _reject_cycle3_artifact(registry, terminal)
    registry.authorize_cycle3_retry_continuation(
        "registered-retry", created_at_utc="2026-10-21T00:01:00Z")
    before = registry.path.read_bytes()

    with pytest.raises(RegistryError, match="artifact hashes"):
        registry.record_exploratory_kernel_observation(
            version_number=15, run_uuid="b9941b1a-de48-4274-bafa-0d750ca40520",
            terminal_status="COMPLETE", artifact_sha256={"report": "not-a-sha256"},
            actual_gpu_seconds=10, output_bytes=5,
            created_at_utc="2026-10-21T00:02:00Z")
    assert registry.path.read_bytes() == before

    registry.record_exploratory_kernel_observation(
        version_number=15, run_uuid="b9941b1a-de48-4274-bafa-0d750ca40520",
        terminal_status="COMPLETE", artifact_sha256={"report": "b" * 64},
        actual_gpu_seconds=10, output_bytes=5,
        created_at_utc="2026-10-21T00:02:00Z")
    pf_data = _preflight(registration, attempt_id="must-not-be-authorized-by-exploration", timeout=2800)
    pf_data.update(cycle_id="g008-recovery-cycle-3", expected_current_version=15)
    with pytest.raises(RegistryError, match="continuation|policy amendment"):
        registry.preflight_workload(pf_data)


def test_cycle3_retry_rebase_authorizes_one_exact_fresh_attempt_from_observed_cursor(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    terminal = _cycle3_complete_retry(registry, registration, attempt_id="registered-retry", version=14)
    _reject_cycle3_artifact(registry, terminal)
    registry.authorize_cycle3_retry_continuation(
        "registered-retry", created_at_utc="2026-10-21T00:01:00Z")
    observation = registry.record_exploratory_kernel_observation(
        version_number=15, run_uuid="b9941b1a-de48-4274-bafa-0d750ca40520",
        terminal_status="COMPLETE", artifact_sha256={"report": "b" * 64},
        actual_gpu_seconds=10, output_bytes=5,
        created_at_utc="2026-10-21T00:02:00Z")

    authorization = registry.authorize_cycle3_retry_rebased(
        "rebased-attempt-15", created_at_utc="2026-10-21T00:03:00Z")
    assert authorization["event_type"] == "cycle3_retry_rebased_authorization_v1"
    assert authorization["event_data"]["observation_event_index"] == observation["event_index"]
    assert authorization["event_data"]["observation_event_hash"] == observation["event_hash"]
    assert authorization["event_data"]["expected_current_version"] == 15
    assert authorization["event_data"]["reserved_gpu_seconds"] == 2800

    wrong_attempt = _preflight(registration, attempt_id="some-other-attempt", timeout=2800)
    wrong_attempt.update(cycle_id="g008-recovery-cycle-3", expected_current_version=15)
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="rebase authorization"):
        registry.preflight_workload(wrong_attempt)
    assert registry.path.read_bytes() == before

    right_attempt = _preflight(registration, attempt_id="rebased-attempt-15", timeout=2800)
    right_attempt.update(cycle_id="g008-recovery-cycle-3", expected_current_version=15)
    preflight = registry.preflight_workload(right_attempt)
    assert preflight["event_data"]["attempt_id"] == "rebased-attempt-15"
    assert registry.observed_kernel_cursor("owner/kernel")["current_version_number"] == 15


def test_cycle3_exploratory_observation_and_rebase_reserve_enforce_frozen_benchmark_cap(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    terminal = _cycle3_complete_retry(registry, registration, attempt_id="registered-retry", version=14)
    _reject_cycle3_artifact(registry, terminal)
    registry.authorize_cycle3_retry_continuation(
        "registered-retry", created_at_utc="2026-10-21T00:01:00Z")
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="benchmark GPU cap"):
        registry.record_exploratory_kernel_observation(
            version_number=15, run_uuid="b9941b1a-de48-4274-bafa-0d750ca40520",
            terminal_status="COMPLETE", artifact_sha256={"report": "b" * 64},
            actual_gpu_seconds=5600, output_bytes=5,
            created_at_utc="2026-10-21T00:02:00Z")
    assert registry.path.read_bytes() == before


def test_cycle3_retry_complete_requires_explicit_validation_rejection(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    completed = _cycle3_complete_retry(registry, registration)
    with pytest.raises(RegistryError, match="explicit latest artifact-validation rejection"):
        registry.authorize_cycle3_retry_continuation(
            "cycle3-benchmark-retry-attempt", created_at_utc="2026-10-21T00:00:00Z")
    assert registry.events()[-1]["event_hash"] == completed["event_hash"]


def test_cycle3_retry_continuation_rejects_stale_head_or_registration_binding(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    terminal = _cycle3_complete_retry(registry, registration, status="ERROR")
    events = registry.events()
    base = {
        "schema_id": "g008-cycle3-retry-policy-amendment", "schema_version": 1,
        "cycle_id": "g008-recovery-cycle-3-retry", "created_at_utc": "2026-10-21T00:00:00Z",
        "policy_sha256": registry_module._CYCLE3_RETRY_POLICY_SHA256,
        "prior_head_hash": "f" * 64, "prior_final_event_index": len(events) - 1,
        "registration_event_index": registration["event_index"],
        "registration_event_hash": registration["event_hash"],
        "registration_id": registration["event_data"]["registration_id"],
        "prior_attempt_id": "cycle3-benchmark-retry-attempt",
        "prior_attempt_event_index": terminal["event_index"],
        "prior_attempt_event_hash": terminal["event_hash"], "prior_terminal_status": "ERROR",
        "validation_rejection_event_index": None, "validation_rejection_event_hash": None,
    }
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="current head"):
        registry._append("cycle3_retry_policy_amendment_v1", base)
    assert registry.path.read_bytes() == before


def test_cycle3_retry_continuation_cannot_exceed_cumulative_budget_reservation(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch, actual_gpu_seconds=2800)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    first = _cycle3_complete_retry(registry, registration, status="ERROR", actual=2800)
    registry.authorize_cycle3_retry_continuation(
        "cycle3-benchmark-retry-attempt", created_at_utc="2026-10-21T00:00:00Z")
    pf_data = _preflight(registration, attempt_id="over-budget-retry", timeout=2800)
    pf_data.update(cycle_id="g008-recovery-cycle-3", expected_current_version=14)
    with pytest.raises(RegistryError, match="cumulative benchmark GPU cap"):
        registry.preflight_workload(pf_data)
    assert registry.events()[-1]["event_hash"] != first["event_hash"]


def test_cycle3_retry_budget_counts_first_attempt_and_retry_reservation_cumulatively(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch, actual_gpu_seconds=2800)
    _, registration = _append_cycle3_retry(registry, _cycle3_retry_workload(registry))
    _cycle3_complete_retry(registry, registration, actual=2800)
    assert registry.events()[-1]["event_data"]["actual_gpu_seconds"] == 2800


@pytest.mark.parametrize("field,value", [
    ("validation_game_count", 511),
    ("validation_root_range", [1_000_001, 1_000_255]),
    ("final_opponent_id", "g007-step-300"),
])
def test_recovery_amendment_rejects_resealed_changes_to_frozen_reservations(tmp_path, monkeypatch, field, value):
    registry, _, _, _ = _legacy_registry_with_recovery_amendment(tmp_path, monkeypatch)
    events = [json.loads(line) for line in registry.path.read_bytes().splitlines()[:14]]
    amendment = events[13]
    amendment["event_data"]["reservation_metadata"][field] = value
    prev = events[12]["event_hash"]
    amendment["prev_event_hash"] = prev
    payload = {k: amendment[k] for k in ("event_index", "event_type", "event_data", "prev_event_hash")}
    amendment["event_hash"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    registry.path.write_bytes(b"\n".join(canonical_bytes(event) for event in events) + b"\n")
    with pytest.raises(RegistryError, match="frozen|changed"):
        registry.events()


@pytest.mark.parametrize("field,value", [
    ("status", "ERROR"),
    ("current_version_number", 99),
    ("kernel_id", "other/kernel"),
])
def test_terminal_status_evidence_must_confirm_terminal_status_version_and_kernel(tmp_path, monkeypatch, field, value):
    registry, benchmark, _, _ = _legacy_registry_with_recovery_amendment(tmp_path, monkeypatch)
    pf_data = _preflight(benchmark)
    pf = registry.preflight_workload(pf_data)
    ds_data = _dispatch(pf_data)
    ds_data.update(preflight_event_index=pf["event_index"], preflight_event_hash=pf["event_hash"])
    ds = registry.dispatch_workload_push(ds_data)
    accepted = registry.accept_workload_push({
        "attempt_id": "attempt-1", "registration_event_index": benchmark["event_index"],
        "registration_event_hash": benchmark["event_hash"], "preflight_event_index": pf["event_index"],
        "preflight_event_hash": pf["event_hash"], "dispatch_event_index": ds["event_index"],
        "dispatch_event_hash": ds["event_hash"], "assigned_version": 11,
        "implementation_variant_id": "paired", "push_response": {"assigned_version": 11},
        "confirming_metadata": {"current_version_number": 11, "kernel_id": "owner/kernel"},
        "lock_head_hash": None,
    })
    terminal = _attempt(pf, ds, accepted_event=accepted)
    terminal["status_evidence"][-1][field] = value
    with pytest.raises(RegistryError, match="last observed status response"):
        registry.resolve_workload_attempt(terminal)


@pytest.mark.parametrize("evidence", [
    _selection_evidence("candidate", parity=False, ratios=[1.2] * 5, median=1.2),
    _selection_evidence("candidate", ratios=[1.2, 1.2, 1.2, 1.0, 1.0], median=1.2),
    _selection_evidence("candidate", ratios=[1.2] * 5, median=1.0),
    {**_selection_evidence(), "pairs": [{"candidate_speedup_ratio": 1.1}] * 4},
])
def test_paired_selection_evidence_rejects_unproven_or_inconsistent_candidate(evidence):
    with pytest.raises(RegistryError):
        registry_module._validate_selection_evidence(evidence)


def test_paired_selection_evidence_accepts_candidate_only_after_all_frozen_gates():
    evidence = _selection_evidence("candidate", ratios=[1.10, 1.11, 1.12, 1.13, 1.20], median=1.12)
    registry_module._validate_selection_evidence(evidence)


def _competitive_campaign_fixture(registry):
    campaign_id = "g008-competitive-20261020-v1"
    hashes = {"package_sha256": "1" * 64, "source_sha256": "2" * 64,
              "seed_plan_sha256": registry_module._RECOVERY_SEED_PLAN_SHA256, "backend_sha256": "4" * 64,
              "trainer_sha256": "5" * 64, "engine_sha256": "6" * 64}
    records = []
    for pair_no, (ppo_seed, root_seed, mixed_root_seed) in enumerate(
            ((1234, 2_100_001, 2_100_011), (1234, 2_100_002, 2_100_012),
             (1234, 2_100_003, 2_100_013)), 1):
        pair_id = f"seed-pair-{pair_no}"
        for role in ("baseline", "mixed"):
            ppo_config = {"learning_rate_micros": 3, "entropy_coefficient_micros": 2,
                          "checkpoint_every": 50}
            shared_config_sha256 = hashlib.sha256(canonical_bytes(ppo_config)).hexdigest()
            record = {**hashes, "campaign_id": campaign_id, "run_uuid": str(uuid.uuid4()),
                      "registration_kind": "train", "strategy_role": role, "pair_id": pair_id,
                      "ppo_seed": ppo_seed, "root_seed": root_seed if role == "baseline" else mixed_root_seed,
                      "update_budget": 236,
                      "shared_config_sha256": shared_config_sha256, "ppo_config": ppo_config,
                      "config_sha256": ("8" if role == "baseline" else "9") * 64,
                      "architecture": {"width": 64, "layers": 2}, "opponent_probabilities":
                      ({"latest": 10_000, "step0": 0, "cruise": 0, "local_mobility": 0, "interceptor": 0}
                       if role == "baseline" else
                       {"latest": 6_000, "step0": 1_000, "cruise": 1_000, "local_mobility": 1_000, "interceptor": 1_000}),
                      "kernel_id": "owner/kernel", "session_timeout_seconds": 1_200,
                      "workload_gpu_seconds_cap": 1_200, "workload_output_bytes_cap": 100_000,
                      "accelerator_count": 1, "machine_shape": "NvidiaTeslaT4", "open_space_allowed": False}
            records.append((f"train-{pair_no}-{role}", "train", record))
    benchmark = {**hashes, "config_sha256": "a" * 64, "campaign_id": campaign_id,
                 "run_uuid": str(uuid.uuid4()), "registration_kind": "benchmark", "strategy_role": "benchmark",
                 "kernel_id": "owner/kernel", "session_timeout_seconds": 600,
                 "workload_gpu_seconds_cap": 1_800, "workload_output_bytes_cap": 100_000,
                 "accelerator_count": 1, "machine_shape": "NvidiaTeslaT4", "open_space_allowed": False}
    records.insert(0, ("benchmark-corrected", "benchmark", benchmark))
    rosters = {
        "validation": {"roster_sha256": registry_module._VALIDATION_OPPONENT_ROSTER_SHA256,
                       "opponent_code_sha256": registry_module._VALIDATION_OPPONENT_SHA256,
                       "game_count": 512, "seat_order": [0, 1]},
        "final": {"roster_sha256": registry_module._FINAL_OPPONENT_ROSTER_SHA256,
                  "opponent_code_sha256": registry_module._FINAL_OPPONENT_CODE_SHA256,
                  "game_count": 512, "seat_order": [0, 1]},
    }
    for kind, timeout, roster in (("validation", 600, rosters["validation"]), ("final", 600, rosters["final"])):
        record = {**hashes, "config_sha256": "f" * 64, "campaign_id": campaign_id,
                  "run_uuid": str(uuid.uuid4()), "registration_kind": kind, "strategy_role": kind,
                  **roster, "kernel_id": "owner/kernel", "session_timeout_seconds": timeout,
                  "workload_gpu_seconds_cap": timeout, "workload_output_bytes_cap": 100_000,
                  "accelerator_count": 1, "machine_shape": "NvidiaTeslaT4", "open_space_allowed": False,
                  "pre_final_data_visible": False}
        records.append((f"{kind}-run", kind, record))
    rows = [{"registration_id": rid, "registration_kind": kind, "record": record,
             "record_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest()}
            for rid, kind, record in records]
    events = registry.events()
    rejection = events[-1]
    return {
        "schema_id": "g008-competitive-campaign", "schema_version": 1, "campaign_id": campaign_id,
        "created_at_utc": "2026-10-21T00:00:00Z", "prior_head_hash": rejection["event_hash"],
        "prior_final_event_index": rejection["event_index"], "rejection_event_index": rejection["event_index"],
        "rejection_event_hash": rejection["event_hash"], "kernel_id": "owner/kernel",
        "current_version_number": 14, "benchmark_gpu_seconds_cap": 1_800,
        "training_gpu_seconds_cap": 8_000, "evaluation_gpu_seconds_cap": 1_500,
        "total_gpu_seconds_cap": 11_300, "protected_final_reserve_gpu_seconds": 7_200,
        "evaluation_rosters": rosters,
        "validation_reference": {"report_sha256": "22f7e0fca065eba24ab57c5fae4ea78223fa6d01153a7a8d90e83ac8269228d7",
                                "game_count": 512, "wins": 217, "draws": 51, "losses": 244,
                                "noninferiority_margin": "0.065000"},
        "records": rows,
    }


def _campaign_rejected_registry(tmp_path, monkeypatch):
    registry, _, _ = _cycle3_failed_attempt_registry(tmp_path, monkeypatch)
    workload = _cycle3_retry_workload(registry)
    _, registration = _append_cycle3_retry(registry, workload)
    terminal = _cycle3_complete_retry(registry, registration, version=14)
    registry.reject_benchmark_artifact_validation(
        terminal["event_index"], terminal["event_hash"], artifact_sha256="a" * 64,
        output_sha256="b" * 64, validation_codes=["device_count_mismatch"],
        validator_id="test-validator-v1", created_at_utc="2026-10-21T00:00:00Z")
    return registry


def test_competitive_campaign_atomically_freezes_nine_registrations_after_exact_rejection(tmp_path, monkeypatch):
    registry = _campaign_rejected_registry(tmp_path, monkeypatch)
    prefix = registry.path.read_bytes()
    amendment = _competitive_campaign_fixture(registry)
    events = registry.register_g008_competitive_campaign_batch(amendment)
    assert [event["event_type"] for event in events] == [
        "g008_competitive_campaign_amendment_v1", *(["registration"] * 9)]
    assert [event["event_data"]["registration_kind"] for event in events[1:]] == [
        "benchmark", *(["train"] * 6), "validation", "final"]
    assert registry.path.read_bytes().startswith(prefix)
    assert registry.events()[-1]["event_hash"] == events[-1]["event_hash"]


def test_competitive_final_statistics_derive_rate_wilson_interval_and_60_percent_gate():
    below = registry_module.competitive_final_statistics(307, 0, 205)
    at_target = registry_module.competitive_final_statistics(308, 0, 204)
    assert below == {"game_count": 512, "wins": 307, "draws": 0, "losses": 205,
                     "win_rate": "0.599609", "wilson_95_lower": "0.556578",
                     "wilson_95_upper": "0.641157", "passed": False}
    assert at_target["win_rate"] == "0.601562"
    assert at_target["wilson_95_lower"] < "0.60"
    assert at_target["passed"] is True


def test_competitive_validation_derives_reference_noninferiority_from_wdl():
    reference = registry_module.competitive_noninferiority_statistics(217, 51, 244)
    regressed = registry_module.competitive_noninferiority_statistics(0, 0, 512)
    assert reference["reference_report_sha256"] == "22f7e0fca065eba24ab57c5fae4ea78223fa6d01153a7a8d90e83ac8269228d7"
    assert reference["win_rate_difference_95_lower"] == "-0.060319"
    assert reference["passed"] is True
    assert regressed["passed"] is False


def test_competitive_result_schemas_require_artifact_and_attempt_bindings():
    freeze = {"schema_id": "g008-candidate-freeze", "schema_version": 1,
              "campaign_id": "g008-competitive-20261020-v1", "created_at_utc": "2026-10-21T00:00:00Z",
              "candidate_registration_event_index": 50, "candidate_registration_event_hash": "a" * 64,
              "train_attempt_event_index": 60, "train_attempt_event_hash": "b" * 64,
              "checkpoint_sha256": "c" * 64, "checkpoint_artifact_sha256": "c" * 64,
              "output_manifest_sha256": "d" * 64}
    registry_module._validate_data("competitive_candidate_freeze", freeze)
    bad_freeze = {**freeze, "checkpoint_artifact_sha256": "e" * 64}
    with pytest.raises(RegistryError, match="candidate-freeze"):
        registry_module._validate_data("competitive_candidate_freeze", bad_freeze)

    validation_stats = registry_module.competitive_noninferiority_statistics(217, 51, 244)
    validation = {"schema_id": "g008-candidate-validation-result", "schema_version": 1,
                  "campaign_id": "g008-competitive-20261020-v1", "created_at_utc": "2026-10-21T00:00:00Z",
                  "candidate_freeze_event_index": 61, "candidate_freeze_event_hash": "f" * 64,
                  "validation_registration_event_index": 55, "validation_registration_event_hash": "a" * 64,
                  "validation_attempt_event_index": 70, "validation_attempt_event_hash": "b" * 64,
                  "candidate_checkpoint_sha256": "c" * 64, "report_sha256": "e" * 64,
                  "output_manifest_sha256": "d" * 64,
                  "reference_report_sha256": validation_stats["reference_report_sha256"],
                  "reference_game_count": 512, "reference_wins": 217, "reference_draws": 51,
                  "reference_losses": 244, "noninferiority_margin": "0.065000",
                  "game_count": 512, "wins": 217, "draws": 51, "losses": 244,
                  "win_rate": validation_stats["win_rate"], "wilson_95_lower": validation_stats["wilson_95_lower"],
                  "wilson_95_upper": validation_stats["wilson_95_upper"],
                  "win_rate_difference_95_lower": validation_stats["win_rate_difference_95_lower"],
                  "win_rate_difference_95_upper": validation_stats["win_rate_difference_95_upper"],
                  "passed": validation_stats["passed"], "opponent_id": "g007-step-300"}
    registry_module._validate_data("competitive_validation_result", validation)
    with pytest.raises(RegistryError, match="validation result"):
        registry_module._validate_data("competitive_validation_result",
                                       {**validation, "output_manifest_sha256": "invalid"})

    final_stats = registry_module.competitive_final_statistics(308, 0, 204)
    final = {"schema_id": "g008-final-evaluation-result", "schema_version": 1,
             "campaign_id": "g008-competitive-20261020-v1", "created_at_utc": "2026-10-21T00:00:00Z",
             "candidate_freeze_event_index": 61, "candidate_freeze_event_hash": "f" * 64,
             "final_registration_event_index": 56, "final_registration_event_hash": "a" * 64,
             "final_attempt_event_index": 80, "final_attempt_event_hash": "b" * 64,
             "candidate_checkpoint_sha256": "c" * 64, "report_sha256": "e" * 64,
             "output_manifest_sha256": "d" * 64, "roster_sha256": "1" * 64,
             "opponent_code_sha256": "2" * 64,
             "game_count": 512, "wins": 308, "draws": 0, "losses": 204,
             "win_rate": final_stats["win_rate"], "wilson_95_lower": final_stats["wilson_95_lower"],
             "wilson_95_upper": final_stats["wilson_95_upper"], "opponent_id": "open-space", "passed": True}
    registry_module._validate_data("competitive_final_result", final)


def _complete_registered_campaign_attempt(registry, registration, *, version, variant_id, attempt_id, lock_hash=None):
    timeout = registration["event_data"]["record"]["session_timeout_seconds"]
    preflight_data = _preflight(registration, attempt_id=attempt_id, timeout=timeout,
                                variant_id=variant_id, lock_hash=lock_hash)
    preflight_data.update(cycle_id="g008-competitive-campaign-v1", expected_current_version=version - 1,
                          lock_head_hash=lock_hash)
    preflight = registry.preflight_workload(preflight_data)
    dispatch_data = _dispatch(preflight_data, attempt_id=attempt_id, lock_hash=lock_hash)
    dispatch_data.update(cycle_id="g008-competitive-campaign-v1", preflight_event_index=preflight["event_index"],
                         preflight_event_hash=preflight["event_hash"])
    dispatch = registry.dispatch_workload_push(dispatch_data)
    accepted = registry.accept_workload_push({
        "attempt_id": attempt_id, "cycle_id": "g008-competitive-campaign-v1",
        "registration_event_index": registration["event_index"],
        "registration_event_hash": registration["event_hash"],
        "preflight_event_index": preflight["event_index"], "preflight_event_hash": preflight["event_hash"],
        "dispatch_event_index": dispatch["event_index"], "dispatch_event_hash": dispatch["event_hash"],
        "assigned_version": version, "implementation_variant_id": variant_id,
        "push_response": {"assigned_version": version},
        "confirming_metadata": {"current_version_number": version, "kernel_id": "owner/kernel"},
        "lock_head_hash": lock_hash,
    })
    terminal_data = _attempt(preflight, dispatch, accepted_event=accepted, lock_hash=lock_hash)
    terminal_data.update(cycle_id="g008-competitive-campaign-v1", actual_gpu_seconds=1, output_bytes=10,
                         terminal_status="COMPLETE",
                         terminal_metadata={"current_version_number": version, "kernel_id": "owner/kernel",
                                            "status": "COMPLETE"},
                         status_evidence=[{"status": "COMPLETE", "current_version_number": version,
                                           "kernel_id": "owner/kernel"}])
    return registry.resolve_workload_attempt(terminal_data)


def test_competitive_results_bind_exact_complete_attempts_and_frozen_checkpoint(tmp_path, monkeypatch):
    registry = _campaign_rejected_registry(tmp_path, monkeypatch)
    registrations = registry.register_g008_competitive_campaign_batch(
        _competitive_campaign_fixture(registry))[1:]
    benchmark = registrations[0]
    benchmark_terminal = _complete_registered_campaign_attempt(
        registry, benchmark, version=15, variant_id="paired", attempt_id="binding-benchmark")
    artifact = registry.register_benchmark_artifact(
        "a" * 64, benchmark["event_index"], benchmark["event_hash"],
        attempt_event_index=benchmark_terminal["event_index"],
        attempt_event_hash=benchmark_terminal["event_hash"], report_sha256="b" * 64,
        selection_evidence=_selection_evidence())
    lock = registry.close_registrations()
    registry.attest_benchmark(artifact["event_index"])

    train_terminals = {}
    candidate_registration = None
    for offset, registration in enumerate(registrations[1:7], 16):
        role = registration["event_data"]["record"]["strategy_role"]
        pair_id = registration["event_data"]["record"]["pair_id"]
        terminal = _complete_registered_campaign_attempt(
            registry, registration, version=offset, variant_id=role,
            attempt_id=f"binding-{pair_id}-{role}", lock_hash=lock["event_hash"])
        train_terminals[registration["event_index"]] = terminal
        if role == "mixed" and pair_id == "seed-pair-3":
            candidate_registration = registration
    checkpoint_sha = "c" * 64
    candidate_terminal = train_terminals[candidate_registration["event_index"]]
    freeze_data = {"schema_id": "g008-candidate-freeze", "schema_version": 1,
                   "campaign_id": "g008-competitive-20261020-v1", "created_at_utc": "2026-10-21T00:00:00Z",
                   "candidate_registration_event_index": candidate_registration["event_index"],
                   "candidate_registration_event_hash": candidate_registration["event_hash"],
                   "train_attempt_event_index": candidate_terminal["event_index"],
                   "train_attempt_event_hash": candidate_terminal["event_hash"],
                   "checkpoint_sha256": checkpoint_sha, "checkpoint_artifact_sha256": checkpoint_sha,
                   "output_manifest_sha256": "d" * 64}
    wrong_freeze = {**freeze_data, "train_attempt_event_index": benchmark_terminal["event_index"],
                    "train_attempt_event_hash": benchmark_terminal["event_hash"]}
    with pytest.raises(RegistryError, match="exact COMPLETE third mixed-arm attempt"):
        registry.freeze_competitive_candidate(wrong_freeze)
    freeze = registry.freeze_competitive_candidate(freeze_data)

    validation_registration = registrations[7]
    validation_terminal = _complete_registered_campaign_attempt(
        registry, validation_registration, version=22, variant_id="frozen-candidate",
        attempt_id="binding-validation", lock_hash=lock["event_hash"])
    stats = registry_module.competitive_noninferiority_statistics(217, 51, 244)
    validation_data = {"schema_id": "g008-candidate-validation-result", "schema_version": 1,
                       "campaign_id": "g008-competitive-20261020-v1", "created_at_utc": "2026-10-21T00:00:00Z",
                       "candidate_freeze_event_index": freeze["event_index"],
                       "candidate_freeze_event_hash": freeze["event_hash"],
                       "validation_registration_event_index": validation_registration["event_index"],
                       "validation_registration_event_hash": validation_registration["event_hash"],
                       "validation_attempt_event_index": validation_terminal["event_index"],
                       "validation_attempt_event_hash": validation_terminal["event_hash"],
                       "candidate_checkpoint_sha256": checkpoint_sha, "report_sha256": "e" * 64,
                       "output_manifest_sha256": "f" * 64,
                       "reference_report_sha256": stats["reference_report_sha256"],
                       "reference_game_count": 512, "reference_wins": 217, "reference_draws": 51,
                       "reference_losses": 244, "noninferiority_margin": "0.065000",
                       "game_count": 512, "wins": 217, "draws": 51, "losses": 244,
                       "win_rate": stats["win_rate"], "wilson_95_lower": stats["wilson_95_lower"],
                       "wilson_95_upper": stats["wilson_95_upper"],
                       "win_rate_difference_95_lower": stats["win_rate_difference_95_lower"],
                       "win_rate_difference_95_upper": stats["win_rate_difference_95_upper"],
                       "passed": stats["passed"], "opponent_id": "g007-step-300"}
    wrong_validation = {**validation_data,
                        "validation_attempt_event_index": candidate_terminal["event_index"],
                        "validation_attempt_event_hash": candidate_terminal["event_hash"]}
    with pytest.raises(RegistryError, match="exact COMPLETE attempt"):
        registry.record_competitive_validation_result(wrong_validation)
    registry.record_competitive_validation_result(validation_data)

    final_registration = registrations[8]
    final_terminal = _complete_registered_campaign_attempt(
        registry, final_registration, version=23, variant_id="frozen-candidate",
        attempt_id="binding-final", lock_hash=lock["event_hash"])
    final_stats = registry_module.competitive_final_statistics(308, 0, 204)
    final_data = {"schema_id": "g008-final-evaluation-result", "schema_version": 1,
                  "campaign_id": "g008-competitive-20261020-v1", "created_at_utc": "2026-10-21T00:00:00Z",
                  "candidate_freeze_event_index": freeze["event_index"],
                  "candidate_freeze_event_hash": freeze["event_hash"],
                  "final_registration_event_index": final_registration["event_index"],
                  "final_registration_event_hash": final_registration["event_hash"],
                  "final_attempt_event_index": final_terminal["event_index"],
                  "final_attempt_event_hash": final_terminal["event_hash"],
                  "candidate_checkpoint_sha256": checkpoint_sha, "report_sha256": "1" * 64,
                  "output_manifest_sha256": "2" * 64,
                  "roster_sha256": final_registration["event_data"]["record"]["roster_sha256"],
                  "opponent_code_sha256": final_registration["event_data"]["record"]["opponent_code_sha256"],
                  "game_count": 512, "wins": 308, "draws": 0, "losses": 204,
                  "win_rate": final_stats["win_rate"], "wilson_95_lower": final_stats["wilson_95_lower"],
                  "wilson_95_upper": final_stats["wilson_95_upper"],
                  "opponent_id": "open-space", "passed": True}
    wrong_final = {**final_data, "candidate_checkpoint_sha256": "3" * 64}
    with pytest.raises(RegistryError, match="exact COMPLETE final attempt"):
        registry.record_competitive_final_result(wrong_final)
    result = registry.record_competitive_final_result(final_data)
    assert registry.events()[-1]["event_hash"] == result["event_hash"]


def test_competitive_campaign_rejects_stale_head_missing_or_mismatched_pairs_without_mutation(tmp_path, monkeypatch):
    registry = _campaign_rejected_registry(tmp_path, monkeypatch)
    amendment = _competitive_campaign_fixture(registry)
    amendment["prior_head_hash"] = "f" * 64
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="exact current rejected-artifact head"):
        registry.register_g008_competitive_campaign_batch(amendment)
    assert registry.path.read_bytes() == before

    amendment = _competitive_campaign_fixture(registry)
    amendment["records"].pop()
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="nine|nine|benchmark, six"):
        registry.register_g008_competitive_campaign_batch(amendment)
    assert registry.path.read_bytes() == before

    amendment = _competitive_campaign_fixture(registry)
    base, mixed = amendment["records"][1], amendment["records"][2]
    mixed["record"]["root_seed"] = base["record"]["root_seed"]
    mixed["record_sha256"] = hashlib.sha256(canonical_bytes(mixed["record"])).hexdigest()
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="three complete blocks"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v1", amendment)
    assert registry.path.read_bytes() == before


def test_competitive_campaign_requires_attestation_before_training_and_freeze_before_final(tmp_path, monkeypatch):
    registry = _campaign_rejected_registry(tmp_path, monkeypatch)
    events = registry.register_g008_competitive_campaign_batch(_competitive_campaign_fixture(registry))
    registrations = events[1:]
    training = registrations[1]
    train_pf = _preflight(training, attempt_id="campaign-train-before-attest", timeout=1_200,
                          variant_id="baseline")
    train_pf.update(cycle_id="g008-competitive-campaign-v1", expected_current_version=14)
    with pytest.raises(RegistryError, match="attestation"):
        registry.preflight_workload(train_pf)
    freeze = {"schema_id": "g008-candidate-freeze", "schema_version": 1,
              "campaign_id": "g008-competitive-20261020-v1", "created_at_utc": "2026-10-21T00:00:00Z",
              "candidate_registration_event_index": training["event_index"],
              "candidate_registration_event_hash": training["event_hash"],
              "train_attempt_event_index": 99, "train_attempt_event_hash": "b" * 64,
              "checkpoint_sha256": "a" * 64, "checkpoint_artifact_sha256": "a" * 64,
              "output_manifest_sha256": "c" * 64}
    with pytest.raises(RegistryError, match="candidate freeze requires attested benchmark"):
        registry.freeze_competitive_candidate(freeze)

    final = registrations[-1]
    final_pf = _preflight(final, attempt_id="campaign-final-too-early", timeout=600,
                          variant_id="frozen-candidate")
    final_pf.update(cycle_id="g008-competitive-campaign-v1", expected_current_version=14)
    with pytest.raises(RegistryError, match="attestation|candidate freeze|non-inferiority"):
        registry.preflight_workload(final_pf)


def test_competitive_campaign_budget_and_benchmark_artifact_lock_attestation_adjacency(tmp_path, monkeypatch):
    registry = _campaign_rejected_registry(tmp_path, monkeypatch)
    amendment = _competitive_campaign_fixture(registry)
    amendment["evaluation_gpu_seconds_cap"] = 1_100
    with pytest.raises(RegistryError, match="evaluation reservations"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v1", amendment)

    amendment = _competitive_campaign_fixture(registry)
    _, *registrations = registry.register_g008_competitive_campaign_batch(amendment)
    benchmark = registrations[0]
    pf_data = _preflight(benchmark, attempt_id="campaign-benchmark", timeout=600, variant_id="paired")
    pf_data.update(cycle_id="g008-competitive-campaign-v1", expected_current_version=14)
    pf = registry.preflight_workload(pf_data)
    dispatch_data = _dispatch(pf_data, attempt_id="campaign-benchmark")
    dispatch_data.update(cycle_id="g008-competitive-campaign-v1", preflight_event_index=pf["event_index"],
                         preflight_event_hash=pf["event_hash"])
    dispatch = registry.dispatch_workload_push(dispatch_data)
    accepted = registry.accept_workload_push({
        "attempt_id": "campaign-benchmark", "cycle_id": "g008-competitive-campaign-v1",
        "registration_event_index": benchmark["event_index"], "registration_event_hash": benchmark["event_hash"],
        "preflight_event_index": pf["event_index"], "preflight_event_hash": pf["event_hash"],
        "dispatch_event_index": dispatch["event_index"], "dispatch_event_hash": dispatch["event_hash"],
        "assigned_version": 15, "implementation_variant_id": "paired", "push_response": {"assigned_version": 15},
        "confirming_metadata": {"current_version_number": 15, "kernel_id": "owner/kernel"}, "lock_head_hash": None,
    })
    terminal_data = _attempt(pf, dispatch, accepted_event=accepted)
    terminal_data.update(cycle_id="g008-competitive-campaign-v1", actual_gpu_seconds=900, output_bytes=10,
                         terminal_status="COMPLETE",
                         terminal_metadata={"current_version_number": 15, "kernel_id": "owner/kernel", "status": "COMPLETE"},
                         status_evidence=[{"status": "COMPLETE", "current_version_number": 15,
                                           "kernel_id": "owner/kernel"}])
    terminal = registry.resolve_workload_attempt(terminal_data)
    artifact = registry.register_benchmark_artifact(
        "a" * 64, benchmark["event_index"], benchmark["event_hash"],
        attempt_event_index=terminal["event_index"], attempt_event_hash=terminal["event_hash"],
        report_sha256="b" * 64, selection_evidence=_selection_evidence())
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError):
        registry._append("evidence", {"registration_event_index": registrations[1]["event_index"],
                                       "registration_event_hash": registrations[1]["event_hash"],
                                       "lock_head_hash": "c" * 64, "artifact_sha256": "d" * 64})
    assert registry.path.read_bytes() == before
    lock = registry.close_registrations()
    assert lock["event_index"] == artifact["event_index"] + 1
    attestation = registry.attest_benchmark(artifact["event_index"])
    assert attestation["event_index"] == lock["event_index"] + 1
    assert registry.events()[-1]["event_hash"] == attestation["event_hash"]


def test_competitive_campaign_failed_benchmark_can_retry_within_budget_but_pending_attempt_blocks(tmp_path, monkeypatch):
    registry = _campaign_rejected_registry(tmp_path, monkeypatch)
    _, *registrations = registry.register_g008_competitive_campaign_batch(_competitive_campaign_fixture(registry))
    benchmark = registrations[0]
    first_data = _preflight(benchmark, attempt_id="campaign-failed-benchmark", timeout=600, variant_id="paired")
    first_data.update(cycle_id="g008-competitive-campaign-v1", expected_current_version=14)
    first = registry.preflight_workload(first_data)
    dispatch_data = _dispatch(first_data, attempt_id="campaign-failed-benchmark")
    dispatch_data.update(cycle_id="g008-competitive-campaign-v1", preflight_event_index=first["event_index"],
                         preflight_event_hash=first["event_hash"])
    dispatch = registry.dispatch_workload_push(dispatch_data)
    accepted = registry.accept_workload_push({
        "attempt_id": "campaign-failed-benchmark", "cycle_id": "g008-competitive-campaign-v1",
        "registration_event_index": benchmark["event_index"], "registration_event_hash": benchmark["event_hash"],
        "preflight_event_index": first["event_index"], "preflight_event_hash": first["event_hash"],
        "dispatch_event_index": dispatch["event_index"], "dispatch_event_hash": dispatch["event_hash"],
        "assigned_version": 15, "implementation_variant_id": "paired", "push_response": {"assigned_version": 15},
        "confirming_metadata": {"current_version_number": 15, "kernel_id": "owner/kernel"}, "lock_head_hash": None,
    })
    terminal_data = _attempt(first, dispatch, accepted_event=accepted)
    terminal_data.update(cycle_id="g008-competitive-campaign-v1", actual_gpu_seconds=500, output_bytes=10,
                         terminal_status="ERROR",
                         terminal_metadata={"current_version_number": 15, "kernel_id": "owner/kernel", "status": "ERROR"},
                         status_evidence=[{"status": "ERROR", "current_version_number": 15,
                                           "kernel_id": "owner/kernel"}])
    terminal = registry.resolve_workload_attempt(terminal_data)
    retry = _preflight(benchmark, attempt_id="campaign-benchmark-retry", timeout=600, variant_id="paired")
    retry.update(cycle_id="g008-competitive-campaign-v1", expected_current_version=15)
    retry_event = registry.preflight_workload(retry)
    assert retry_event["event_data"]["attempt_id"] == "campaign-benchmark-retry"
    pending_before = registry.path.read_bytes()
    another = _preflight(benchmark, attempt_id="campaign-illegal-overlap", timeout=600, variant_id="paired")
    another.update(cycle_id="g008-competitive-campaign-v1", expected_current_version=15)
    with pytest.raises(RegistryError, match="pending campaign attempt"):
        registry.preflight_workload(another)
    assert registry.path.read_bytes() == pending_before
    assert terminal["event_data"]["terminal_status"] == "ERROR"


def _competitive_campaign_v2_fixture():
    campaign_id = "g008-competitive-20261021-v2"
    hashes = {"package_sha256": "1" * 64, "source_sha256": "2" * 64,
              "seed_plan_sha256": registry_module._RECOVERY_SEED_PLAN_SHA256,
              "backend_sha256": "4" * 64, "trainer_sha256": "5" * 64,
              "engine_sha256": "6" * 64}
    rows = []
    for pair_no in range(1, 4):
        for role in ("baseline", "mixed"):
            train_config = {"width": 12, "height": 12, "arena_count": 128, "rollout_steps": 32,
                            "total_updates": 1_000, "update_epochs": 1, "minibatch_size": 1_024,
                            "hidden_size": 128, "learning_rate": "0.0003", "gamma": "0.99",
                            "gae_lambda": "0.95", "clip_ratio": "0.2", "entropy_coef": "0.01",
                            "value_coef": "0.5", "seed": 3_300 + pair_no, "device": "auto",
                            "checkpoint_every": 50, "onnx_opset": 17,
                            "snapshot_opponent_probability": "0.1" if role == "mixed" else "0.0",
                            "heuristic_opponent_probability": "0.3333333333333333" if role == "mixed" else "0.0"}
            probabilities = ({"latest": 10_000, "step0": 0, "cruise": 0,
                              "local_mobility": 0, "interceptor": 0} if role == "baseline"
                             else {"latest": 6_000, "step0": 1_000, "cruise": 1_000,
                                   "local_mobility": 1_000, "interceptor": 1_000})
            record = {**hashes, "campaign_id": campaign_id, "run_uuid": str(uuid.uuid4()),
                      "registration_kind": "train", "strategy_role": role,
                      "pair_id": f"seed-pair-{pair_no}", "ppo_seed": 3_300 + pair_no,
                      "root_seed": 2_000_000 + pair_no * 100_000 + 1,
                      "update_budget": 1_000, "shared_config_sha256":
                          registry_module._g008_v2_shared_config_sha256(train_config),
                      "train_config": train_config,
                      "architecture": {"width": 12, "height": 12, "hidden_size": 128},
                      "opponent_probabilities": probabilities, "config_sha256": "8" * 64,
                      "schedule_descriptor_sha256": "9" * 64,
                      "kernel_id": "pascalcoimbra/circuit-ai", "session_timeout_seconds": 1_200,
                      "workload_gpu_seconds_cap": 1_200, "workload_output_bytes_cap": 100_000_000,
                      "accelerator_count": 1, "machine_shape": "NvidiaTeslaT4", "open_space_allowed": False}
            record["config_sha256"] = registry_module._g008_v2_train_config_sha256(train_config)
            rows.append((f"train-{pair_no}-{role}", "train", record))
    validation_roster = {"roster_sha256": registry_module._VALIDATION_OPPONENT_ROSTER_SHA256,
                         "opponent_code_sha256": registry_module._VALIDATION_OPPONENT_SHA256,
                         "game_count": 512, "seat_order": [0, 1],
                         "paired_seeds_sha256": "0" * 64}
    for train_id, _, training in rows[:]:
        pair_id, role = training["pair_id"], training["strategy_role"]
        record = {**hashes, "config_sha256": "f" * 64, "campaign_id": campaign_id,
                  "run_uuid": str(uuid.uuid4()), "registration_kind": "validation",
                  "strategy_role": role, "pair_id": pair_id,
                  "training_registration_id": train_id, "training_run_uuid": training["run_uuid"],
                  "training_record_sha256": hashlib.sha256(canonical_bytes(training)).hexdigest(),
                  "roster_sha256": validation_roster["roster_sha256"],
                  "opponent_code_sha256": validation_roster["opponent_code_sha256"],
                  "paired_seeds_sha256": validation_roster["paired_seeds_sha256"],
                  "game_count": 512, "seat_order": [0, 1], "kernel_id": "pascalcoimbra/circuit-ai",
                  "session_timeout_seconds": 600, "workload_gpu_seconds_cap": 600,
                  "workload_output_bytes_cap": 100_000_000, "accelerator_count": 1,
                  "machine_shape": "NvidiaTeslaT4", "open_space_allowed": False}
        rows.append((f"validation-{pair_id}-{role}", "validation", record))
    record = {**hashes, "config_sha256": "f" * 64, "campaign_id": campaign_id,
              "run_uuid": str(uuid.uuid4()), "registration_kind": "final", "strategy_role": "final",
              "roster_sha256": registry_module._FINAL_OPPONENT_ROSTER_SHA256,
              "opponent_code_sha256": registry_module._FINAL_OPPONENT_CODE_SHA256,
              "game_count": 512, "seat_order": [0, 1], "kernel_id": "pascalcoimbra/circuit-ai",
              "session_timeout_seconds": 600, "workload_gpu_seconds_cap": 600,
              "workload_output_bytes_cap": 100_000_000, "accelerator_count": 1,
              "machine_shape": "NvidiaTeslaT4", "open_space_allowed": False,
              "pre_final_data_visible": False}
    rows.append(("final-run", "final", record))
    records = [{"registration_id": rid, "registration_kind": kind, "record": record,
                "record_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest()}
               for rid, kind, record in rows]
    pair_seed_commitment = hashlib.sha256(canonical_bytes([
        {"pair_id": f"seed-pair-{pair_no}", "ppo_seed": 3_300 + pair_no,
         "root_seed": 2_000_000 + pair_no * 100_000 + 1, "seat_order": [0, 1]}
        for pair_no in range(1, 4)
    ])).hexdigest()
    validation_roster["paired_seeds_sha256"] = pair_seed_commitment
    for row in records[6:12]:
        row["record"]["paired_seeds_sha256"] = pair_seed_commitment
        row["record_sha256"] = hashlib.sha256(canonical_bytes(row["record"])).hexdigest()
    return {
        "schema_id": "g008-competitive-campaign", "schema_version": 2,
        "campaign_id": campaign_id, "created_at_utc": "2026-10-21T02:00:00Z",
        "prior_head_hash": "a" * 64, "prior_final_event_index": 54,
        "benchmark_artifact_event_index": 48, "benchmark_artifact_event_hash": "b" * 64,
        "benchmark_lock_event_index": 49, "benchmark_lock_event_hash": "c" * 64,
        "benchmark_attestation_event_index": 50, "benchmark_attestation_event_hash": "d" * 64,
        "selected_training_registration_event_index": 29,
        "selected_training_registration_event_hash": "e" * 64,
        "selected_training_attempt_event_index": 54, "selected_training_attempt_event_hash": "f" * 64,
        "selected_training_run_uuid": str(uuid.uuid4()),
        "kernel_id": "pascalcoimbra/circuit-ai", "current_version_number": 17,
        "training_gpu_seconds_cap": 7_200, "evaluation_gpu_seconds_cap": 4_200,
        "total_gpu_seconds_cap": 11_400, "protected_final_reserve_gpu_seconds": 7_200,
        "output_caps_bytes": {"aggregate": 5_000_000_000, "per_run": 100_000_000},
        "quota_gpu_seconds_remaining_before": 99_000, "validation_roster": validation_roster,
        "candidate_selection_rule_id":
        "paired-score-gain-t95-lower-positive-no-pair-regresses-over-5pp-then-median-arm-checkpoint-v1",
        "records": records,
    }


def test_competitive_campaign_v2_schema_freezes_exact_anchors_pairs_and_budget():
    amendment = _competitive_campaign_v2_fixture()
    registry_module._validate_data("g008_competitive_campaign_amendment_v2", amendment)

    def refresh_train_row_and_validation_link(value, train_index):
        train_row = value["records"][train_index]
        train_row["record_sha256"] = hashlib.sha256(canonical_bytes(train_row["record"])).hexdigest()
        validation = value["records"][6 + train_index]
        validation["record"]["training_record_sha256"] = train_row["record_sha256"]
        validation["record_sha256"] = hashlib.sha256(canonical_bytes(validation["record"])).hexdigest()

    wrong_anchor = _competitive_campaign_v2_fixture()
    wrong_anchor["selected_training_attempt_event_index"] = 53
    with pytest.raises(RegistryError, match="provenance anchor"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v2", wrong_anchor)

    mismatched_pair = _competitive_campaign_v2_fixture()
    baseline, mixed = mismatched_pair["records"][:2]
    mixed["record"]["ppo_seed"] = mismatched_pair["records"][2]["record"]["ppo_seed"]
    mixed["record_sha256"] = hashlib.sha256(canonical_bytes(mixed["record"])).hexdigest()
    with pytest.raises(RegistryError, match="schedule/seeds"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v2", mismatched_pair)

    per_arm_packages = _competitive_campaign_v2_fixture()
    per_arm_mixed = per_arm_packages["records"][1]
    per_arm_mixed["record"]["package_sha256"] = "a" * 64
    refresh_train_row_and_validation_link(per_arm_packages, 1)
    registry_module._validate_data("g008_competitive_campaign_amendment_v2", per_arm_packages)

    mismatched_source = _competitive_campaign_v2_fixture()
    mismatched_source["records"][1]["record"]["source_sha256"] = "a" * 64
    refresh_train_row_and_validation_link(mismatched_source, 1)
    with pytest.raises(RegistryError, match="shared source/component/config identity"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v2", mismatched_source)

    over_budget = _competitive_campaign_v2_fixture()
    over_budget["quota_gpu_seconds_remaining_before"] = 18_599
    with pytest.raises(RegistryError, match="compute, quota, or output budget"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v2", over_budget)

    incomplete_validation = _competitive_campaign_v2_fixture()
    incomplete_validation["records"].pop(11)
    with pytest.raises(RegistryError, match="six training, six validation, and final"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v2", incomplete_validation)

    exposed_final_roster = _competitive_campaign_v2_fixture()
    exposed_final_roster["records"][12]["record"]["pre_final_data_visible"] = True
    exposed_final_roster["records"][12]["record_sha256"] = hashlib.sha256(
        canonical_bytes(exposed_final_roster["records"][12]["record"])).hexdigest()
    with pytest.raises(RegistryError, match="roster must remain hidden"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v2", exposed_final_roster)

    wrong_final_roster_hash = _competitive_campaign_v2_fixture()
    wrong_final_roster_hash["records"][12]["record"]["roster_sha256"] = "a" * 64
    wrong_final_roster_hash["records"][12]["record_sha256"] = hashlib.sha256(
        canonical_bytes(wrong_final_roster_hash["records"][12]["record"])).hexdigest()
    with pytest.raises(RegistryError, match="frozen open-space roster hashes"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v2", wrong_final_roster_hash)

    noncanonical_config_float = _competitive_campaign_v2_fixture()
    noncanonical_config_float["records"][0]["record"]["train_config"]["learning_rate"] = "0.00030"
    noncanonical_config_float["records"][0]["record_sha256"] = hashlib.sha256(
        canonical_bytes(noncanonical_config_float["records"][0]["record"])).hexdigest()
    with pytest.raises(RegistryError, match="round-trip through repr"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v2", noncanonical_config_float)

    wrong_config_digest = _competitive_campaign_v2_fixture()
    wrong_config_digest["records"][0]["record"]["config_sha256"] = "a" * 64
    wrong_config_digest["records"][0]["record_sha256"] = hashlib.sha256(
        canonical_bytes(wrong_config_digest["records"][0]["record"])).hexdigest()
    with pytest.raises(RegistryError, match="full numeric TrainConfig"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v2", wrong_config_digest)

    native_float_config = _competitive_campaign_v2_fixture()
    native_float_config["records"][0]["record"]["train_config"]["learning_rate"] = 0.0003
    with pytest.raises(RegistryError, match="registry JSON forbids floats"):
        registry_module._validate_data("g008_competitive_campaign_amendment_v2", native_float_config)


def test_competitive_campaign_v2_comparison_selects_mixed_only_on_preregistered_gain():
    events = []
    train_indices = {}
    validation_indices = {}
    for pair_no in range(1, 4):
        for role in ("baseline", "mixed"):
            train_indices[(pair_no, role)] = len(events)
            registration_id = f"train-{pair_no}-{role}"
            events.append({"event_index": len(events), "event_type": "registration",
                           "event_hash": f"{len(events):064x}", "event_data": {
                               "registration_id": registration_id, "registration_kind": "train",
                               "cycle_id": "g008-competitive-campaign-v2",
                               "record": {"run_uuid": f"run-{pair_no}-{role}"}}})
    for pair_no in range(1, 4):
        for role in ("baseline", "mixed"):
            validation_indices[(pair_no, role)] = len(events)
            events.append({"event_index": len(events), "event_type": "registration",
                           "event_hash": f"{len(events):064x}", "event_data": {
                               "registration_id": f"validation-{pair_no}-{role}",
                               "registration_kind": "validation", "cycle_id": "g008-competitive-campaign-v2",
                               "record": {"pair_id": f"seed-pair-{pair_no}", "strategy_role": role,
                                          "run_uuid": f"validation-run-{pair_no}-{role}"}}})
    artifact_indices = {}
    for pair_no in range(1, 4):
        for role in ("baseline", "mixed"):
            train_index = train_indices[(pair_no, role)]
            artifact_indices[(pair_no, role)] = len(events)
            events.append({"event_index": len(events), "event_type": "competitive_training_artifact_v2",
                           "event_hash": f"{len(events):064x}", "event_data": {
                               "training_registration_event_index": train_index,
                               "checkpoint_sha256": f"{train_index + 1:064x}",
                               "output_manifest_sha256": f"{train_index + 20:064x}"}})
    baseline_wins = [205, 256, 307]
    mixed_wins = [256, 307, 358]
    for pair_no in range(1, 4):
        for role in ("baseline", "mixed"):
            train_index = train_indices[(pair_no, role)]
            validation_index = validation_indices[(pair_no, role)]
            wins = (baseline_wins if role == "baseline" else mixed_wins)[pair_no - 1]
            results_index = len(events)
            events.append({"event_index": results_index, "event_type": "competitive_validation_result_v2",
                           "event_hash": f"{results_index:064x}", "event_data": {
                               "training_registration_event_index": train_index,
                               "training_attempt_event_index": train_index,
                               "validation_registration_event_index": validation_index,
                               "validation_attempt_event_index": validation_index,
                               "wins": wins, "draws": 0, "losses": 512 - wins}})

    comparison = registry_module._competitive_v2_comparison_payload(
        events, created_at_utc="2026-10-21T02:00:00Z")
    assert comparison["mixed_strategy_passed"] is True
    assert comparison["selected_strategy"] == "mixed"
    assert comparison["selected_candidate"]["pair_id"] == "seed-pair-2"
    assert comparison["selected_candidate"]["training_registration_event_index"] == train_indices[(2, "mixed")]
    assert comparison["selected_candidate"]["validation_registration_event_index"] == validation_indices[(2, "mixed")]


def test_competitive_campaign_v2_final_result_must_repeat_registered_roster_hashes():
    final_record = _competitive_campaign_v2_fixture()["records"][12]["record"]
    result = {"roster_sha256": final_record["roster_sha256"],
              "opponent_code_sha256": final_record["opponent_code_sha256"]}
    registry_module._validate_g008_v2_final_result_roster(final_record, result)

    with pytest.raises(RegistryError, match="differ from exact registered final roster"):
        registry_module._validate_g008_v2_final_result_roster(
            final_record, {**result, "roster_sha256": "a" * 64})


def test_competitive_campaign_v2_stale_head_does_not_mutate_registry(tmp_path, monkeypatch):
    registry = _campaign_rejected_registry(tmp_path, monkeypatch)
    amendment = _competitive_campaign_v2_fixture()
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="exact current parent head at event 54"):
        registry.register_g008_competitive_campaign_v2_batch(amendment)
    assert registry.path.read_bytes() == before


def _competitive_campaign_v2_replay_events(amendment):
    events = []

    def append(event_type, event_data):
        previous = events[-1]["event_hash"] if events else registry_module.GENESIS
        payload = registry_module._payload(len(events), event_type, event_data, previous)
        event = {**payload, "event_hash": registry_module._event_hash(payload)}
        events.append(event)
        return event

    for index in range(55):
        event_type, data = "placeholder", {}
        if index == 27:
            event_type = "recovery_cycle_amendment"
            data = {"project_gpu_seconds_before": 8_664, "project_output_bytes_before": 34_685,
                    "compute_caps_gpu_seconds": {"project_experiment": 21_600, "project_total": 28_800},
                    "output_caps_bytes": {"aggregate": 5_000_000_000}}
        elif index == 29:
            event_type = "registration"
            data = {"registration_id": "historical-train", "registration_kind": "train",
                    "record": {"run_uuid": "historical-v17"}}
        elif index == 33:
            event_type = "workload_attempt"
            data = {"outcome": "terminal", "actual_gpu_seconds": 509, "output_bytes": 13_725_702}
        elif index == 42:
            event_type = "exploratory_kernel_observation_v1"
            data = {"actual_gpu_seconds": 10, "output_bytes": 13_464_938}
        elif index == 48:
            event_type = "benchmark_artifact"
        elif index == 49:
            event_type = "registration_closed"
        elif index == 50:
            event_type = "benchmark_attestation"
            data = {"selection_evidence": {"selected_variant_id": "incumbent"}}
        elif index == 54:
            event_type = "workload_attempt"
            data = {"registration_event_index": 29, "registration_event_hash": events[29]["event_hash"],
                    "outcome": "terminal", "terminal_status": "COMPLETE", "assigned_version": 17,
                    "implementation_variant_id": "incumbent", "actual_gpu_seconds": 59, "output_bytes": 0,
                    "artifact_sha256": None}
        append(event_type, data)

    amendment = dict(amendment)
    amendment.update({"prior_head_hash": events[54]["event_hash"],
                      "benchmark_artifact_event_hash": events[48]["event_hash"],
                      "benchmark_lock_event_hash": events[49]["event_hash"],
                      "benchmark_attestation_event_hash": events[50]["event_hash"],
                      "selected_training_registration_event_hash": events[29]["event_hash"],
                      "selected_training_attempt_event_hash": events[54]["event_hash"],
                      "selected_training_run_uuid": "historical-v17"})
    append("g008_competitive_campaign_amendment_v2", amendment)
    for row in amendment["records"]:
        record = dict(row["record"])
        record.update({"competitive_campaign_v2_event_index": 55,
                       "competitive_campaign_v2_event_hash": events[55]["event_hash"]})
        append("registration", {"registration_id": row["registration_id"],
                                 "registration_kind": row["registration_kind"],
                                 "record": record, "cycle_id": "g008-competitive-campaign-v2"})
    return events


def test_competitive_campaign_v2_replay_requires_atomic_thirteen_registration_batch(monkeypatch):
    monkeypatch.setattr(registry_module, "_validate_recovery_cycle3_retry_suffix", lambda _events: None)
    events = _competitive_campaign_v2_replay_events(_competitive_campaign_v2_fixture())
    registry_module._validate_g008_competitive_campaign_v2(events)

    incomplete = events[:-2]
    with pytest.raises(RegistryError, match="all six train, six validation, and final"):
        registry_module._validate_g008_competitive_campaign_v2(incomplete)


def _append_campaign_v2_test_event(events, event_type, data):
    previous = events[-1]["event_hash"]
    payload = registry_module._payload(len(events), event_type, data, previous)
    event = {**payload, "event_hash": registry_module._event_hash(payload)}
    events.append(event)
    return event


def _campaign_v2_revision_lifecycle(amendment, *, retry_status="COMPLETE", retry_gpu_seconds=500):
    events = _competitive_campaign_v2_replay_events(amendment)
    ri = 56
    registration = events[ri]
    record = registration["event_data"]["record"]
    lock_hash = events[49]["event_hash"]

    def attempt(attempt_id, package_hash, expected_version, assigned_version, status, gpu_seconds):
        preflight = _append_campaign_v2_test_event(events, "workload_preflight", {
            "cycle_id": "g008-competitive-campaign-v2", "attempt_id": attempt_id,
            "registration_event_index": ri, "registration_event_hash": registration["event_hash"],
            "registration_id": registration["event_data"]["registration_id"],
            "kernel_id": "pascalcoimbra/circuit-ai", "package_sha256": package_hash,
            "source_sha256": record["source_sha256"], "config_sha256": record["config_sha256"],
            "seed_plan_sha256": record["seed_plan_sha256"], "backend_sha256": record["backend_sha256"],
            "trainer_sha256": record["trainer_sha256"], "engine_sha256": record["engine_sha256"],
            "session_timeout_seconds": 1_200, "expected_current_version": expected_version,
            "implementation_variant_id": "baseline", "lock_head_hash": lock_hash,
        })
        dispatch = _append_campaign_v2_test_event(events, "workload_push_dispatched", {
            "cycle_id": "g008-competitive-campaign-v2", "attempt_id": attempt_id,
            "registration_event_index": ri, "registration_event_hash": registration["event_hash"],
            "preflight_event_index": preflight["event_index"], "preflight_event_hash": preflight["event_hash"],
            "package_sha256": package_hash, "source_sha256": record["source_sha256"],
            "config_sha256": record["config_sha256"], "seed_plan_sha256": record["seed_plan_sha256"],
            "backend_sha256": record["backend_sha256"], "trainer_sha256": record["trainer_sha256"],
            "engine_sha256": record["engine_sha256"],
        })
        accepted = _append_campaign_v2_test_event(events, "workload_push_accepted", {
            "cycle_id": "g008-competitive-campaign-v2", "attempt_id": attempt_id,
            "registration_event_index": ri, "registration_event_hash": registration["event_hash"],
            "dispatch_event_hash": dispatch["event_hash"], "preflight_event_index": preflight["event_index"],
            "implementation_variant_id": "baseline", "lock_head_hash": lock_hash,
            "assigned_version": assigned_version,
        })
        terminal_data = {
            "cycle_id": "g008-competitive-campaign-v2", "attempt_id": attempt_id,
            "registration_event_index": ri, "registration_event_hash": registration["event_hash"],
            "preflight_event_index": preflight["event_index"], "preflight_event_hash": preflight["event_hash"],
            "dispatch_event_index": dispatch["event_index"], "dispatch_event_hash": dispatch["event_hash"],
            "accepted_event_index": accepted["event_index"], "accepted_event_hash": accepted["event_hash"],
            "assigned_version": assigned_version, "implementation_variant_id": "baseline",
            "package_sha256": package_hash, "source_sha256": record["source_sha256"],
            "config_sha256": record["config_sha256"], "seed_plan_sha256": record["seed_plan_sha256"],
            "backend_sha256": record["backend_sha256"], "trainer_sha256": record["trainer_sha256"],
            "engine_sha256": record["engine_sha256"], "outcome": "terminal",
            "terminal_status": status, "actual_gpu_seconds": gpu_seconds,
            "output_bytes": 80_238_970 if status == "ERROR" else 12_000_000,
        }
        return _append_campaign_v2_test_event(events, "workload_attempt", terminal_data)

    failed = attempt("failed-v18", record["package_sha256"], 17, 18, "ERROR", 516)
    bindings = []
    for train_index in range(56, 62):
        train_record = events[train_index]["event_data"]["record"]
        bindings.append({
            "registration_event_index": train_index,
            "registration_event_hash": events[train_index]["event_hash"],
            "package_sha256": "a" * 64,
            **{key: train_record[key] for key in (
                "source_sha256", "config_sha256", "seed_plan_sha256", "backend_sha256",
                "trainer_sha256", "engine_sha256")},
        })
    revision_data = {
        "schema_id": "g008-competitive-campaign-v2-package-revision", "schema_version": 1,
        "campaign_id": "g008-competitive-20261021-v2", "created_at_utc": "2026-10-21T03:00:00Z",
        "amendment_event_index": 55, "amendment_event_hash": events[55]["event_hash"],
        "prior_head_hash": failed["event_hash"], "failed_registration_event_index": ri,
        "failed_registration_event_hash": registration["event_hash"],
        "failed_attempt_event_index": failed["event_index"], "failed_attempt_event_hash": failed["event_hash"],
        "package_bindings": bindings,
    }
    revision = _append_campaign_v2_test_event(
        events, "g008_competitive_campaign_v2_package_revision_v1", revision_data)
    cycle3_caps = events[27]["event_data"]["compute_caps_gpu_seconds"]
    retry_budget_data = {
        "schema_id": "g008-competitive-campaign-v2-retry-budget", "schema_version": 1,
        "campaign_id": "g008-competitive-20261021-v2", "created_at_utc": "2026-10-21T03:01:00Z",
        "prior_head_hash": revision["event_hash"],
        "package_revision_event_index": revision["event_index"],
        "package_revision_event_hash": revision["event_hash"],
        "failed_attempt_event_index": failed["event_index"],
        "failed_attempt_event_hash": failed["event_hash"],
        "failed_registration_event_index": ri,
        "failed_registration_event_hash": registration["event_hash"],
        "retry_attempt_id": "g008-v2-pair1-baseline-a2",
        "retry_timeout_seconds": 1_200, "additional_gpu_seconds": 516,
        "effective_training_gpu_seconds_cap": 7_716,
        "effective_evaluation_gpu_seconds_cap": 4_200,
        "effective_total_gpu_seconds_cap": 11_916,
        "project_experiment_gpu_seconds_cap": cycle3_caps["project_experiment"],
        "project_total_gpu_seconds_cap": cycle3_caps["project_total"],
        "protected_final_reserve_gpu_seconds": 7_200,
    }
    retry_budget = _append_campaign_v2_test_event(
        events, "g008_competitive_campaign_v2_retry_budget_v1", retry_budget_data)
    retried = attempt("g008-v2-pair1-baseline-a2", "a" * 64, 18, 19,
                      retry_status, retry_gpu_seconds)
    if retry_status == "COMPLETE":
        _append_campaign_v2_test_event(events, "competitive_training_artifact_v2", {
            "training_registration_event_index": ri, "training_registration_event_hash": registration["event_hash"],
            "training_attempt_event_index": retried["event_index"],
            "training_attempt_event_hash": retried["event_hash"], "step": 1_000,
            "package_sha256": "a" * 64, "source_sha256": record["source_sha256"],
            "config_sha256": record["config_sha256"],
            "schedule_descriptor_sha256": record["schedule_descriptor_sha256"],
        })
    return events, revision, failed, {"revision": revision_data, "retry_budget": retry_budget_data,
                                      "retry_budget_event": retry_budget}


def _campaign_v2_second_retry(events, *, package_hash="31500d68f6fcbdb4eded6247e20b83d10b450f08915f1bf21e7534b865329906"):
    registration = events[56]
    record = registration["event_data"]["record"]
    cycle3_caps = events[27]["event_data"]["compute_caps_gpu_seconds"]
    failed = events[78]
    first_revision = events[73]
    first_budget = events[74]
    binding = {
        "registration_event_index": 56,
        "registration_event_hash": registration["event_hash"],
        "package_sha256": package_hash,
        "source_sha256": "3ac7f5bcc24a62f9e5015d2e77bbb270f075802fffc8b2f50bdfe256830f8f52",
        "config_sha256": record["config_sha256"],
        "seed_plan_sha256": record["seed_plan_sha256"],
        "backend_sha256": record["backend_sha256"],
        "trainer_sha256": "30f571f9f225b7ce88257e59cea772972488a88f9260b3ef24a3c6a21765f2a7",
        "engine_sha256": record["engine_sha256"],
    }
    revision_data = {
        "schema_id": "g008-competitive-campaign-v2-package-revision", "schema_version": 2,
        "campaign_id": "g008-competitive-20261021-v2", "created_at_utc": "2026-10-21T04:00:00Z",
        "prior_head_hash": failed["event_hash"],
        "prior_package_revision_event_index": first_revision["event_index"],
        "prior_package_revision_event_hash": first_revision["event_hash"],
        "prior_retry_budget_event_index": first_budget["event_index"],
        "prior_retry_budget_event_hash": first_budget["event_hash"],
        "failed_registration_event_index": 56,
        "failed_registration_event_hash": registration["event_hash"],
        "failed_attempt_event_index": failed["event_index"],
        "failed_attempt_event_hash": failed["event_hash"],
        "package_binding": binding,
    }
    revision = _append_campaign_v2_test_event(
        events, "g008_competitive_campaign_v2_package_revision_v2", revision_data)
    budget_data = {
        "schema_id": "g008-competitive-campaign-v2-retry-budget", "schema_version": 2,
        "campaign_id": "g008-competitive-20261021-v2", "created_at_utc": "2026-10-21T04:01:00Z",
        "prior_head_hash": revision["event_hash"],
        "package_revision_event_index": revision["event_index"],
        "package_revision_event_hash": revision["event_hash"],
        "prior_retry_budget_event_index": first_budget["event_index"],
        "prior_retry_budget_event_hash": first_budget["event_hash"],
        "failed_attempt_event_index": failed["event_index"],
        "failed_attempt_event_hash": failed["event_hash"],
        "failed_registration_event_index": 56,
        "failed_registration_event_hash": registration["event_hash"],
        "retry_attempt_id": "g008-v2-pair1-baseline-a3",
        "retry_timeout_seconds": 1_200, "additional_gpu_seconds": 36,
        "effective_training_gpu_seconds_cap": 7_752,
        "effective_evaluation_gpu_seconds_cap": 4_200,
        "effective_total_gpu_seconds_cap": 11_952,
        "project_experiment_gpu_seconds_cap": cycle3_caps["project_experiment"],
        "project_total_gpu_seconds_cap": cycle3_caps["project_total"],
        "protected_final_reserve_gpu_seconds": 7_200,
    }
    budget = _append_campaign_v2_test_event(
        events, "g008_competitive_campaign_v2_retry_budget_v2", budget_data)
    preflight = _append_campaign_v2_test_event(events, "workload_preflight", {
        "cycle_id": "g008-competitive-campaign-v2", "attempt_id": "g008-v2-pair1-baseline-a3",
        "registration_event_index": 56, "registration_event_hash": registration["event_hash"],
        "registration_id": registration["event_data"]["registration_id"],
        "kernel_id": "pascalcoimbra/circuit-ai", "package_sha256": package_hash,
        "source_sha256": binding["source_sha256"], "config_sha256": binding["config_sha256"],
        "seed_plan_sha256": binding["seed_plan_sha256"], "backend_sha256": binding["backend_sha256"],
        "trainer_sha256": binding["trainer_sha256"], "engine_sha256": binding["engine_sha256"],
        "session_timeout_seconds": 1_200, "expected_current_version": 19,
        "implementation_variant_id": "baseline", "lock_head_hash": events[49]["event_hash"],
    })
    return revision, budget, preflight


def test_competitive_campaign_v2_package_revision_rejects_wrong_error_and_component_mutation(monkeypatch):
    monkeypatch.setattr(registry_module, "_validate_recovery_cycle3_retry_suffix", lambda _events: None)
    events, revision, failed, _ = _campaign_v2_revision_lifecycle(_competitive_campaign_v2_fixture())

    wrong_error = [dict(event) for event in events]
    wrong_data = dict(wrong_error[failed["event_index"]]["event_data"])
    wrong_data["terminal_status"] = "COMPLETE"
    wrong_error[failed["event_index"]] = {
        **wrong_error[failed["event_index"]], "event_data": wrong_data,
    }
    revision_event_index = next(i for i, event in enumerate(wrong_error)
                                if event["event_type"] == "g008_competitive_campaign_v2_package_revision_v1")
    revision_data = dict(wrong_error[revision_event_index]["event_data"])
    wrong_error[revision_event_index] = {
        **wrong_error[revision_event_index], "event_data": revision_data,
    }
    # Rehash the suffix because the tampered terminal must remain a valid append-only chain.
    previous = wrong_error[failed["event_index"] - 1]["event_hash"]
    for index in range(failed["event_index"], len(wrong_error)):
        if index == revision_event_index:
            wrong_error[index]["event_data"]["prior_head_hash"] = previous
            wrong_error[index]["event_data"]["failed_attempt_event_hash"] = previous
        payload = registry_module._payload(index, wrong_error[index]["event_type"],
                                           wrong_error[index]["event_data"], previous)
        wrong_error[index] = {**payload, "event_hash": registry_module._event_hash(payload)}
        previous = wrong_error[index]["event_hash"]
    with pytest.raises(RegistryError, match="terminal ERROR"):
        registry_module._validate_g008_competitive_campaign_v2(wrong_error)

    changed_component = [dict(event) for event in events]
    revision_event = changed_component[revision["event_index"]]
    changed_data = dict(revision_event["event_data"])
    bindings = [dict(binding) for binding in changed_data["package_bindings"]]
    bindings[0]["trainer_sha256"] = "b" * 64
    changed_data["package_bindings"] = bindings
    changed_component[revision["event_index"]] = {**revision_event, "event_data": changed_data}
    previous = changed_component[revision["event_index"] - 1]["event_hash"]
    for index in range(revision["event_index"], len(changed_component)):
        payload = registry_module._payload(index, changed_component[index]["event_type"],
                                           changed_component[index]["event_data"], previous)
        changed_component[index] = {**payload, "event_hash": registry_module._event_hash(payload)}
        previous = changed_component[index]["event_hash"]
    with pytest.raises(RegistryError, match="component hash"):
        registry_module._validate_g008_competitive_campaign_v2(changed_component)


def test_competitive_campaign_v2_package_revision_retries_with_effective_package(monkeypatch):
    monkeypatch.setattr(registry_module, "_validate_recovery_cycle3_retry_suffix", lambda _events: None)
    events, revision, _, _ = _campaign_v2_revision_lifecycle(_competitive_campaign_v2_fixture())
    registry_module._validate_g008_competitive_campaign_v2(events)
    assert events[revision["event_index"]]["event_type"] == "g008_competitive_campaign_v2_package_revision_v1"
    assert events[-1]["event_type"] == "competitive_training_artifact_v2"
    assert events[-1]["event_data"]["package_sha256"] == "a" * 64


def test_competitive_campaign_v2_retry_budget_is_exactly_bound_and_single_use(monkeypatch):
    monkeypatch.setattr(registry_module, "_validate_recovery_cycle3_retry_suffix", lambda _events: None)
    events, _, _, context = _campaign_v2_revision_lifecycle(_competitive_campaign_v2_fixture())
    registry_module._validate_g008_competitive_campaign_v2(events)
    budget_index = context["retry_budget_event"]["event_index"]
    assert events[budget_index]["event_type"] == "g008_competitive_campaign_v2_retry_budget_v1"

    def tamper_and_rehash(field, value):
        changed = [dict(event) for event in events]
        changed_data = dict(changed[budget_index]["event_data"])
        changed_data[field] = value
        previous = changed[budget_index - 1]["event_hash"]
        for index in range(budget_index, len(changed)):
            data = changed_data if index == budget_index else changed[index]["event_data"]
            payload = registry_module._payload(index, changed[index]["event_type"], data, previous)
            changed[index] = {**payload, "event_hash": registry_module._event_hash(payload)}
            previous = changed[index]["event_hash"]
        return changed

    with pytest.raises(RegistryError, match="retry budget authorization is malformed"):
        registry_module._validate_g008_competitive_campaign_v2(
            tamper_and_rehash("additional_gpu_seconds", 515))
    with pytest.raises(RegistryError, match="retry budget authorization is malformed"):
        registry_module._validate_g008_competitive_campaign_v2(
            tamper_and_rehash("retry_attempt_id", "unregistered-second-retry"))
    with pytest.raises(RegistryError, match="retry budget must bind"):
        registry_module._validate_g008_competitive_campaign_v2(
            tamper_and_rehash("prior_head_hash", "0" * 64))

    second_retry = [dict(event) for event in events[:budget_index + 1]]
    _append_campaign_v2_test_event(second_retry, "workload_preflight", {
        **events[budget_index + 1]["event_data"], "attempt_id": "g008-v2-pair1-baseline-a3",
    })
    with pytest.raises(RegistryError, match="preflight changed registration"):
        registry_module._validate_g008_competitive_campaign_v2(second_retry)


def test_competitive_campaign_v2_retry_budget_reserves_full_timeout(monkeypatch):
    monkeypatch.setattr(registry_module, "_validate_recovery_cycle3_retry_suffix", lambda _events: None)
    events, _, _, context = _campaign_v2_revision_lifecycle(_competitive_campaign_v2_fixture())
    retry_preflight_index = context["retry_budget_event"]["event_index"] + 1
    retry_preflight = events[retry_preflight_index]
    assert retry_preflight["event_type"] == "workload_preflight"
    pending_retry = events[:retry_preflight_index + 1]
    # The failed 516 GPU-s remain actual spend, and the retried job reserves its full 1,200-s timeout.
    registry_module._validate_g008_competitive_campaign_v2(pending_retry)


def test_competitive_campaign_v2_second_error_allows_only_exact_a3_retry(monkeypatch):
    monkeypatch.setattr(registry_module, "_validate_recovery_cycle3_retry_suffix", lambda _events: None)
    events, _, _, _ = _campaign_v2_revision_lifecycle(
        _competitive_campaign_v2_fixture(), retry_status="ERROR", retry_gpu_seconds=36)
    revision, budget, preflight = _campaign_v2_second_retry(events)
    registry_module._validate_g008_competitive_campaign_v2(events)
    assert revision["event_index"] == 79
    assert budget["event_index"] == 80
    assert preflight["event_index"] == 81
    assert events[56]["event_data"]["record"]["package_sha256"] != events[revision["event_index"]]["event_data"]["package_binding"]["package_sha256"]
    assert events[preflight["event_index"]]["event_data"]["package_sha256"] == "31500d68f6fcbdb4eded6247e20b83d10b450f08915f1bf21e7534b865329906"

    binding = events[revision["event_index"]]["event_data"]["package_binding"]
    dispatch = _append_campaign_v2_test_event(events, "workload_push_dispatched", {
        "cycle_id": "g008-competitive-campaign-v2", "attempt_id": "g008-v2-pair1-baseline-a3",
        "registration_event_index": 56, "registration_event_hash": events[56]["event_hash"],
        "preflight_event_index": preflight["event_index"], "preflight_event_hash": preflight["event_hash"],
        **{key: binding[key] for key in ("package_sha256", "source_sha256", "config_sha256",
                                         "seed_plan_sha256", "backend_sha256", "trainer_sha256",
                                         "engine_sha256")},
    })
    accepted = _append_campaign_v2_test_event(events, "workload_push_accepted", {
        "cycle_id": "g008-competitive-campaign-v2", "attempt_id": "g008-v2-pair1-baseline-a3",
        "registration_event_index": 56, "registration_event_hash": events[56]["event_hash"],
        "dispatch_event_hash": dispatch["event_hash"], "preflight_event_index": preflight["event_index"],
        "implementation_variant_id": "baseline", "lock_head_hash": events[49]["event_hash"],
        "assigned_version": 20,
    })
    terminal = _append_campaign_v2_test_event(events, "workload_attempt", {
        "cycle_id": "g008-competitive-campaign-v2", "attempt_id": "g008-v2-pair1-baseline-a3",
        "registration_event_index": 56, "registration_event_hash": events[56]["event_hash"],
        "preflight_event_index": preflight["event_index"], "preflight_event_hash": preflight["event_hash"],
        "dispatch_event_index": dispatch["event_index"], "dispatch_event_hash": dispatch["event_hash"],
        "accepted_event_index": accepted["event_index"], "accepted_event_hash": accepted["event_hash"],
        "assigned_version": 20, "implementation_variant_id": "baseline",
        **{key: binding[key] for key in ("package_sha256", "source_sha256", "config_sha256",
                                         "seed_plan_sha256", "backend_sha256", "trainer_sha256",
                                         "engine_sha256")},
        "outcome": "terminal", "terminal_status": "COMPLETE", "actual_gpu_seconds": 1_000,
        "output_bytes": 12_000_000,
    })
    extra_retry = list(events)
    extra_preflight_data = dict(preflight["event_data"])
    extra_preflight_data["attempt_id"] = "g008-v2-pair1-baseline-a4"
    _append_campaign_v2_test_event(extra_retry, "workload_preflight", extra_preflight_data)
    with pytest.raises(RegistryError, match="preflight changed registration"):
        registry_module._validate_g008_competitive_campaign_v2(extra_retry)
    artifact = {
        "schema_id": "g008-training-artifact-attestation", "schema_version": 2,
        "campaign_id": "g008-competitive-20261021-v2", "created_at_utc": "2026-10-21T05:00:00Z",
        "training_registration_event_index": 56, "training_registration_event_hash": events[56]["event_hash"],
        "training_attempt_event_index": terminal["event_index"],
        "training_attempt_event_hash": terminal["event_hash"], "step": 1_000,
        "checkpoint_sha256": "1" * 64, "output_manifest_sha256": "2" * 64,
        "metrics_sha256": "3" * 64, "package_sha256": binding["package_sha256"],
        "source_sha256": binding["source_sha256"], "config_sha256": binding["config_sha256"],
        "schedule_descriptor_sha256": events[56]["event_data"]["record"]["schedule_descriptor_sha256"],
    }
    _append_campaign_v2_test_event(events, "competitive_training_artifact_v2", artifact)
    registry_module._validate_g008_competitive_campaign_v2(events)

    wrong_attempt = [dict(event) for event in events]
    changed = dict(wrong_attempt[preflight["event_index"]]["event_data"])
    changed["attempt_id"] = "g008-v2-pair1-baseline-a4"
    payload = registry_module._payload(preflight["event_index"], "workload_preflight", changed,
                                       wrong_attempt[preflight["event_index"] - 1]["event_hash"])
    wrong_attempt[preflight["event_index"]] = {
        **payload, "event_hash": registry_module._event_hash(payload),
    }
    with pytest.raises(RegistryError, match="preflight changed registration"):
        registry_module._validate_g008_competitive_campaign_v2(wrong_attempt)


def test_competitive_campaign_v2_second_retry_rejects_changed_error_and_package(monkeypatch):
    monkeypatch.setattr(registry_module, "_validate_recovery_cycle3_retry_suffix", lambda _events: None)
    events, _, _, _ = _campaign_v2_revision_lifecycle(
        _competitive_campaign_v2_fixture(), retry_status="ERROR", retry_gpu_seconds=36)
    revision, _, _ = _campaign_v2_second_retry(events)

    bad_error = [dict(event) for event in events]
    data = dict(bad_error[78]["event_data"])
    data["actual_gpu_seconds"] = 35
    payload = registry_module._payload(78, "workload_attempt", data, bad_error[77]["event_hash"])
    bad_error[78] = {**payload, "event_hash": registry_module._event_hash(payload)}
    # Rebuild the anchored suffix so the rejection exercises semantic provenance checks.
    previous = bad_error[78]["event_hash"]
    for index in range(79, len(bad_error)):
        suffix_data = dict(bad_error[index]["event_data"])
        if index == 79:
            suffix_data["prior_head_hash"] = previous
            suffix_data["failed_attempt_event_hash"] = previous
        elif index == 80:
            suffix_data["prior_head_hash"] = bad_error[79]["event_hash"]
            suffix_data["package_revision_event_hash"] = bad_error[79]["event_hash"]
            suffix_data["failed_attempt_event_hash"] = previous
        elif index == 81:
            suffix_data["package_sha256"] = bad_error[79]["event_data"]["package_binding"]["package_sha256"]
        payload = registry_module._payload(index, bad_error[index]["event_type"], suffix_data, previous)
        bad_error[index] = {**payload, "event_hash": registry_module._event_hash(payload)}
        previous = bad_error[index]["event_hash"]
    with pytest.raises(RegistryError, match="exact a2 ERROR"):
        registry_module._validate_g008_competitive_campaign_v2(bad_error)

    wrong_package = [dict(event) for event in events]
    package_data = dict(wrong_package[revision["event_index"]]["event_data"])
    binding = dict(package_data["package_binding"])
    binding["package_sha256"] = "f" * 64
    package_data["package_binding"] = binding
    with pytest.raises(RegistryError, match="schema or row-56 binding"):
        registry_module._validate_data("g008_competitive_campaign_v2_package_revision_v2", package_data)

