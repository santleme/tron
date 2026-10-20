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

