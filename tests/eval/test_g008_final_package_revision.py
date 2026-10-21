from pathlib import Path

import pytest

from training.eval.registry import Registry, RegistryError


SOURCE_REGISTRY = ".omx/evidence/g008-experiment-registry.jsonl"


def _registry_copy(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "registry.jsonl"
    lines = Path(SOURCE_REGISTRY).read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:171]) + "\n", encoding="utf-8")
    return Registry(path)


def _revision(registry):
    events = registry.events()
    registration = events[68]
    freeze = events[170]
    candidate_train = events[freeze["event_data"]["train_attempt_event_index"]]
    original = registration["event_data"]["record"]
    binding = {key: original[key] for key in (
        "config_sha256", "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256")}
    binding.update({
        "registration_event_index": 68,
        "registration_event_hash": registration["event_hash"],
        "package_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "trainer_sha256": candidate_train["event_data"]["trainer_sha256"],
    })
    return {
        "schema_id": "g008-competitive-campaign-v2-final-package-revision",
        "schema_version": 1,
        "campaign_id": "g008-competitive-20261021-v2",
        "created_at_utc": "2026-10-21T12:00:00Z",
        "prior_head_event_index": 170,
        "prior_head_event_hash": events[170]["event_hash"],
        "candidate_freeze_event_index": 170,
        "candidate_freeze_event_hash": freeze["event_hash"],
        "candidate_checkpoint_sha256": freeze["event_data"]["checkpoint_sha256"],
        "candidate_train_attempt_event_index": candidate_train["event_index"],
        "candidate_train_attempt_event_hash": candidate_train["event_hash"],
        "candidate_trainer_sha256": candidate_train["event_data"]["trainer_sha256"],
        "registration_event_index": 68,
        "registration_event_hash": registration["event_hash"],
        "original_package_sha256": original["package_sha256"],
        "original_source_sha256": original["source_sha256"],
        "package_binding": binding,
    }


def test_final_package_revision_rebinds_only_package_and_source(tmp_path):
    registry = _registry_copy(tmp_path)
    revision = _revision(registry)

    event = registry.revise_g008_competitive_campaign_v2_final_package(revision)

    assert event["event_index"] == 171
    assert registry.head == (event["event_hash"], 171)
    assert registry.events()[68]["event_data"]["record"]["package_sha256"] == revision["original_package_sha256"]


def test_final_package_revision_rejects_wrong_head_or_frozen_binding(tmp_path):
    registry = _registry_copy(tmp_path)
    revision = _revision(registry)
    revision["prior_head_event_hash"] = "c" * 64
    with pytest.raises(RegistryError, match="undispatched row-68"):
        registry.revise_g008_competitive_campaign_v2_final_package(revision)

    registry = _registry_copy(tmp_path / "second")
    revision = _revision(registry)
    revision["candidate_checkpoint_sha256"] = "d" * 64
    with pytest.raises(RegistryError, match="undispatched row-68"):
        registry.revise_g008_competitive_campaign_v2_final_package(revision)


def test_final_package_revision_is_rejected_after_dispatch(tmp_path):
    registry = _registry_copy(tmp_path)
    record = registry.events()[68]["event_data"]["record"]
    attempt_id = "g008-v2-final-suite-a1"
    preflight = registry.preflight_workload({
        "cycle_id": "g008-competitive-campaign-v2",
        "attempt_id": attempt_id,
        "registration_event_index": 68,
        "registration_event_hash": registry.events()[68]["event_hash"],
        "registration_id": registry.events()[68]["event_data"]["registration_id"],
        "kernel_id": record["kernel_id"],
        "package_sha256": record["package_sha256"],
        "source_sha256": record["source_sha256"],
        "config_sha256": record["config_sha256"],
        "seed_plan_sha256": record["seed_plan_sha256"],
        "backend_sha256": record["backend_sha256"],
        "trainer_sha256": record["trainer_sha256"],
        "engine_sha256": record["engine_sha256"],
        "session_timeout_seconds": record["session_timeout_seconds"],
        "accelerator_count": 1,
        "reserved_gpu_seconds": 600,
        "expected_current_version": 32,
        "quota_evidence": {"remaining_gpu_seconds": 50000},
        "policy_evidence": {"paid_services": False, "billing_changes": False,
                             "machine_shape": "NvidiaTeslaT4", "prior_runs_terminal": True,
                             "no_queued_or_running": True},
        "prior_attempts": [],
        "implementation_variant_id": "frozen-candidate",
        "no_outside_writer_confirmed": True,
        "exclusive_lock_id": "test-writer-lock",
        "reconciliation_slack_seconds": 0,
        "lock_head_hash": registry.events()[49]["event_hash"],
    })
    registry.dispatch_workload_push({
        "cycle_id": "g008-competitive-campaign-v2",
        "attempt_id": attempt_id,
        "registration_event_index": 68,
        "registration_event_hash": registry.events()[68]["event_hash"],
        "preflight_event_index": preflight["event_index"],
        "preflight_event_hash": preflight["event_hash"],
        "kernel_id": record["kernel_id"],
        "package_sha256": record["package_sha256"],
        "source_sha256": record["source_sha256"],
        "config_sha256": record["config_sha256"],
        "seed_plan_sha256": record["seed_plan_sha256"],
        "backend_sha256": record["backend_sha256"],
        "trainer_sha256": record["trainer_sha256"],
        "engine_sha256": record["engine_sha256"],
        "session_timeout_seconds": 600,
        "accelerator_count": 1,
        "reserved_gpu_seconds": 600,
        "implementation_variant_id": "frozen-candidate",
        "dispatch_timestamp_utc": "2026-10-21T12:01:00Z",
        "request_identity": "test-request",
        "lock_head_hash": registry.events()[49]["event_hash"],
    })

    with pytest.raises(RegistryError):
        registry.revise_g008_competitive_campaign_v2_final_package(_revision(registry))
