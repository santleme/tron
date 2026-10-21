from pathlib import Path

import pytest

from tools.kaggle.cli import KaggleError, _recovery_record, _status_is_waiting_for_worker
from training.eval.registry import Registry


REGISTRY_PATH = Path(__file__).resolve().parents[2] / ".omx/evidence/g008-experiment-registry.jsonl"


def _events_with_final_revision():
    events = Registry(REGISTRY_PATH).events()[:171]
    registration = events[68]
    freeze = events[170]
    candidate_train = events[freeze["event_data"]["train_attempt_event_index"]]
    record = registration["event_data"]["record"]
    binding = {key: record[key] for key in (
        "config_sha256", "seed_plan_sha256", "backend_sha256", "trainer_sha256", "engine_sha256",
    )}
    binding.update({
        "registration_event_index": registration["event_index"],
        "registration_event_hash": registration["event_hash"],
        "package_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "trainer_sha256": candidate_train["event_data"]["trainer_sha256"],
    })
    revision = {
        "event_type": "g008_competitive_campaign_v2_final_package_revision_v1",
        "event_data": {
            "campaign_id": record["campaign_id"],
            "registration_event_index": registration["event_index"],
            "registration_event_hash": registration["event_hash"],
            "original_package_sha256": record["package_sha256"],
            "original_source_sha256": record["source_sha256"],
            "candidate_freeze_event_index": freeze["event_index"],
            "candidate_freeze_event_hash": freeze["event_hash"],
            "candidate_checkpoint_sha256": freeze["event_data"]["checkpoint_sha256"],
            "candidate_train_attempt_event_index": candidate_train["event_index"],
            "candidate_train_attempt_event_hash": candidate_train["event_hash"],
            "candidate_trainer_sha256": candidate_train["event_data"]["trainer_sha256"],
            "package_binding": binding,
        },
    }
    return events + [revision], revision


def test_final_recovery_uses_append_only_package_binding():
    events, _ = _events_with_final_revision()
    record, hashes = _recovery_record(events[68], "frozen-candidate", events)

    assert record["registration_kind"] == "final"
    assert hashes["package_sha256"] == "1" * 64
    assert hashes["source_sha256"] == "2" * 64
    assert hashes["trainer_sha256"] == events[100]["event_data"]["trainer_sha256"]
    assert hashes["config_sha256"] == events[68]["event_data"]["record"]["config_sha256"]


def test_final_recovery_rejects_component_drift():
    events, revision = _events_with_final_revision()
    revision["event_data"]["package_binding"]["config_sha256"] = "3" * 64

    with pytest.raises(KaggleError, match="preserve the frozen row"):
        _recovery_record(events[68], "frozen-candidate", events)


@pytest.mark.parametrize("status", ["QUEUED", "PENDING", "STARTING", "NEW_SCRIPT"])
def test_kaggle_queue_states_do_not_consume_session_budget(status):
    assert _status_is_waiting_for_worker({"status": status})


def test_running_worker_consumes_session_budget():
    assert not _status_is_waiting_for_worker({"status": "RUNNING"})
