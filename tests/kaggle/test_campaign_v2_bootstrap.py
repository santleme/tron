from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.kaggle import campaign_v2


def test_source_inventory_digest_is_stable_and_tracks_staged_code(tmp_path):
    package = tmp_path / "package"
    (package / "training/ppo").mkdir(parents=True)
    (package / "training/eval/assets").mkdir(parents=True)
    (package / "schemas").mkdir()
    (package / "training/ppo/train.py").write_text("trainer-v1", encoding="utf-8")
    (package / "training/eval/assets/g007-step-300.pt").write_bytes(b"opponent")
    (package / "schemas/schema.json").write_text("{}", encoding="utf-8")

    first = campaign_v2._source_inventory_sha256(package)
    assert first == campaign_v2._source_inventory_sha256(package)
    assert first == hashlib.sha256(json.dumps([
        {"path": "schemas/schema.json", "sha256": hashlib.sha256(b"{}").hexdigest()},
        {"path": "training/ppo/train.py", "sha256": hashlib.sha256(b"trainer-v1").hexdigest()},
    ], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

    (package / "training/ppo/train.py").write_text("trainer-v2", encoding="utf-8")
    assert campaign_v2._source_inventory_sha256(package) != first


def test_source_inventory_rejects_symlinks(tmp_path):
    package = tmp_path / "package"
    (package / "training").mkdir(parents=True)
    (package / "schemas").mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("unsafe", encoding="utf-8")
    (package / "training/linked.py").symlink_to(outside)
    with pytest.raises(campaign_v2.CampaignBootstrapError, match="symlink"):
        campaign_v2._source_inventory_sha256(package)


def test_campaign_run_uuids_are_stable_per_registration_id():
    first = campaign_v2._run_uuid("g008-v2-train-seed-pair-1-baseline")
    assert first == campaign_v2._run_uuid("g008-v2-train-seed-pair-1-baseline")
    assert first != campaign_v2._run_uuid("g008-v2-train-seed-pair-1-mixed")


def test_training_settings_bind_compact_schedule_and_exact_opponent_arm():
    from training.ppo.g008_benchmark import load_principal_config
    from dataclasses import replace

    frozen, _ = load_principal_config()
    config = replace(frozen, device="auto", seed=3301, total_updates=1_000,
                     checkpoint_every=50, snapshot_opponent_probability=0.1,
                     heuristic_opponent_probability=1 / 3)
    settings = campaign_v2._training_settings(
        "registration", "00000000-0000-4000-8000-000000000001", "seed-pair-1",
        "mixed", 2_100_001, 3301, config, campaign_v2.registry_module._RECOVERY_SEED_PLAN_SHA256)
    schedule = settings["g008_campaign_train"]
    assert settings["run_id"] == schedule["run_uuid"]
    assert schedule["opponent_probabilities"] == campaign_v2._POOL_BPS["mixed"]
    assert schedule["update_budget"] == 1_000
    assert len(schedule["schedule_descriptor_sha256"]) == 64
    assert settings["train_config"] == config.__dict__


def test_registration_rejects_mutated_package_before_quota_query_or_append(tmp_path):
    class FakeRegistry:
        def __init__(self):
            self.appended = False

        def events(self):
            events = [{"event_index": i, "event_type": "other", "event_hash": f"{i:064x}",
                       "event_data": {}} for i in range(55)]
            for index, event_type in ((29, "registration"), (48, "benchmark_artifact"),
                                      (49, "registration_closed"), (50, "benchmark_attestation"),
                                      (54, "workload_attempt")):
                events[index]["event_type"] = event_type
            events[54]["event_data"] = {"terminal_status": "COMPLETE"}
            return events

        def register_g008_competitive_campaign_v2_batch(self, amendment):
            self.appended = True
            raise AssertionError("append must not be reached")

    class FakeKaggleApi:
        def quota_view(self):
            raise AssertionError("quota must not be queried for a changed package")

    registry = FakeRegistry()
    package = tmp_path / "packages" / "row-1"
    package.mkdir(parents=True)
    (package / "run-config.json").write_text('{"run_id":"run-1"}', encoding="utf-8")
    amendment = {
        "prior_head_hash": f"{54:064x}",
        "benchmark_artifact_event_hash": f"{48:064x}",
        "benchmark_lock_event_hash": f"{49:064x}",
        "benchmark_attestation_event_hash": f"{50:064x}",
        "records": [{"registration_id": "row-1", "record": {
            "package_sha256": "0" * 64, "source_sha256": "1" * 64, "run_uuid": "run-1",
        }}],
    }
    with pytest.raises(campaign_v2.CampaignBootstrapError, match="package"):
        campaign_v2.register_campaign_v2(amendment, tmp_path, registry=registry, api=FakeKaggleApi())
    assert not registry.appended
