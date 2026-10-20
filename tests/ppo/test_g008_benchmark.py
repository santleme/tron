import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from training.ppo.g008_benchmark import (CONFIG_PATH, MEASURED_REPEATS, WARMUP_REPEATS,
                                         load_principal_config, prepare_benchmark_schedules,
                                         run_principal_benchmark)
from training.eval.provenance import benchmark_run_keys, derive_environment_seed
from training.ppo.train import G008EnvironmentSeedSchedule


def principal_document():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def run_ids():
    return {key: str(uuid.uuid5(uuid.NAMESPACE_URL, "g008-test:" + key))
            for key in benchmark_run_keys()}


class G008PrincipalBenchmarkTests(unittest.TestCase):
    def test_frozen_config_digest_and_exact_fields(self):
        config, document = load_principal_config()
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.total_updates, 3)
        encoded = json.dumps(document["config"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        self.assertEqual(document["config_sha256"], __import__("hashlib").sha256(encoded).hexdigest())

    def test_rejects_nested_digest_mismatch_and_field_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            doc = principal_document()
            doc["config"]["seed"] += 1
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_principal_config(path)
            doc = principal_document()
            doc["config"]["surprise"] = 1
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "fields differ"):
                load_principal_config(path)

    def test_rejects_non_cuda_principal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            doc = principal_document()
            doc["config"]["device"] = "cpu"
            import hashlib
            encoded = json.dumps(doc["config"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            doc["config_sha256"] = hashlib.sha256(encoded).hexdigest()
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "frozen G008 baseline digest"):
                load_principal_config(path)

    def test_runs_two_warmups_then_five_three_update_measurements(self):
        calls = []
        ids = run_ids()

        def fake_train(config, output, **kwargs):
            output = Path(output)
            calls.append((output.parts[-2:], config.total_updates, kwargs))
            output.mkdir(parents=True, exist_ok=True)
            (output / "checkpoints").mkdir()
            (output / "checkpoints" / "step-0.pt").write_bytes(b"ephemeral")
            logs = [{"optimizer_updates": 4, "cuda_updates": 4, "device": "cuda", "optimizer_seconds": .1,
                     "update_latency_seconds": .4, "checkpoint_seconds": .05, "rollout_seconds": .2,
                     "mean_loss": 9.0, "episodes_completed": 17,
                     "snapshot_opponent_arenas": 3, "latest_opponent_arenas": 125}
                    for _ in range(config.total_updates)]
            metrics = {"device": {"device": "cuda", "cuda_available": True, "cuda_name": "NVIDIA T4",
                                  "optimizer_updates": 4 * config.total_updates,
                                  "cuda_optimizer_updates": 4 * config.total_updates},
                       "training": {"updates": logs, "mean_rollout_seconds": .2,
                                    "update_elapsed_seconds": .4 * config.total_updates,
                                    "elapsed_seconds": .5 * config.total_updates},
                       "strategy_outcomes": {"wins": 3, "losses": 1},
                       "initial_checkpoint": "checkpoints/step-0.pt"}
            (output / "training-metrics.json").write_text(json.dumps(metrics))
            return metrics

        with tempfile.TemporaryDirectory() as tmp, patch("training.ppo.g008_benchmark.train", side_effect=fake_train):
            report = run_principal_benchmark(Path(tmp) / "out", run_ids=ids)
        self.assertEqual(len(calls), WARMUP_REPEATS + MEASURED_REPEATS)
        self.assertEqual([count for _, count, _ in calls], [1, 1, 3, 3, 3, 3, 3])
        self.assertTrue(all(call[2].get("g008_schedule") for call in calls))
        self.assertEqual([call[2].get("g008_backend_role") for call in calls],
                         [None, None, "incumbent", "incumbent", "incumbent", "incumbent", "incumbent"])
        self.assertTrue(all("save_checkpoints" not in call[2] for call in calls))
        out = Path(tmp) / "out"
        self.assertFalse(any(out.rglob("*.pt")))
        self.assertFalse(any(out.rglob("*.onnx")))
        for metrics_path in out.rglob("training-metrics.json"):
            saved = json.loads(metrics_path.read_text())
            self.assertEqual(saved["format"], "circuit-ai-g008-benchmark-run-metrics/1")
            self.assertEqual(set(saved), {"format", "device", "training"})
            self.assertEqual(set(saved["device"]), {"device", "cuda_available", "cuda_name",
                                                       "optimizer_updates", "cuda_optimizer_updates"})
            self.assertIn("update_elapsed_seconds", saved["training"])
            self.assertTrue(saved["training"]["updates"])
            forbidden = {"mean_loss", "episodes_completed", "snapshot_opponent_arenas",
                         "latest_opponent_arenas", "strategy_outcomes", "wins", "losses",
                         "initial_checkpoint"}
            def keys(value):
                if isinstance(value, dict):
                    return set(value).union(*(keys(item) for item in value.values()))
                if isinstance(value, list):
                    return set().union(*(keys(item) for item in value))
                return set()
            self.assertFalse(keys(saved) & forbidden)
        report_keys = set(report)
        self.assertIn("summary", report_keys)
        self.assertIn("update_only_joint_transitions_per_second",
                      report["summary"])
        self.assertNotIn("strategy_outcomes", report)
        self.assertEqual(report["warmups"]["count"], 2)
        self.assertEqual(len(report["repeats"]), 5)
        repeat = report["repeats"][0]
        expected = 128 * 32 * 3
        self.assertEqual(repeat["completed_joint_arena_transitions"], expected)
        self.assertEqual(repeat["agent_decisions"], expected * 2)
        self.assertEqual(repeat["cuda_optimizer_updates"], 12)
        self.assertAlmostEqual(repeat["update_only_seconds"], 1.05)
        self.assertAlmostEqual(repeat["checkpoint_inclusive_update_elapsed_seconds"], 1.2)
        self.assertAlmostEqual(repeat["checkpoint_seconds"], .15)
        self.assertAlmostEqual(report["summary"]["update_only_joint_transitions_per_second"]["median"], expected / 1.05)
        self.assertIn("No speedup", report["baseline_note"])
        self.assertEqual(report["device"]["cuda_name"], "NVIDIA T4")
        self.assertEqual(len({item["schedule_digest"] for item in report["repeats"]}), 5)
        self.assertFalse(any("outcome" in key or "win" in key or "checkpoint" in key
                             for key in report))

    def test_rejects_incomplete_cuda_evidence_and_nonempty_output(self):
        def cpu_train(config, output, **kwargs):
            return {"device": {"device": "cpu", "cuda_available": False}}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with patch("training.ppo.g008_benchmark.train", side_effect=cpu_train):
                with self.assertRaisesRegex(RuntimeError, "did not prove CUDA"):
                    run_principal_benchmark(out, run_ids=run_ids())
            (out / "sentinel").write_text("x")
            with self.assertRaises(FileExistsError):
                run_principal_benchmark(out, run_ids=run_ids())

    def test_frozen_measured_and_warmup_schedules_cover_every_root_reset_and_backend(self):
        ids = run_ids()
        schedules = prepare_benchmark_schedules(ids)
        # Canonical run-config JSON sorts map keys; schedule semantics must be
        # invariant to object insertion order after serialization.
        self.assertEqual(prepare_benchmark_schedules(dict(sorted(ids.items()))), schedules)
        measured = schedules["pair-00"]
        slots_by_coordinate = {(slot["update_index"], slot["arena_index"], slot["game_index"],
                                slot["reset_index"]): slot for slot in measured["slots"]}
        adapter = G008EnvironmentSeedSchedule(measured, run_id=ids["pair-00"], arena_count=128,
                                              total_updates=3, rollout_steps=32,
                                              benchmark_role="incumbent")
        self.assertEqual(len(adapter._slots), 128 * (1 + 3 * 32))
        for arena in range(128):
            initial = slots_by_coordinate[(0, arena, 0, 0)]
            # Collision pair has one shared tuple, but the backend role resolves exactly one slot.
            expected_initial = derive_environment_seed({key: initial[key] for key in (
                "split", "domain_label", "env_root_seed", "run_key", "update_index", "rollout_index",
                "arena_index", "game_index", "reset_index")})[1]
            self.assertEqual(adapter.benchmark_initial_seed(arena), expected_initial)
            self.assertEqual(initial["env_root_seed"], 5_000_000)
            for update in range(3):
                for ordinal in range(1, 33):
                    game_index = update * 32 + ordinal
                    slot = slots_by_coordinate[(update, arena, game_index, ordinal)]
                    self.assertEqual(adapter.benchmark_reset_seed(update=update, arena=arena,
                                                                  reset_ordinal=ordinal),
                                     slot["derived_seed_uint64"])
                    self.assertEqual(slot["env_root_seed"], 5_000_000)
        warm_key = "warmup-python-oracle-incumbent-00"
        warm = schedules[warm_key]
        warm_adapter = G008EnvironmentSeedSchedule(warm, run_id=ids[warm_key], arena_count=128,
                                                   total_updates=1, rollout_steps=32)
        self.assertEqual(len(warm_adapter._slots), 128 * 33)
        self.assertEqual(warm_adapter.run_key, warm_key)
        self.assertEqual(next(slot["env_root_seed"] for slot in warm["slots"]), 5_000_256)

    def test_schedule_preflight_rejects_wrong_backend_missing_slots_and_unknown_run_keys(self):
        from training.eval.provenance import ProvenanceError

        ids = run_ids()
        schedules = prepare_benchmark_schedules(ids)
        with self.assertRaisesRegex(ProvenanceError, "explicit incumbent/candidate"):
            G008EnvironmentSeedSchedule(schedules["pair-00"], run_id=ids["pair-00"],
                                        arena_count=128, total_updates=3, rollout_steps=32)
        with self.assertRaisesRegex(ProvenanceError, "frozen measured/warmup"):
            prepare_benchmark_schedules({key: value for key, value in ids.items() if key != "pair-04"})
        with self.assertRaisesRegex(ProvenanceError, "frozen measured/warmup"):
            prepare_benchmark_schedules({**ids, "pair-99": str(uuid.uuid4())})
        missing = dict(ids)
        missing["pair-00"] = "12345678-1234-4234-8234-123456789abc"
        schedule = schedules["pair-00"]
        # The provided run-ID mapping cannot be substituted after expansion.
        with self.assertRaisesRegex(ProvenanceError, "run_id"):
            G008EnvironmentSeedSchedule(schedule, run_id=missing["pair-00"], arena_count=128,
                                        total_updates=3, rollout_steps=32, benchmark_role="incumbent")
        incomplete = copy.deepcopy(schedule)
        incomplete["slots"].pop()
        with self.assertRaises(ProvenanceError):
            G008EnvironmentSeedSchedule(incomplete, run_id=ids["pair-00"], arena_count=128,
                                        total_updates=3, rollout_steps=32, benchmark_role="incumbent")


if __name__ == "__main__":
    unittest.main()
