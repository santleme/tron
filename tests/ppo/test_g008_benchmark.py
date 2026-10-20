import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from training.ppo import g008_benchmark
from training.ppo.g008_benchmark import (CONFIG_PATH, MEASURED_REPEATS, WARMUP_REPEATS,
                                         load_principal_config, prepare_benchmark_schedules,
                                         run_principal_benchmark)
from training.eval.provenance import benchmark_run_keys, derive_environment_seed
from training.ppo.protocol import file_sha256
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

    def test_selector_binds_exact_snapshot_hashes_and_fails_on_drift(self):
        baseline = g008_benchmark.implementation_hashes("incumbent")
        self.assertEqual(baseline, {"trainer_sha256": "419db34631b461193a9942bddc9b65841ed4a984c5acd9b1bfaade33a75f55d9",
                                    "engine_sha256": "a93dc0c8001325dbb55bb90da5cd9ae4e262948de9b27be01a195299fd8be25b"})
        incumbent_train, identity = g008_benchmark.select_implementation("incumbent")
        self.assertEqual(identity["role"], "incumbent")
        self.assertEqual(identity["variant_id"], "incumbent")
        self.assertTrue(callable(incumbent_train))
        with patch.object(g008_benchmark, "INCUMBENT_TRAIN_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "frozen source identity"):
                g008_benchmark.implementation_hashes("incumbent")
        candidate = g008_benchmark.implementation_hashes("candidate")
        self.assertEqual(candidate, {"trainer_sha256": "9991dc8e0c45cf586d2597f82f1cc30ceaf62c6754b976bd39eca70a3f86c38c",
                                     "engine_sha256": "cfd6fe1817e06e5e892306f1fac2b5eef3cdeb4e68da2e83b26c8f4f23a6c7d0"})
        self.assertEqual(g008_benchmark.select_implementation("candidate")[1]["role"], "candidate")

    def test_incumbent_engine_resolves_seed_import_without_snapshot_drift(self):
        hashes_before = g008_benchmark.implementation_hashes("incumbent")
        incumbent = g008_benchmark._load_incumbent_train_module()

        state, seeds = incumbent.initialize_episode(24, 24, 1729, 0)

        self.assertEqual((state.width, state.height), (24, 24))
        self.assertEqual(len(state.agents), 2)
        self.assertEqual([seed.label for seed in seeds], ["episode", "agent/0", "agent/1"])
        self.assertEqual(g008_benchmark.implementation_hashes("incumbent"), hashes_before)

    def test_parity_comparison_detects_collector_mismatch(self):
        config, _ = load_principal_config()
        traces = [{"sha256": "1" * 64, "transition_count": 4096, "reset_count": 12},
                  {"sha256": "2" * 64, "transition_count": 4096, "reset_count": 12}]
        with patch("training.ppo.g008_benchmark._transition_parity_digest", side_effect=traces):
            parity = g008_benchmark.compute_transition_parity(config, {}, str(uuid.uuid4()))
        self.assertFalse(parity["passed"])
        self.assertEqual(parity["incumbent"]["sha256"], "1" * 64)
        self.assertEqual(parity["candidate"]["sha256"], "2" * 64)

    def test_selection_requires_parity_median_and_four_pairs(self):
        def report(ratios, passed=True):
            return {"parity": {"passed": passed},
                    "pairs": [{"candidate_speedup_ratio": value} for value in ratios]}

        self.assertEqual(g008_benchmark.select_winning_implementation(report([1.2] * 5)), "candidate")
        self.assertEqual(g008_benchmark.select_winning_implementation(report([1.2] * 3 + [1.0] * 2)),
                         "incumbent")
        self.assertEqual(g008_benchmark.select_winning_implementation(report([1.2] * 5, passed=False)),
                         "incumbent")
        with self.assertRaisesRegex(ValueError, "positive numeric"):
            g008_benchmark.select_winning_implementation(report([float("inf")] * 5))

    def test_runs_role_warmups_then_five_alternating_matched_pairs(self):
        calls = []
        ids = run_ids()
        metrics_hashes = {}
        sanitize_outputs = g008_benchmark._remove_checkpoint_outputs
        implementation_id = {"trainer_sha256": "a" * 64, "engine_sha256": "b" * 64}

        def sanitize_and_persist_metrics(run_dir, metrics):
            sanitize_outputs(run_dir, metrics)
            metrics_path = Path(run_dir) / "training-metrics.json"
            if not metrics_path.exists():
                metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            metrics_hashes[str(Path(run_dir).parent.name) + ":" + Path(run_dir).name] = file_sha256(metrics_path)

        def fake_train(config, output, **kwargs):
            output = Path(output)
            calls.append((output, config.total_updates, kwargs))
            output.mkdir(parents=True, exist_ok=True)
            (output / "checkpoints").mkdir()
            (output / "checkpoints" / "step-0.pt").write_bytes(b"ephemeral")
            role = kwargs.get("g008_backend_role") or output.parts[-2]
            duration = .35 if role == "incumbent" else .30
            logs = [{"optimizer_updates": 4, "cuda_updates": 4, "device": "cuda", "optimizer_seconds": .1,
                     "update_latency_seconds": duration, "checkpoint_seconds": .05, "rollout_seconds": .2,
                     "mean_loss": 9.0, "episodes_completed": 17,
                     "snapshot_opponent_arenas": 3, "latest_opponent_arenas": 125}
                    for _ in range(config.total_updates)]
            metrics = {"device": {"device": "cuda", "cuda_available": True, "cuda_name": "NVIDIA T4",
                                  "optimizer_updates": 4 * config.total_updates,
                                  "cuda_optimizer_updates": 4 * config.total_updates},
                       "training": {"updates": logs, "mean_rollout_seconds": .2,
                                    "update_elapsed_seconds": duration * config.total_updates,
                                    "elapsed_seconds": (duration + .1) * config.total_updates},
                       "strategy_outcomes": {"wins": 3, "losses": 1},
                       "initial_checkpoint": "checkpoints/step-0.pt"}
            (output / "training-metrics.json").write_text(json.dumps(metrics))
            return metrics

        def fake_selector(role):
            return fake_train, {"role": role, **implementation_id}

        parity = {"passed": True, "incumbent": {"sha256": "c" * 64},
                  "candidate": {"sha256": "c" * 64},
                  "implementation_hashes": {"incumbent": implementation_id,
                                            "candidate": implementation_id}}
        with tempfile.TemporaryDirectory() as tmp, \
                patch("training.ppo.g008_benchmark.select_implementation", side_effect=fake_selector), \
                patch("training.ppo.g008_benchmark._reset_arm_rngs"), \
                patch("training.ppo.g008_benchmark.compute_transition_parity", return_value=parity), \
                patch("training.ppo.g008_benchmark._assert_paired_environment_seeds"), \
                patch("training.ppo.g008_benchmark._remove_checkpoint_outputs",
                      side_effect=sanitize_and_persist_metrics):
            report = run_principal_benchmark(Path(tmp) / "out", run_ids=ids)
        self.assertEqual(len(calls), WARMUP_REPEATS * 8 + MEASURED_REPEATS * 2)
        self.assertEqual([count for _, count, _ in calls], [1] * 16 + [3] * 10)
        self.assertTrue(all(call[2].get("g008_schedule") for call in calls))
        self.assertEqual([call[2].get("g008_backend_role") for call in calls[:16]], [None] * 16)
        self.assertEqual(sum(call[2].get("g008_backend_role") == "incumbent" for call in calls[16:]), 5)
        self.assertEqual(sum(call[2].get("g008_backend_role") == "candidate" for call in calls[16:]), 5)
        out = Path(tmp) / "out"
        self.assertFalse(any(out.rglob("*.pt")))
        self.assertFalse(any(out.rglob("*.onnx")))
        for metrics_path in out.rglob("training-metrics.json"):
            saved = json.loads(metrics_path.read_text())
            self.assertEqual(set(saved), {"format", "device", "training"})
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
        from training.eval.provenance import schedule_spec_digest, seed_plan_sha256
        keys = benchmark_run_keys()
        self.assertEqual(report["format"], "circuit-ai-g008-principal-benchmark/3")
        self.assertEqual(report["seed_plan_sha256"], seed_plan_sha256())
        self.assertEqual(report["schedule_spec_digest"], schedule_spec_digest())
        schedules = prepare_benchmark_schedules(ids)
        self.assertEqual(report["schedule_roster"], [
            {"run_key": key, "run_id": ids[key],
             "expanded_schedule_digest": schedules[key]["expanded_schedule_digest"]}
            for key in keys])
        self.assertEqual([item["run_key"] for item in report["pairs"]],
                         [f"pair-{index:02d}" for index in range(MEASURED_REPEATS)])
        self.assertEqual([item["run_id"] for item in report["pairs"]],
                         [ids[f"pair-{index:02d}"] for index in range(MEASURED_REPEATS)])
        expected_orders = [["incumbent", "candidate"] if index % 2 == 0
                           else ["candidate", "incumbent"] for index in range(MEASURED_REPEATS)]
        self.assertEqual([item["order"] for item in report["pairs"]], expected_orders)
        for pair in report["pairs"]:
            self.assertEqual(pair["arms"]["incumbent"]["run_id"], pair["arms"]["candidate"]["run_id"])
            self.assertEqual(pair["arms"]["incumbent"]["seed"], pair["arms"]["candidate"]["seed"])
            self.assertEqual(pair["arms"]["incumbent"]["expanded_schedule_digest"],
                             pair["arms"]["candidate"]["expanded_schedule_digest"])
            self.assertEqual(pair["arms"]["incumbent"]["run_key"], pair["arms"]["candidate"]["run_key"])
            self.assertEqual(pair["arms"]["incumbent"]["config_sha256"], pair["arms"]["candidate"]["config_sha256"])
            self.assertEqual(pair["arms"]["incumbent"]["seed"], report["config"]["seed"])
            self.assertEqual(pair["arms"]["incumbent"]["implementation"]["role"], "incumbent")
            self.assertEqual(pair["arms"]["candidate"]["implementation"]["role"], "candidate")
        self.assertEqual([item["run_key"] for item in report["warmups"]["runs"]],
                         [f"warmup-{backend}-{role}-{index:02d}"
                          for backend in ("python-oracle", "low-risk", "numpy-control", "numba-candidate")
                          for role in ("incumbent", "candidate") for index in range(WARMUP_REPEATS)])
        self.assertEqual(len(report["warmups"]["runs"]), 16)
        self.assertEqual(report["warmups"]["count_per_role"], 2)
        self.assertEqual(len(report["pairs"]), 5)
        self.assertEqual(report["selection"], "candidate")
        self.assertTrue(report["parity"]["passed"])
        expected = 128 * 32 * 3
        repeat = report["pairs"][0]["arms"]["incumbent"]
        self.assertEqual(repeat["completed_joint_arena_transitions"], expected)
        self.assertEqual(repeat["agent_decisions"], expected * 2)
        self.assertEqual(repeat["cuda_optimizer_updates"], 12)
        self.assertAlmostEqual(repeat["update_only_seconds"], .9)
        self.assertAlmostEqual(repeat["checkpoint_inclusive_update_elapsed_seconds"], 1.05)
        self.assertAlmostEqual(repeat["checkpoint_seconds"], .15)
        self.assertAlmostEqual(report["summary"]["incumbent_update_only_joint_transitions_per_second"]["median"], expected / .9)
        self.assertEqual(report["device"]["cuda_name"], "NVIDIA T4")
        self.assertEqual(len({item["expanded_schedule_digest"] for item in report["pairs"]}), 5)
        self.assertFalse(any("outcome" in key or "win" in key or "checkpoint" in key
                             for key in report))

    def test_seed_plan_version_and_hash_drift_fail_before_training(self):
        from training.eval import provenance

        ids = run_ids()
        # Keep the schedule cache hot across each modified-path attempt. The
        # preflight must bind to current seed-plan bytes despite cached rosters.
        prepare_benchmark_schedules(ids)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed-plan.json"
            doc = json.loads(provenance.SEED_PLAN_PATH.read_text(encoding="utf-8"))
            cases = []
            wrong_version = copy.deepcopy(doc)
            wrong_version["format"] = "circuit-ai-g008-split-seed-plan/2"
            cases.append(wrong_version)
            cases.append(doc)  # Same valid JSON/schema, but changed artifact bytes below.
            for index, candidate in enumerate(cases):
                payload = json.dumps(candidate, indent=2, ensure_ascii=False)
                if index == 1:
                    payload += "\n\n"
                path.write_text(payload, encoding="utf-8")
                with patch("training.eval.provenance.SEED_PLAN_PATH", path), \
                        patch("training.ppo.g008_benchmark.select_implementation") as train_mock:
                    train_mock.side_effect = AssertionError(
                        "benchmark reached training before rejecting seed-plan drift")
                    with self.assertRaises(provenance.ProvenanceError):
                        provenance.benchmark_run_keys()
                    with self.assertRaises(provenance.ProvenanceError):
                        run_principal_benchmark(Path(tmp) / f"out-{index}", run_ids=ids)
                    train_mock.assert_not_called()

    def test_missing_or_stale_schedule_binding_fails_before_training(self):
        from training.eval import provenance

        ids = run_ids()
        keys = benchmark_run_keys()
        schedules = prepare_benchmark_schedules(ids)
        with tempfile.TemporaryDirectory() as tmp:
            seed_path = Path(tmp) / "seed-plan.json"
            seed_path.write_bytes(provenance.SEED_PLAN_PATH.read_bytes())
            cases = (("missing", None), ("stale", "0" * 64))
            for label, value in cases:
                damaged = copy.deepcopy(schedules)
                if label == "missing":
                    damaged[keys[0]].pop("schedule_spec_digest")
                else:
                    damaged[keys[0]]["schedule_spec_digest"] = value
                with patch("training.eval.provenance.SEED_PLAN_PATH", seed_path), \
                        patch("training.ppo.g008_benchmark._cached_benchmark_schedules",
                              return_value=damaged), \
                        patch("training.ppo.g008_benchmark.select_implementation") as train_mock:
                    with self.assertRaises(ValueError):
                        run_principal_benchmark(Path(tmp) / f"out-{label}", run_ids=ids)
                    train_mock.assert_not_called()

    def test_rejects_incomplete_cuda_evidence_and_nonempty_output(self):
        def cpu_train(config, output, **kwargs):
            return {"device": {"device": "cpu", "cuda_available": False}}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            fake_hashes = {"trainer_sha256": "a" * 64, "engine_sha256": "b" * 64}
            with patch("training.ppo.g008_benchmark.compute_transition_parity",
                       return_value={"passed": True, "incumbent": {"sha256": "c" * 64},
                                     "candidate": {"sha256": "c" * 64},
                                     "implementation_hashes": {"incumbent": fake_hashes,
                                                               "candidate": fake_hashes}}), \
                    patch("training.ppo.g008_benchmark._assert_paired_environment_seeds"), \
                    patch("training.ppo.g008_benchmark._reset_arm_rngs"), \
                    patch("training.ppo.g008_benchmark.select_implementation",
                          return_value=(cpu_train, {"role": "incumbent", **fake_hashes})):
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
