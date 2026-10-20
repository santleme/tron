import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from training.ppo.benchmark import (MEASURED_REPEATS, build_candidates, make_benchmark_report,
                                    run_benchmark, select_throughput_winner, summarize_repeats)
from training.ppo.train import TrainConfig


class BenchmarkPlanningTests(unittest.TestCase):
    def test_candidates_are_bounded_and_hold_device_updates_constant(self):
        base = TrainConfig(device="cpu", total_updates=2, width=8, height=8)
        configs = build_candidates([
            {"arena_count": 8, "rollout_steps": 32, "minibatch_size": 64},
            {"arena_count": 16, "rollout_steps": 32, "minibatch_size": 128},
        ], base)
        self.assertEqual(len(configs), 2)
        self.assertEqual({c.device for c in configs}, {"cpu"})
        self.assertEqual({c.total_updates for c in configs}, {2})
        self.assertEqual(configs[0].arena_count, 8)
        with self.assertRaisesRegex(ValueError, "at most 5"):
            build_candidates([{}] * 6, base)
        with self.assertRaisesRegex(ValueError, "unsupported overrides"):
            build_candidates([{"device": "cuda"}], base)

    def test_selection_uses_throughput_then_update_latency_and_reports_cuda_evidence(self):
        trials = [
            {"index": 0, "trial_id": "trial-00", "median_environment_steps_per_second": 80.0, "median_wall_environment_steps_per_second": 72.0,
             "median_update_latency_seconds": 0.12, "median_rollout_seconds": 0.08,
             "median_optimizer_seconds": 0.04, "median_rollout_to_optimizer_ratio": 2.0,
             "optimizer_updates": 36, "cuda_optimizer_updates": 0},
            {"index": 1, "trial_id": "trial-01", "median_environment_steps_per_second": 95.0, "median_wall_environment_steps_per_second": 90.0,
             "median_update_latency_seconds": 0.15, "median_rollout_seconds": 0.05,
             "median_optimizer_seconds": 0.10, "median_rollout_to_optimizer_ratio": 0.5,
             "optimizer_updates": 36, "cuda_optimizer_updates": 36},
            {"index": 2, "trial_id": "trial-02", "median_environment_steps_per_second": 95.0, "median_wall_environment_steps_per_second": 91.0,
             "median_update_latency_seconds": 0.10, "median_rollout_seconds": 0.02,
             "median_optimizer_seconds": 0.08, "median_rollout_to_optimizer_ratio": 0.25,
             "optimizer_updates": 36, "cuda_optimizer_updates": 36},
        ]
        self.assertEqual(select_throughput_winner(trials)["index"], 2)
        report = make_benchmark_report(trials, "cuda")
        self.assertEqual(report["selected_trial"], 2)
        self.assertIn("median update-only environment steps/s", report["selection_rationale"])
        self.assertIn("Checkpoint I/O is excluded", report["selection_rationale"])
        self.assertIn("36 CUDA optimizer updates out of 36", report["selection_rationale"])
        self.assertIn("single device", report["selection_rationale"])
        self.assertEqual(report["warmup"], {"updates_per_candidate": 1, "discarded": True})
        self.assertEqual(report["measured_repeats_per_candidate"], 3)
        self.assertNotIn("occupancy", report["selection_metric"])

    def test_bad_empty_and_incomplete_trial_results_fail(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            select_throughput_winner([])
        with self.assertRaisesRegex(ValueError, "missing metric"):
            select_throughput_winner([{"environment_steps_per_second": 2}])

    def test_repeated_sample_medians_not_best_or_mean_drive_selection(self):
        repeats = [
            {"repeat": 0, "environment_steps_per_second": 10.0, "wall_environment_steps_per_second": 8.0, "mean_update_latency_seconds": 0.5,
             "rollout_seconds": 0.2, "optimizer_seconds": 0.3,
             "optimizer_updates": 2, "cuda_optimizer_updates": 2},
            {"repeat": 1, "environment_steps_per_second": 100.0, "wall_environment_steps_per_second": 50.0, "mean_update_latency_seconds": 0.1,
             "rollout_seconds": 0.02, "optimizer_seconds": 0.08,
             "optimizer_updates": 2, "cuda_optimizer_updates": 2},
            {"repeat": 2, "environment_steps_per_second": 20.0, "wall_environment_steps_per_second": 15.0, "mean_update_latency_seconds": 0.3,
             "rollout_seconds": 0.15, "optimizer_seconds": 0.15,
             "optimizer_updates": 2, "cuda_optimizer_updates": 2},
        ]
        summary = summarize_repeats(0, "trial-00", TrainConfig(), {"device": "cpu"}, repeats)
        self.assertEqual(summary["median_environment_steps_per_second"], 20.0)
        self.assertEqual(summary["median_wall_environment_steps_per_second"], 15.0)
        self.assertEqual(summary["median_update_latency_seconds"], 0.3)
        self.assertEqual(summary["median_rollout_to_optimizer_ratio"], 1.0)
        self.assertEqual(summary["optimizer_updates"], 6)
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            summarize_repeats(0, "trial-00", TrainConfig(), {}, repeats[:2])

    def test_summary_reports_rollout_and_optimizer_medians_separately(self):
        repeats = [
            {"repeat": 0, "environment_steps_per_second": 10.0, "wall_environment_steps_per_second": 8.0, "mean_update_latency_seconds": 0.6,
             "rollout_seconds": 0.2, "optimizer_seconds": 0.4,
             "optimizer_updates": 2, "cuda_optimizer_updates": 0},
            {"repeat": 1, "environment_steps_per_second": 20.0, "wall_environment_steps_per_second": 12.0, "mean_update_latency_seconds": 0.6,
             "rollout_seconds": 0.3, "optimizer_seconds": 0.3,
             "optimizer_updates": 2, "cuda_optimizer_updates": 0},
            {"repeat": 2, "environment_steps_per_second": 30.0, "wall_environment_steps_per_second": 20.0, "mean_update_latency_seconds": 0.7,
             "rollout_seconds": 0.1, "optimizer_seconds": 0.6,
             "optimizer_updates": 2, "cuda_optimizer_updates": 0},
        ]
        summary = summarize_repeats(0, "trial-00", TrainConfig(), {"device": "cpu"}, repeats)
        self.assertEqual(summary["median_rollout_seconds"], 0.2)
        self.assertEqual(summary["median_optimizer_seconds"], 0.4)
        self.assertEqual(summary["median_wall_environment_steps_per_second"], 12.0)
        self.assertEqual(summary["median_rollout_to_optimizer_ratio"], 0.5)

    def test_runner_invokes_one_discarded_warmup_and_three_measured_jobs_per_candidate(self):
        config = TrainConfig(device="cpu", total_updates=2, width=8, height=8,
                             arena_count=1, rollout_steps=2, minibatch_size=4)
        calls = []

        def fake_train(candidate, output, export_onnx, run_evaluation):
            output = Path(output)
            calls.append((output, candidate.total_updates, export_onnx, run_evaluation))
            output.mkdir(parents=True)
            if output.name == "warmup":
                speed, latency = 999.0, 0.001
            else:
                speed = {"repeat-00": 10.0, "repeat-01": 20.0, "repeat-02": 30.0}[output.name]
                latency = 1 / speed
            return {"device": {"device": "cpu", "cuda_name": None,
                               "optimizer_updates": 2, "cuda_optimizer_updates": 0},
                    "training": {"environment_steps_per_second": speed,
                                 "wall_environment_steps_per_second": speed / 2,
                                 "mean_update_latency_seconds": latency,
                                 "mean_rollout_seconds": 0.4,
                                 "mean_optimizer_seconds": 0.6}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "benchmark"
            with patch("training.ppo.benchmark.train", side_effect=fake_train):
                report = run_benchmark([config], root)
            self.assertEqual(len(calls), 1 + MEASURED_REPEATS)
            self.assertEqual(calls[0][1], 1)
            self.assertEqual([call[1] for call in calls[1:]], [2, 2, 2])
            self.assertTrue(all(call[2:] == (False, False) for call in calls))
            self.assertEqual(report["trials"][0]["median_environment_steps_per_second"], 20.0)
            self.assertEqual(report["trials"][0]["median_wall_environment_steps_per_second"], 10.0)
            saved = json.loads((root / "benchmark-results.json").read_text())
            self.assertEqual(saved["selected_trial"], 0)
            with self.assertRaises(FileExistsError):
                run_benchmark([config], root)

    def test_runner_rejects_nonempty_output_directory_before_training(self):
        config = TrainConfig(device="cpu", total_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "used"
            root.mkdir()
            (root / "existing.txt").write_text("keep")
            with patch("training.ppo.benchmark.train") as fake_train:
                with self.assertRaises(FileExistsError):
                    run_benchmark([config], root)
                fake_train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
