import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from training.ppo.protocol import canonical_json, digest_json, safe_relative_path
from training.ppo.train import (TrainConfig, make_checkpoint_metadata, make_run_manifest,
                                reset_arena_index, sample_snapshot_opponents, terminal_rewards,
                                updates_remaining)


class ArtifactProtocolTests(unittest.TestCase):
    def test_digest_excludes_own_field_and_is_canonical(self):
        doc = {"z": [1, 2], "a": "é", "digest": "ignored"}
        self.assertEqual(canonical_json(doc, "digest"), b'{"a":"\xc3\xa9","z":[1,2]}')
        self.assertEqual(digest_json(doc, "digest"), hashlib.sha256(b'{"a":"\xc3\xa9","z":[1,2]}').hexdigest())

    def test_non_nfc_nonfinite_and_unsafe_path_fail_closed(self):
        with self.assertRaises(ValueError):
            canonical_json({"value": "e\u0301"})
        with self.assertRaises(ValueError):
            canonical_json({"value": float("nan")})
        for path in ("/tmp/model.onnx", "../model.onnx", "a/../model.onnx", "a\\model.onnx"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_relative_path(path)
        self.assertEqual(safe_relative_path("checkpoints/model.onnx"), "checkpoints/model.onnx")

    def test_number_spelling_matches_ecmascript_json_stringify_boundaries(self):
        document = {"values": [0.0, -0.0, 1.0, 1e-6, 1e-7, 1e15, 1.2345e-7]}
        expected = b'{"values":[0,0,1,0.000001,1e-7,1000000000000000,1.2345e-7]}'
        self.assertEqual(canonical_json(document), expected)
        with self.assertRaisesRegex(ValueError, "safe range"):
            canonical_json({"value": 1e20})

    def test_float_bearing_evaluation_fixture_hashes_with_ecmascript_canonicalization(self):
        root = Path(__file__).resolve().parents[2]
        report = json.loads((root / "shared/fixtures/lc-eval-float-1.json").read_text())
        schema = json.loads((root / "schemas/lc-eval-1.schema.json").read_text())
        Draft202012Validator(schema).validate(report)
        self.assertTrue(any(g["summary"]["mean_survival_ticks"] % 1 for g in report["matchups"]))
        self.assertEqual(report["report_sha256"], digest_json(report, "report_sha256"))
        self.assertIn(b"0.40625", canonical_json(report))

    def test_tiny_cpu_configuration_is_validated_without_torch(self):
        TrainConfig(width=4, height=4, arena_count=1, rollout_steps=1, total_updates=1,
                    update_epochs=1, minibatch_size=2, hidden_size=8, device="cpu").validate()
        with self.assertRaises(ValueError):
            TrainConfig(width=1).validate()

    def test_terminal_rewards_match_zero_sum_outcomes_in_both_seats(self):
        self.assertEqual(terminal_rewards("draw"), (0.0, 0.0))
        self.assertEqual(terminal_rewards("agent_0_win"), (1.0, -1.0))
        self.assertEqual(terminal_rewards("agent_1_win"), (-1.0, 1.0))
        with self.assertRaises(ValueError):
            terminal_rewards("not-terminal")

    def test_reset_arena_indices_are_repeatable_and_unique_across_updates(self):
        indices = [reset_arena_index(update, arena, step, 3, 4)
                   for update in range(4) for step in range(5) for arena in range(3)]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(indices, [reset_arena_index(update, arena, step, 3, 4)
                                   for update in range(4) for step in range(5) for arena in range(3)])

    def test_resume_total_updates_is_global_target(self):
        self.assertEqual(updates_remaining(10, 0), 10)
        self.assertEqual(updates_remaining(10, 4), 6)
        with self.assertRaisesRegex(ValueError, "already reached target"):
            updates_remaining(10, 10)
        with self.assertRaisesRegex(ValueError, "already reached target"):
            updates_remaining(10, 12)

    def test_snapshot_opponent_sampling_is_opt_in_and_seeded(self):
        self.assertEqual(sample_snapshot_opponents(5, 0), [False] * 5)
        self.assertEqual(sample_snapshot_opponents(5, 1), [True] * 5)
        first = sample_snapshot_opponents(20, 0.35, random.Random(74))
        second = sample_snapshot_opponents(20, 0.35, random.Random(74))
        self.assertEqual(first, second)

    def test_checkpoint_metadata_builder_matches_v1_schema_and_hashes_files(self):
        root = Path(__file__).resolve().parents[2]
        schema = json.loads((root / "schemas/checkpoint-metadata-1.schema.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            pt = Path(directory) / "step-2.pt"
            onnx = Path(directory) / "step-2.onnx"
            pt.write_bytes(b"native checkpoint")
            onnx.write_bytes(b"onnx model")
            metadata = make_checkpoint_metadata("step-2", 2, TrainConfig(), onnx, pt)
            metadata["metadata_sha256"] = digest_json(metadata, "metadata_sha256")
            Draft202012Validator(schema).validate(metadata)

    def test_run_manifest_builder_matches_lc_run_schema_and_eval_contract(self):
        root = Path(__file__).resolve().parents[2]
        schema = json.loads((root / "schemas/lc-run-1.schema.json").read_text())
        fixture = json.loads((root / "shared/fixtures/lc-eval-1.valid.json").read_text())
        config = TrainConfig(snapshot_opponent_probability=0.25)
        checksum = "a" * 64
        checkpoint_refs = [{"checkpoint_id": "step-0", "step": 0, "file": f"checkpoints/step-0.{fmt}",
                            "sha256": checksum, "format": fmt,
                            "observation_schema": "lc-observation/1", "action_schema": "lc-action/1"}
                           for fmt in ("pytorch", "onnx")]
        manifest = make_run_manifest("test-run", "2026-10-19T00:00:00Z", config, 0, checkpoint_refs,
                                     fixture, {"python": "3.12", "torch": "unavailable", "cuda": "unavailable",
                                               "onnx_exporter": "torch.onnx.export", "onnx_opset": 17,
                                               "kaggle_cli": "not-used"}, {"device": "cpu"}, [],
                                     [{"path": ref["file"], "sha256": checksum} for ref in checkpoint_refs])
        manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
        Draft202012Validator(schema).validate(manifest)
        self.assertEqual(manifest["training_config"]["self_play"]["opponent_sampling"], "latest-step-0-mixture")
        self.assertEqual(manifest["training_config"]["self_play"]["snapshot_checkpoint_id"], "step-0")
        self.assertEqual(manifest["training_config"]["self_play"]["snapshot_opponent_probability"], 0.25)


if __name__ == "__main__":
    unittest.main()
