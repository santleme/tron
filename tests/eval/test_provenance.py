import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from training.eval.provenance import (
    OPEN_SPACE_CODE_SHA256,
    OPEN_SPACE_CONFIG_SHA256,
    ProvenanceError,
    manifest_sha256,
    seed_plan_sha256,
    schedule_spec_digest,
    build_expanded_schedule,
    validate_realization,
    validate_manifest,
)


def _hash(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _schedule_artifacts():
    run_id = "fixture-run"
    expanded = build_expanded_schedule(suite_id="fixture-suite", run_id=run_id,
                                       run_key="validation-suite-v1", slots=[])
    realized = validate_realization(expanded, realized_slots=[],
                                    realized_tuple_list=[], episode_outcomes=[])
    return {run_id: expanded}, {run_id: realized}


def _schedule_artifacts_kwargs():
    expanded, realized = _schedule_artifacts()
    return {"expanded_schedules": expanded, "realized_schedules": realized}


def _validate_manifest(manifest):
    expanded, realized = _schedule_artifacts()
    return validate_manifest(manifest, expanded_schedules=expanded,
                             realized_schedules=realized)


def _row(split, opponent_id, seed, game=0, usage=None):
    prefix = f"{split}:{opponent_id}"
    row = {
        "split": split,
        "opponent_id": opponent_id,
        "opponent_sha256": _hash(prefix + ":opponent"),
        "opponent_code_sha256": _hash(prefix + ":code"),
        "opponent_config_sha256": _hash(prefix + ":config"),
        "opponent_model_sha256": _hash(prefix + ":model"),
        "target_checkpoint_sha256": _hash("target-checkpoint"),
        "training_config_sha256": _hash("training-config"),
        "source_id": f"{opponent_id}-games",
        "source_data_sha256": _hash(prefix + ":data"),
        "seed": {"train": 2_000_000, "imitation_demo": 3_000_000, "benchmark": 5_000_000,
                 "validation": 1_000_000, "final": 4_000_000}[split] + seed,
        "arena_index": game,
        "game_index": game,
        "target_seat": game % 2,
        "usages": usage or ({"train": ["train"], "imitation_demo": ["imitation"],
                              "benchmark": ["benchmark"],
                              "validation": ["validation", "checkpoint_selection"],
                              "final": ["final_eval"]}[split]),
    }
    if opponent_id == "open-space":
        row["opponent_code_sha256"] = OPEN_SPACE_CODE_SHA256
        row["opponent_config_sha256"] = OPEN_SPACE_CONFIG_SHA256
    return row


def _valid_manifest():
    manifest = {
        "format": "lc-split-manifest/3",
        "frozen": True,
        "manifest_sha256": "0" * 64,
        "seed_plan_sha256": seed_plan_sha256(),
        "schedule_spec_digest": schedule_spec_digest(),
        "schedule_bindings": [{
            "run_id": "fixture-run",
            "seed_plan_sha256": seed_plan_sha256(),
            "schedule_spec_digest": schedule_spec_digest(),
            "expanded_schedule_digest": _schedule_artifacts()[0]["fixture-run"]["expanded_schedule_digest"],
            "realized_schedule_digest": _schedule_artifacts()[1]["fixture-run"]["realized_schedule_digest"],
        }],
        "rows": [
            _row("train", "safe-turn", 1),
            _row("imitation_demo", "imitable-turn", 1),
            _row("validation", "heldout-contour", 2),
            _row("final", "open-space", 3),
            _row("benchmark", "baseline-only", 1),
        ],
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def _reseal(manifest):
    manifest["manifest_sha256"] = manifest_sha256(manifest)


class SplitProvenanceTests(unittest.TestCase):
    def test_valid_frozen_disjoint_manifest_passes_schema_and_semantics(self):
        manifest = _valid_manifest()
        schema = json.loads((Path(__file__).resolve().parents[2] /
                             "schemas/lc-split-manifest-3.schema.json").read_text())
        Draft202012Validator(schema).validate(manifest)
        self.assertIsNone(_validate_manifest(manifest))

    def test_missing_provenance_fails_closed(self):
        manifest = _valid_manifest()
        del manifest["rows"][0]["source_data_sha256"]
        _reseal(manifest)
        with self.assertRaises(ProvenanceError):
            _validate_manifest(manifest)

    def test_open_space_leakage_and_non_final_usage_are_rejected(self):
        cases = []
        leaked_policy = _valid_manifest()
        leaked_policy["rows"][0] = _row("train", "open-space", 1, usage=["train"])
        cases.append(leaked_policy)
        leaked_data = _valid_manifest()
        leaked_data["rows"][0]["source_id"] = "open-space"
        cases.append(leaked_data)
        wrong_open_usage = _valid_manifest()
        wrong_open_usage["rows"][3]["usages"] = ["checkpoint_selection"]
        cases.append(wrong_open_usage)
        for manifest in cases:
            _reseal(manifest)
            with self.subTest(rows=manifest["rows"]):
                with self.assertRaises(ProvenanceError):
                    _validate_manifest(manifest)

    def test_open_space_source_identifier_is_detected_inside_normalized_id(self):
        manifest = _valid_manifest()
        manifest["rows"][1]["source_id"] = "validation.OPEN_space.labels"
        _reseal(manifest)
        with self.assertRaisesRegex(ProvenanceError, "open-space"):
            _validate_manifest(manifest)

    def test_open_space_fingerprints_cannot_be_hidden_by_renaming_opponent(self):
        for field, value in (("opponent_code_sha256", OPEN_SPACE_CODE_SHA256),
                             ("opponent_config_sha256", OPEN_SPACE_CONFIG_SHA256)):
            manifest = _valid_manifest()
            alias = manifest["rows"][0]
            alias["opponent_id"] = "renamed-heuristic"
            alias["source_id"] = "renamed-heuristic-data"
            alias[field] = value
            _reseal(manifest)
            with self.subTest(field=field), self.assertRaisesRegex(ProvenanceError, "open-space"):
                _validate_manifest(manifest)

    def test_final_open_space_identity_must_match_locked_fingerprints(self):
        manifest = _valid_manifest()
        manifest["rows"][3]["opponent_code_sha256"] = _hash("wrong-open-space-code")
        _reseal(manifest)
        with self.assertRaisesRegex(ProvenanceError, "exact open-space"):
            _validate_manifest(manifest)

        manifest = _valid_manifest()
        alias = _row("final", "renamed-open-space", 5, usage=["final_eval"])
        alias["opponent_code_sha256"] = OPEN_SPACE_CODE_SHA256
        manifest["rows"].append(alias)
        _reseal(manifest)
        with self.assertRaisesRegex(ProvenanceError, "exact final-evaluation identity"):
            _validate_manifest(manifest)

    def test_g008_loader_validates_before_exposing_rows(self):
        from training.eval.g008_manifest import load_manifest, load_selection_rows, rows_for_usage

        manifest = _valid_manifest()
        expanded, realized = _schedule_artifacts()
        schedule_kwargs = {"expanded_schedules": expanded, "realized_schedules": realized}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "schedule artifacts are required"):
                load_manifest(path)
            with self.assertRaisesRegex(ProvenanceError, "schedule artifacts are required"):
                rows_for_usage(manifest, "train")
            loaded = load_manifest(path, **schedule_kwargs)
            self.assertEqual(len(rows_for_usage(loaded, "final_eval", **schedule_kwargs)), 1)
            self.assertEqual(rows_for_usage(loaded, "final_eval", **schedule_kwargs)[0]["opponent_id"], "open-space")
            selection_rows = load_selection_rows(path, **schedule_kwargs)
            self.assertTrue(selection_rows)
            self.assertTrue(all(row["split"] == "validation" for row in selection_rows))
            self.assertFalse(any(row["split"] == "final" for row in selection_rows))

            manifest["rows"][0]["source_id"] = "tampered-after-freeze"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "manifest SHA-256"):
                load_selection_rows(path, **schedule_kwargs)

            manifest = _valid_manifest()
            with self.assertRaisesRegex(ProvenanceError, "manifest SHA-256"):
                rows_for_usage(manifest | {"manifest_sha256": "0" * 64}, "train", **schedule_kwargs)

            stale_schedule = _valid_manifest()
            stale_schedule["schedule_bindings"][0]["expanded_schedule_digest"] = _hash("forged-expanded")
            stale_schedule["schedule_bindings"][0]["realized_schedule_digest"] = _hash("forged-realized")
            _reseal(stale_schedule)
            path.write_text(json.dumps(stale_schedule), encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "expanded digest"):
                load_manifest(path, **schedule_kwargs)
            with self.assertRaisesRegex(ProvenanceError, "expanded digest"):
                rows_for_usage(stale_schedule, "train", **schedule_kwargs)

    def test_manifest_is_bound_to_exact_seed_plan_artifact(self):
        from unittest.mock import patch

        manifest = _valid_manifest()
        mismatch = copy.deepcopy(manifest)
        mismatch["seed_plan_sha256"] = "0" * 64
        _reseal(mismatch)
        with self.assertRaisesRegex(ProvenanceError, "seed_plan_sha256"):
            _validate_manifest(mismatch)

        with tempfile.TemporaryDirectory() as directory:
            altered_plan = Path(directory) / "seed-plan.json"
            source_plan = Path(__file__).resolve().parents[2] / "training/configs/g008-split-seed-plan.json"
            altered_plan.write_bytes(source_plan.read_bytes() + b" ")
            with patch("training.eval.provenance.SEED_PLAN_PATH", altered_plan):
                with self.assertRaisesRegex(ProvenanceError, "seed.plan /3 bytes"):
                    _validate_manifest(manifest)

    def test_opponent_identity_or_hash_cannot_overlap_splits(self):
        id_overlap = _valid_manifest()
        row = copy.deepcopy(id_overlap["rows"][1])
        row["split"] = "train"
        row["usages"] = ["train"]
        row["seed"] = 4
        id_overlap["rows"].append(row)
        _reseal(id_overlap)
        with self.assertRaisesRegex(ProvenanceError, "overlaps"):
            _validate_manifest(id_overlap)

        hash_overlap = _valid_manifest()
        hash_overlap["rows"][1]["opponent_code_sha256"] = hash_overlap["rows"][0]["opponent_code_sha256"]
        _reseal(hash_overlap)
        with self.assertRaisesRegex(ProvenanceError, "overlaps"):
            _validate_manifest(hash_overlap)

    def test_opponent_id_must_map_to_one_immutable_hash_identity(self):
        manifest = _valid_manifest()
        second = copy.deepcopy(manifest["rows"][0])
        second["seed"] = 5
        second["game_index"] = 1
        second["arena_index"] = 1
        second["opponent_config_sha256"] = _hash("mutated-safe-turn-config")
        manifest["rows"].append(second)
        _reseal(manifest)
        with self.assertRaisesRegex(ProvenanceError, "inconsistent hashes"):
            _validate_manifest(manifest)

    def test_each_split_uses_its_frozen_seed_range_and_imitation_is_explicit(self):
        manifest = _valid_manifest()
        manifest["rows"][1]["seed"] = manifest["rows"][0]["seed"]
        _reseal(manifest)
        with self.assertRaisesRegex(ProvenanceError, "outside frozen imitation_demo range"):
            _validate_manifest(manifest)

    def test_source_ids_and_data_hashes_must_be_partition_disjoint(self):
        source_id_overlap = _valid_manifest()
        source_id_overlap["rows"].append(_row("final", "other-final", 4))
        source_id_overlap["rows"][1]["source_id"] = source_id_overlap["rows"][4]["source_id"]
        _reseal(source_id_overlap)
        with self.assertRaisesRegex(ProvenanceError, "source ID overlaps"):
            _validate_manifest(source_id_overlap)

        source_hash_overlap = _valid_manifest()
        source_hash_overlap["rows"][1]["source_data_sha256"] = source_hash_overlap["rows"][2]["source_data_sha256"]
        _reseal(source_hash_overlap)
        with self.assertRaisesRegex(ProvenanceError, "source data hash overlaps"):
            _validate_manifest(source_hash_overlap)

    def test_incorrect_manifest_digest_is_rejected(self):
        manifest = _valid_manifest()
        manifest["rows"][0]["source_id"] = "changed-after-freeze"
        with self.assertRaisesRegex(ProvenanceError, "manifest SHA-256"):
            _validate_manifest(manifest)

    def test_duplicate_game_key_is_rejected(self):
        manifest = _valid_manifest()
        manifest["rows"].append(copy.deepcopy(manifest["rows"][0]))
        _reseal(manifest)
        with self.assertRaisesRegex(ProvenanceError, "duplicate game key"):
            _validate_manifest(manifest)

    def test_final_coverage_and_frozen_status_are_required(self):
        no_final = _valid_manifest()
        no_final["rows"] = [row for row in no_final["rows"] if row["split"] != "final"]
        _reseal(no_final)
        with self.assertRaisesRegex(ProvenanceError, "missing split coverage"):
            _validate_manifest(no_final)

        missing_open_space = _valid_manifest()
        missing_open_space["rows"][3] = _row("final", "other-final", 3)
        _reseal(missing_open_space)
        with self.assertRaisesRegex(ProvenanceError, "exact open-space"):
            _validate_manifest(missing_open_space)

        unfrozen = _valid_manifest()
        unfrozen["frozen"] = False
        _reseal(unfrozen)
        with self.assertRaisesRegex(ProvenanceError, "not frozen"):
            _validate_manifest(unfrozen)

    def test_v1_manifest_and_seed_plan_are_explicitly_rejected(self):
        from unittest.mock import patch

        legacy = _valid_manifest()
        legacy["format"] = "lc-split-manifest/1"
        _reseal(legacy)
        with self.assertRaisesRegex(ProvenanceError, "rejects split-manifest v1"):
            _validate_manifest(legacy)
        from training.eval.g008_manifest import load_manifest
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "legacy.json"
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "rejects split-manifest v1"):
                load_manifest(legacy_path, **_schedule_artifacts_kwargs())
        manifest = _valid_manifest()
        with tempfile.TemporaryDirectory() as directory:
            old_plan = Path(directory) / "v1.json"
            old_plan.write_text('{"format":"circuit-ai-g008-split-seed-plan/1"}', encoding="utf-8")
            with patch("training.eval.provenance.SEED_PLAN_PATH", old_plan):
                with self.assertRaisesRegex(ProvenanceError, "seed.plan /3 bytes"):
                    _validate_manifest(manifest)

        incomplete_v2 = _valid_manifest()
        incomplete_v2["format"] = "lc-split-manifest/2"
        _reseal(incomplete_v2)
        with self.assertRaisesRegex(ProvenanceError, "schema violation"):
            _validate_manifest(incomplete_v2)

    def test_v3_seed_plan_preserves_old_ranges_and_open_space_fingerprints(self):
        plan = json.loads((Path(__file__).resolve().parents[2] /
                           "training/configs/g008-split-seed-plan.json").read_text())
        self.assertEqual(plan["format"], "circuit-ai-g008-split-seed-plan/3")
        self.assertEqual(plan["partitions"]["train"]["root_seed_range_inclusive"], [2_000_000, 2_999_999])
        self.assertEqual(plan["partitions"]["imitation_demo"]["root_seed_range_inclusive"], [3_000_000, 3_999_999])
        self.assertEqual(plan["partitions"]["validation"]["root_seed_range_inclusive"], [1_000_000, 1_000_255])
        self.assertEqual(plan["partitions"]["final"]["root_seed_range_inclusive"], [4_000_000, 4_000_255])
        self.assertEqual(plan["historical_exclusions"]["seeds_inclusive"], [0, 31])
        self.assertEqual(plan["open_space_fingerprint"]["function"]["sha256_inspect_getsource_utf8"], OPEN_SPACE_CODE_SHA256)
        self.assertEqual(plan["open_space_fingerprint"]["config"]["sha256"], OPEN_SPACE_CONFIG_SHA256)
        self.assertEqual(plan["model_ppo_seed"], 1234)
        self.assertEqual(plan["partitions"]["benchmark"]["root_seed_range_inclusive"], [5_000_000, 5_000_271])

    def test_manifest_rejects_missing_or_stale_schedule_bindings(self):
        for field in ("schedule_spec_digest", "schedule_bindings"):
            manifest = _valid_manifest()
            del manifest[field]
            _reseal(manifest)
            with self.subTest(field=field), self.assertRaises(ProvenanceError):
                _validate_manifest(manifest)
        stale = _valid_manifest()
        stale["schedule_spec_digest"] = "0" * 64
        _reseal(stale)
        with self.assertRaisesRegex(ProvenanceError, "schedule_spec_digest"):
            _validate_manifest(stale)

        wrong_plan_binding = _valid_manifest()
        wrong_plan_binding["schedule_bindings"][0]["seed_plan_sha256"] = "0" * 64
        _reseal(wrong_plan_binding)
        with self.assertRaisesRegex(ProvenanceError, "schedule binding has a wrong seed-plan hash"):
            _validate_manifest(wrong_plan_binding)

        stale_run_binding = _valid_manifest()
        stale_run_binding["schedule_bindings"][0]["schedule_spec_digest"] = "0" * 64
        _reseal(stale_run_binding)
        with self.assertRaisesRegex(ProvenanceError, "schedule binding has a missing or stale"):
            _validate_manifest(stale_run_binding)

    def test_manifest_rejects_expanded_schedule_run_id_mismatch(self):
        manifest = _valid_manifest()
        expanded, realized = _schedule_artifacts()
        expanded["fixture-run"] = dict(expanded["fixture-run"], run_id="other-run")
        with self.assertRaisesRegex(ProvenanceError, "expanded schedule run_id does not match"):
            validate_manifest(manifest, expanded_schedules=expanded,
                              realized_schedules=realized)


if __name__ == "__main__":
    unittest.main()
