"""Consumer-boundary rejection tests for frozen G008 manifest inputs."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from training.eval.provenance import ProvenanceError
from training.eval.g008_manifest import load_manifest, rows_for_usage
from tests.eval.test_provenance import (
    _valid_manifest,
    _reseal,
    _schedule_artifacts_kwargs,
)


class G008ManifestBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "manifest.json"
        self.schedule_kwargs = _schedule_artifacts_kwargs()

    def _assert_both_boundaries_reject(self, manifest, message, *, seed_plan_path=None):
        self.path.write_text(json.dumps(manifest), encoding="utf-8")
        seed_plan_patch = (patch("training.eval.provenance.SEED_PLAN_PATH", seed_plan_path)
                           if seed_plan_path is not None else nullcontext())
        with seed_plan_patch:
            with self.subTest(boundary="load_manifest"), self.assertRaisesRegex(
                    ProvenanceError, message):
                load_manifest(self.path, **self.schedule_kwargs)
            with self.subTest(boundary="rows_for_usage"), self.assertRaisesRegex(
                    ProvenanceError, message):
                rows_for_usage(manifest, "train", **self.schedule_kwargs)

    def test_manifest_v1_is_rejected_before_rows_are_exposed(self):
        manifest = _valid_manifest()
        manifest["format"] = "lc-split-manifest/1"
        _reseal(manifest)
        self._assert_both_boundaries_reject(manifest, "rejects split-manifest v1")

    def test_incomplete_manifest_v2_is_rejected_before_rows_are_exposed(self):
        manifest = _valid_manifest()
        manifest["format"] = "lc-split-manifest/2"
        _reseal(manifest)
        self._assert_both_boundaries_reject(manifest, "schema violation")

    def test_wrong_approved_seed_plan_hash_is_rejected_before_rows_are_exposed(self):
        manifest = _valid_manifest()
        manifest["seed_plan_sha256"] = "0" * 64
        _reseal(manifest)
        self._assert_both_boundaries_reject(manifest, "seed_plan_sha256")

    def test_wrong_seed_plan_version_is_rejected_before_rows_are_exposed(self):
        old_plan = Path(self.directory.name) / "seed-plan-v2.json"
        old_plan.write_text(json.dumps({"format": "circuit-ai-g008-split-seed-plan/2"}),
                            encoding="utf-8")
        manifest = _valid_manifest()
        self._assert_both_boundaries_reject(manifest, "seed-plan schema violation",
                                            seed_plan_path=old_plan)

    def test_wrong_exact_approved_seed_plan_bytes_are_rejected_before_rows_are_exposed(self):
        source_plan = Path(__file__).resolve().parents[2] / "training/configs/g008-split-seed-plan.json"
        altered_plan = Path(self.directory.name) / "seed-plan-altered.json"
        altered_plan.write_bytes(source_plan.read_bytes() + b" ")
        manifest = _valid_manifest()
        self._assert_both_boundaries_reject(
            manifest, "exact approved seed-plan /3 bytes", seed_plan_path=altered_plan)

    def test_missing_and_stale_schedule_bindings_are_rejected_before_rows(self):
        missing = _valid_manifest()
        del missing["schedule_bindings"]
        _reseal(missing)
        self._assert_both_boundaries_reject(missing, "schema violation")

        stale = copy.deepcopy(_valid_manifest())
        stale["schedule_bindings"][0]["schedule_spec_digest"] = "0" * 64
        _reseal(stale)
        self._assert_both_boundaries_reject(stale, "schedule binding has a missing or stale")


if __name__ == "__main__":
    unittest.main()
