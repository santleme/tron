import hashlib
import json

import pytest

from training.search.distill import DistillConfig, iter_examples, write_dataset


def _small_config():
    return DistillConfig(seeds=(193, 281), width=5, height=5,
                         edge_budget=8, max_depth=2,
                         heuristic_styles=("cruise", "local-mobility"))


def test_distillation_examples_have_canonical_policy_value_and_legality_targets():
    config = _small_config()
    examples = list(iter_examples(config))
    assert examples
    assert {row["seed"] for row in examples} == set(config.seeds)
    assert {row["seat"] for row in examples} == {0, 1}
    assert {row["opponent"] for row in examples} == {
        "fixed-depth", "heuristic:cruise", "heuristic:local-mobility"
    }
    assert {row["seed"]: row["trajectory_teacher_seat"] for row in examples
            if row["tick"] == 0 and row["seat"] == row["trajectory_teacher_seat"]} == {
                193: 0, 281: 1
            }

    for row in examples:
        assert len(row["observation"]) == config.height
        assert all(len(line) == config.width for line in row["observation"])
        assert all(len(cell) == 12 for line in row["observation"] for cell in line)
        assert len(row["policy"]) == len(row["legal_mask"]) == 4
        assert set(row["legal_mask"]) <= {0, 1}
        assert any(row["legal_mask"])
        assert all(probability >= 0 for probability in row["policy"])
        assert sum(row["policy"]) == pytest.approx(1.0)
        assert all(not probability or row["legal_mask"][action]
                   for action, probability in enumerate(row["policy"]))
        assert -1 <= row["value"] <= 1
        assert row["teacher"] == "fixed-depth"
        assert row["teacher_edge_budget"] == config.edge_budget
        assert row["opponent"] == (f"heuristic:{'cruise' if row['seed'] == 193 else 'local-mobility'}"
                                   if row["seat"] == row["trajectory_teacher_seat"]
                                   else "fixed-depth")

    assert examples == list(iter_examples(config))


def test_parallel_seed_generation_preserves_order_and_targets():
    config = _small_config()
    assert list(iter_examples(config, workers=2)) == list(iter_examples(config, workers=1))
    assert list(iter_examples(config, workers=5)) == list(iter_examples(config, workers=1))
    with pytest.raises(ValueError, match="workers"):
        list(iter_examples(config, workers=0))


def test_jsonl_writer_records_hash_pinned_manifest_without_overwriting(tmp_path):
    output = tmp_path / "distillation"
    manifest = write_dataset(_small_config(), output, experiment_id="unit-test")

    data_path = output / "examples.jsonl"
    manifest_path = output / "manifest.json"
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    saved_manifest = json.loads(manifest_path.read_text())
    assert len(rows) == manifest["sample_count"] == saved_manifest["sample_count"]
    assert saved_manifest["experiment_id"] == "unit-test"
    assert saved_manifest["run_id"] == "unit-test"
    assert saved_manifest["data"]["sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert saved_manifest["manifest_sha256"] == manifest["manifest_sha256"]
    assert set(saved_manifest["config"]["seeds"]) == {193, 281}
    assert saved_manifest["source_sha256"]["training/arena/engine.py"]
    with pytest.raises(FileExistsError):
        write_dataset(_small_config(), output)


def test_npz_writer_exports_expected_training_arrays(tmp_path):
    np = pytest.importorskip("numpy")
    output = tmp_path / "npz-distillation"
    manifest = write_dataset(_small_config(), output, format="npz", workers=2)
    assert manifest["generation_workers"] == 2
    with np.load(output / "examples.npz") as data:
        count = manifest["sample_count"]
        assert data["observations"].shape == (count, 5, 5, 12)
        assert data["policies"].shape == (count, 4)
        assert data["values"].shape == (count,)
        assert data["legal_masks"].shape == (count, 4)
        assert data["seats"].shape == (count,)
        assert data["trajectory_teacher_seats"].shape == (count,)
        assert data["teacher_depths"].shape == (count,)
        assert data["teacher_edge_budgets"].shape == (count,)
        assert np.allclose(data["policies"].sum(axis=1), 1.0)


@pytest.mark.parametrize("seed", [0, 31, 1_000_000, 1_000_255, 2_000_000,
                                   2_999_999, 3_000_000, 3_999_999, 4_000_000,
                                   4_000_255, 5_000_000, 5_000_271])
def test_config_rejects_protected_g008_root_seeds(seed):
    with pytest.raises(ValueError, match="protected G008"):
        DistillConfig(seeds=(seed,))
