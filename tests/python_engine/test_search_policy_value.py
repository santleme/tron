from __future__ import annotations

import numpy as np
import pytest

from training.search.train_policy_value import TrainConfig, load_dataset


def _write_dataset(path, *, masks_dtype=np.uint8, include_all=True):
    observations = np.zeros((4, 6, 8, 12), dtype=np.uint8)
    observations[:, 2, 2, 0] = 1
    observations[:, 2, 2, 4] = 1
    policies = np.zeros((4, 4), dtype=np.float32)
    policies[:, 0] = 1
    values = np.asarray([1, -1, 0.25, 0], dtype=np.float32)
    masks = np.zeros((4, 4), dtype=masks_dtype)
    masks[:, :3] = 1
    arrays = {"observations": observations, "policies": policies,
              "values": values, "legal_masks": masks,
              "seeds": np.asarray([101, 101, 202, 202], dtype=np.int64)}
    if not include_all:
        arrays.pop("legal_masks")
    np.savez_compressed(path, **arrays)
    return arrays


def test_loads_compact_distillation_npz_and_normalizes_legal_masks(tmp_path):
    path = tmp_path / "examples.npz"
    expected = _write_dataset(path)

    actual = load_dataset(path)

    assert actual["observations"].shape == (4, 6, 8, 12)
    assert actual["observations"].dtype == np.uint8
    assert actual["policies"].dtype == np.float32
    assert actual["values"].dtype == np.float32
    assert actual["legal_masks"].dtype == np.bool_
    np.testing.assert_array_equal(actual["legal_masks"], expected["legal_masks"].astype(bool))


def test_loads_boolean_masks_too(tmp_path):
    path = tmp_path / "examples.npz"
    _write_dataset(path, masks_dtype=np.bool_)
    assert load_dataset(path)["legal_masks"].dtype == np.bool_


def test_seed_grouped_split_never_leaks_one_episode_across_training_and_validation():
    from training.search.train_policy_value import _grouped_seed_split

    seeds = np.asarray([10, 10, 10, 20, 20, 30, 30, 30, 30], dtype=np.int64)
    training, validation = _grouped_seed_split(seeds, 0.34, 20261022)

    training_seeds = set(seeds[training].tolist())
    validation_seeds = set(seeds[validation].tolist())
    assert training_seeds
    assert validation_seeds
    assert training_seeds.isdisjoint(validation_seeds)
    assert len(training) + len(validation) == len(seeds)


def test_loader_requires_seed_ids_for_grouped_validation(tmp_path):
    path = tmp_path / "seedless.npz"
    arrays = _write_dataset(path)
    arrays.pop("seeds")
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="missing arrays.*seeds"):
        load_dataset(path)


def test_rejects_missing_arrays_and_bad_policy_targets(tmp_path):
    missing = tmp_path / "missing.npz"
    _write_dataset(missing, include_all=False)
    with pytest.raises(ValueError, match="missing arrays"):
        load_dataset(missing)

    illegal = tmp_path / "illegal.npz"
    arrays = _write_dataset(illegal)
    arrays["policies"][0, 3] = 1
    arrays["policies"][0, 0] = 0
    np.savez_compressed(illegal, **arrays)
    with pytest.raises(ValueError, match="illegal action"):
        load_dataset(illegal)


@pytest.mark.parametrize("mutator, message", [
    (lambda cfg: {**cfg, "epochs": 0}, "epochs must be a positive integer"),
    (lambda cfg: {**cfg, "learning_rate": 0}, "learning_rate must be positive"),
    (lambda cfg: {**cfg, "validation_fraction": 1.0}, "validation_fraction must be in \\[0, 1\\)"),
])
def test_train_config_rejects_invalid_values(mutator, message):
    from dataclasses import asdict

    config = TrainConfig(**mutator(asdict(TrainConfig())))
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_training_smoke_is_cpu_runnable_when_torch_is_available(tmp_path):
    pytest.importorskip("torch")
    from training.search.train_policy_value import train

    dataset = tmp_path / "examples.npz"
    _write_dataset(dataset)
    report = train(dataset, tmp_path / "run", TrainConfig(
        epochs=1, batch_size=2, hidden_size=8, validation_fraction=0.25,
        seed=77, device="cpu"), export_onnx=False)

    assert report["examples"] == 4
    assert report["device"] == "cpu"
    assert (tmp_path / "run" / "checkpoints" / "search-pv-seed-77.pt").is_file()
    assert (tmp_path / "run" / "training-report.json").is_file()


def test_policy_value_model_uses_viewer_action_mask_version_two():
    torch = pytest.importorskip("torch")
    from training.search.train_policy_value import _model_class

    model = _model_class(torch)(4, 4, 8).eval()
    observation = torch.zeros((1, 4, 4, 12), dtype=torch.float32)
    observation[0, 1, 1, 0] = 1  # rider at (1, 1), heading right
    observation[0, 1, 1, 5] = 1
    observation[0, 0, 1, 2] = 1  # occupied cell blocks UP
    with torch.no_grad():
        logits, values = model(observation)

    assert logits.shape == (1, 4)
    assert values.shape == (1,)
    assert logits[0, 0] < -1e30  # blocked UP
    assert torch.isfinite(logits[0, 1])  # RIGHT remains legal
    assert torch.isfinite(logits[0, 2])  # DOWN remains legal
    assert logits[0, 3] < -1e30  # reverse LEFT
