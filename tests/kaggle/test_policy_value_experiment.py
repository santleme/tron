import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

import pytest

from tools.kaggle import cli as kaggle_cli
from tools.kaggle.cli import KaggleError, prepare, validate_downloaded_output


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_hash(value: dict, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode())


def _data_config(run_id="pv-distill-dev-01"):
    return {"run_id": run_id, "policy_value_data": {
        "experiment_id": run_id, "seeds": [7_000_000_000_001, 7_000_000_000_002],
        "width": 12, "height": 12, "edge_budget": 56, "max_depth": 3,
        "heuristic_styles": ["cruise", "local-mobility", "interceptor"],
        "max_wall_seconds": 3600,
    }}


def _train_config(run_id="pv-train-dev-01"):
    return {"run_id": run_id, "policy_value_train": {
        "dataset_source": "pascalcoimbra/circuit-ai-distill-dev-01",
        "dataset_experiment_id": "pv-distill-dev-01",
        "dataset_file": "examples.npz", "dataset_manifest_file": "manifest.json",
        "dataset_sha256": "a" * 64, "dataset_manifest_sha256": "b" * 64,
        "epochs": 3, "batch_size": 32, "hidden_size": 64,
        "learning_rate": 0.0003, "value_weight": 1.0,
        "validation_fraction": 0.1, "seed": 20261022,
    }}


def _prepare(tmp_path, settings, name):
    config_path = tmp_path / f"{name}.json"
    config_path.write_text(json.dumps(settings))
    return prepare(f"pascalcoimbra/{name}", f"{name}-revision", tmp_path / name, config_path)


def test_prepare_policy_value_data_is_cpu_offline_and_contains_only_arena_search(tmp_path):
    package = _prepare(tmp_path, _data_config(), "pv-data-package")
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == []
    assert (package / "training/arena/engine.py").is_file()
    assert (package / "training/search/distill.py").is_file()
    assert not (package / "training/search/train_policy_value.py").exists()
    assert not (package / "training/ppo").exists()
    assert not (package / "training/eval").exists()
    manifest = json.loads((package / "source-manifest.json").read_text())
    assert all("g008" not in item["path"].lower() and not item["path"].endswith(".pt")
               for item in manifest["files"])
    script = ast.parse((package / "run_kaggle.py").read_text())
    embedded = next(node.value.value for node in script.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SOURCE_ARCHIVE_B64"
                            for target in node.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(embedded))) as archive:
        paths = archive.namelist()
    assert "training/search/distill.py" in paths
    assert not any(path.startswith("training/ppo/") or "g008" in path.lower() for path in paths)


def test_generated_policy_value_data_runner_executes_without_torch(tmp_path):
    settings = _data_config("pv-distill-runner-smoke")
    settings["policy_value_data"].update(
        seeds=[7_000_000_000_011], width=4, height=4, edge_budget=8, max_depth=1,
        max_wall_seconds=60)
    package = _prepare(tmp_path, settings, "pv-data-runner")
    output_root = tmp_path / "remote-output"
    environment = dict(os.environ, ARENA_OUTPUT_DIR=str(output_root))
    subprocess.run(["python3", str(package / "run_kaggle.py")], check=True,
                   capture_output=True, text=True, env=environment, timeout=30)
    output = output_root / settings["run_id"]
    validate_downloaded_output(output, settings["run_id"])
    assert (output / "examples.npz").is_file()
    generated_manifest = json.loads((output / "manifest.json").read_text())
    assert generated_manifest["kaggle_runtime"]["elapsed_generation_seconds"] >= 0
    assert generated_manifest["kaggle_runtime"]["workers"] == 1


def test_policy_value_data_config_accepts_workers_for_runtime_clamping(tmp_path):
    settings = _data_config()
    settings["policy_value_data"]["workers"] = 100
    path = tmp_path / "many-workers.json"
    path.write_text(json.dumps(settings))
    assert kaggle_cli._load_run_settings(path)["policy_value_data"]["workers"] == 100


def test_policy_value_data_config_has_no_artificial_seed_count_cap(tmp_path):
    settings = _data_config()
    settings["policy_value_data"].update(
        seeds=list(range(7_000_000_000_000_001, 7_000_000_000_000_301)), workers=100)
    path = tmp_path / "larger-seed-plan.json"
    path.write_text(json.dumps(settings))
    parsed = kaggle_cli._load_run_settings(path)
    assert len(parsed["policy_value_data"]["seeds"]) == 300
    assert parsed["policy_value_data"]["workers"] == 100


def test_prepare_policy_value_train_is_gpu_bound_and_pins_private_dataset(tmp_path):
    package = _prepare(tmp_path, _train_config(), "pv-train-package")
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["pascalcoimbra/circuit-ai-distill-dev-01"]
    assert (package / "training/search/train_policy_value.py").is_file()
    assert (package / "training/search/policy_value.py").is_file()
    assert (package / "training/ppo/protocol.py").is_file()
    assert not (package / "training/arena").exists()
    assert not (package / "training/eval").exists()
    assert not (package / "training/configs").exists()
    paths = [item["path"] for item in json.loads((package / "source-manifest.json").read_text())["files"]]
    assert "training/search/train_policy_value.py" in paths
    assert "training/search/policy_value.py" in paths
    assert "training/ppo/protocol.py" in paths
    assert not any("g008" in path.lower() or "seed-plan" in path.lower() for path in paths)
    runner = (package / "run_kaggle.py").read_text()
    # Kaggle exposes dataset files through its read-only input mount, which
    # may use symlinks; resolved files must still stay inside the pinned mount.
    assert "mounted_manifest = candidate_manifest.resolve(strict=True)" in runner
    assert "mounted_data = candidate_data.resolve(strict=True)" in runner
    assert "mounted_manifest.is_relative_to(mounted_root)" in runner
    assert "for candidate_root in mount_roots" in runner
    assert 'input_root / "datasets" / dataset_owner / dataset_slug' in runner
    assert "No mounted Policy-Value dataset matched its pinned hashes" in runner


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["policy_value_data"].update(seeds=[1_000_000]), "protected"),
    (lambda c: c["policy_value_data"].update(seeds=[2, 2]), "unique"),
    (lambda c: c["policy_value_data"].update(seeds=[1 << 53]), r"2\^53"),
    (lambda c: c["policy_value_data"].update(heuristic_styles=["open-space"]), "heuristic_styles"),
    (lambda c: c["policy_value_data"].update(max_depth=256, edge_budget=257), "max_depth"),
])
def test_policy_value_data_config_rejects_invalid_or_protected_inputs(tmp_path, mutate, message):
    settings = _data_config()
    mutate(settings)
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(settings))
    with pytest.raises(KaggleError, match=message):
        kaggle_cli._load_run_settings(config_path)


def test_policy_value_modes_are_isolated_but_allow_descriptive_ids(tmp_path):
    both = _data_config()
    both["policy_value_train"] = _train_config()["policy_value_train"]
    path = tmp_path / "both.json"
    path.write_text(json.dumps(both))
    with pytest.raises(KaggleError, match="mutually exclusive"):
        kaggle_cli._load_run_settings(path)

    descriptive = _train_config("open-space-run")
    path.write_text(json.dumps(descriptive))
    assert kaggle_cli._load_run_settings(path)["run_id"] == "open-space-run"

    frozen = _train_config()
    frozen["policy_value_train"]["dataset_source"] = "pascalcoimbra/open-space-development"
    path.write_text(json.dumps(frozen))
    assert kaggle_cli._load_run_settings(path)["policy_value_train"]["dataset_source"].endswith(
        "open-space-development")


def test_policy_value_download_validators_verify_dataset_and_training_hashes(tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    npz = b"test-only-npz-payload"
    (data_root / "examples.npz").write_bytes(npz)
    dataset_manifest = {
        "format": "circuit-ai-distillation-dataset/1", "run_id": "pv-data",
        "experiment_id": "pv-data", "protection_scope":
        "development seeds only; no official/final/reserved split data accessed",
        "kaggle_runtime": {"gpu_enabled": False, "internet_enabled": False},
        "sample_count": 2, "data": {"file": "examples.npz", "format": "npz", "sha256": _sha(npz)},
    }
    dataset_manifest["manifest_sha256"] = _content_hash(dataset_manifest, "manifest_sha256")
    (data_root / "manifest.json").write_text(json.dumps(dataset_manifest))
    validate_downloaded_output(data_root, "pv-data")
    (data_root / "examples.npz").write_bytes(b"changed")
    with pytest.raises(KaggleError, match="manifest-bound hash"):
        validate_downloaded_output(data_root, "pv-data")

    train_root = tmp_path / "trained"
    (train_root / "checkpoints").mkdir(parents=True)
    checkpoint_id = "pv-run-policy-value"
    payloads = {
        "training-report.json": None,
        f"checkpoints/{checkpoint_id}.pt": b"weights",
        f"checkpoints/{checkpoint_id}.onnx": b"onnx",
    }
    metadata = {"format": "lc-checkpoint/1", "checkpoint_id": checkpoint_id,
                "onnx": {"file": f"checkpoints/{checkpoint_id}.onnx", "sha256": _sha(b"onnx")},
                "pytorch": {"file": f"checkpoints/{checkpoint_id}.pt", "sha256": _sha(b"weights")}}
    metadata["metadata_sha256"] = _content_hash(metadata, "metadata_sha256")
    payloads[f"checkpoints/{checkpoint_id}.metadata.json"] = json.dumps(metadata).encode()
    dataset_identity = {"source": "pascalcoimbra/dataset-dev", "sha256": "a" * 64,
                        "manifest_sha256": "b" * 64}
    report = {"format": "lc-search-policy-value-training/1", "run_id": "pv-run",
              "checkpoint_id": checkpoint_id, "dataset": dataset_identity, "device": "cuda"}
    payloads["training-report.json"] = json.dumps(report).encode()
    for relative, body in payloads.items():
        (train_root / relative).write_bytes(body)
    train_manifest = {"format": "lc-search-policy-value-run/1", "run_id": "pv-run",
                      "dataset": dataset_identity, "runtime": {"gpu_enabled": True, "device": "Tesla T4",
                                                                   "torch_version": "test", "cuda_version": "test"},
                      "files": [{"path": relative, "sha256": _sha(body)}
                                for relative, body in sorted(payloads.items())]}
    train_manifest["manifest_sha256"] = _content_hash(train_manifest, "manifest_sha256")
    (train_root / "manifest.json").write_text(json.dumps(train_manifest))
    validate_downloaded_output(train_root, "pv-run")
    (train_root / f"checkpoints/{checkpoint_id}.pt").write_bytes(b"tampered")
    with pytest.raises(KaggleError, match="hash/path mismatch"):
        validate_downloaded_output(train_root, "pv-run")
