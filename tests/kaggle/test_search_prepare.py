import ast
import base64
import io
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from tools.kaggle import cli as kaggle_cli
from tools.kaggle.cli import prepare, submit


def test_prepare_experimental_search_is_cpu_only_minimal_and_dry_runnable(tmp_path):
    if not shutil.which("rustc"):
        pytest.skip("rustc is unavailable")
    run_id = "simsearch-campaign-run-test"
    config = {
        "experiment_id": "simsearch-campaign-test",
        "run_id": run_id,
        "width": 4,
        "height": 4,
        "seeds": [7000000000000001],
        "planners": ["fixed-depth", "ducb1-tuned", "regret-matching"],
        "heuristics": [
            "heuristic:cruise", "heuristic:local-mobility", "heuristic:interceptor",
        ],
        "edge_budget": 8,
        "max_depth": 1,
        "max_wall_seconds": 60,
        "native_territory": True,
    }
    settings = {"run_id": run_id, "simultaneous_search": config}
    config_path = tmp_path / "run-config.json"
    config_path.write_text(json.dumps(settings))
    package = prepare(
        "pascalcoimbra/simsearch-test-kernel", "test-search-v1",
        tmp_path / "package", config_path,
    )

    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    manifest = json.loads((package / "source-manifest.json").read_text())
    assert manifest["source_manifest_sha256"]
    assert not any("g008" in item["path"] or ".pt" in item["path"] for item in manifest["files"])
    assert (package / "training/search/tournament.py").is_file()
    assert (package / "training/search/native/libcircuit_territory.so").is_file()
    assert not (package / "training/configs").exists()
    assert not (package / "training/eval").exists()

    script = ast.parse((package / "run_kaggle.py").read_text())
    embedded = next(node.value.value for node in script.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SOURCE_ARCHIVE_B64"
                            for target in node.targets))
    archive_bytes = base64.b64decode(embedded)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        embedded_paths = archive.namelist()
    assert "training/search/tournament.py" in embedded_paths
    assert "training/search/native/libcircuit_territory.so" in embedded_paths
    assert not any("g008" in path or ".pt" in path for path in embedded_paths)

    result = submit("pascalcoimbra/simsearch-test-kernel", "test-search-v1", package,
                    confirm=False, dry_run=True)
    assert result.startswith("DRY RUN:")


def test_prepare_frozen_search_binds_dataset_without_embedding_checkpoint(tmp_path):
    run_id = "simsearch-v15-run-test"
    config = {
        "experiment_id": "simsearch-v15-test", "run_id": run_id,
        "width": 12, "height": 12, "seeds": [7000000000000003],
        "planners": ["fixed-depth", "ducb1-tuned", "regret-matching"],
        "heuristics": ["heuristic:cruise", "heuristic:local-mobility", "heuristic:interceptor"],
        "edge_budget": 8, "max_depth": 1, "max_wall_seconds": 60,
        "frozen_opponent": "v15-step236",
        "matchups": ["fixed-depth-vs-v15-step236"],
    }
    config_path = tmp_path / "run-config-v15.json"
    config_path.write_text(json.dumps({"run_id": run_id, "simultaneous_search": config}))
    package = prepare("pascalcoimbra/simsearch-v15-test", "test-search-v15",
                      tmp_path / "package-v15", config_path)
    expected_sha = "a21a9e00d7e6eb9ad9366502d64b0eebe0cd2e46b191dcee45bf8394a5b5f450"
    local_checkpoint = (Path(__file__).resolve().parents[2]
                        / "artifacts/circuit-ai/evaluations/g008-v15-continuation-236/checkpoints/step-236.pt")
    assert kaggle_cli.file_sha256(local_checkpoint) == expected_sha
    assert not (package / "training/search/assets/v15-step236.pt").exists()
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == ["pascalcoimbra/circuit-ai-v15-step236-experimental"]
    manifest = json.loads((package / "source-manifest.json").read_text())
    staged_checkpoints = [item for item in manifest["files"] if item["path"].endswith(".pt")]
    assert staged_checkpoints == []
    assert not (package / "training/eval").exists()
    assert (package / "run_kaggle.py").stat().st_size < 1_000_000
    script = ast.parse((package / "run_kaggle.py").read_text())
    embedded = next(node.value.value for node in script.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SOURCE_ARCHIVE_B64"
                            for target in node.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(embedded))) as archive:
        staged = [name for name in archive.namelist() if name.endswith(".pt")]
        assert staged == []


def test_prepare_policy_value_search_binds_pinned_dataset_without_gpu(tmp_path):
    run_id = "simsearch-policy-value-test"
    config = {
        "experiment_id": run_id, "run_id": run_id,
        "width": 12, "height": 12, "seeds": [7000000000000007],
        "planners": ["fixed-depth", "ducb1-tuned", "regret-matching"],
        "heuristics": ["heuristic:cruise", "heuristic:local-mobility", "heuristic:interceptor"],
        "edge_budget": 8, "max_depth": 1, "max_wall_seconds": 60,
        "policy_value_checkpoint": "pv-scale-20261022",
        "matchups": ["fixed-depth-vs-pv-scale-20261022"],
    }
    config_path = tmp_path / "run-config-pv.json"
    config_path.write_text(json.dumps({"run_id": run_id, "simultaneous_search": config}))
    package = prepare("pascalcoimbra/simsearch-pv-test", "test-search-pv",
                      tmp_path / "package-pv", config_path)

    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == [
        "pascalcoimbra/circuit-ai-pv-policy-value-scale-20261022"]
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    manifest = json.loads((package / "source-manifest.json").read_text())
    assert not any(item["path"].endswith(".pt") for item in manifest["files"])
    assert (package / "training/search/train_policy_value.py").is_file()


def test_prepare_native_fixed_depth_compiles_the_viewer_rust_source(tmp_path):
    if not shutil.which("rustc"):
        pytest.skip("rustc is unavailable")
    run_id = "simsearch-native-planner-test"
    config = {
        "experiment_id": run_id, "run_id": run_id,
        "width": 12, "height": 12, "seeds": [7000000000000009],
        "planners": ["fixed-depth", "ducb1-tuned", "regret-matching"],
        "heuristics": ["heuristic:cruise", "heuristic:local-mobility", "heuristic:interceptor"],
        "edge_budget": 9, "max_depth": 1, "max_wall_seconds": 60,
        "policy_value_checkpoint": "pv-scale-20261022",
        "native_fixed_depth": True,
        "matchups": ["native-fixed-depth-vs-pv-scale-20261022"],
    }
    config_path = tmp_path / "run-config-native.json"
    config_path.write_text(json.dumps({"run_id": run_id, "simultaneous_search": config}))
    package = prepare("pascalcoimbra/simsearch-native-test", "test-search-native",
                      tmp_path / "package-native", config_path)

    assert (package / "training/search/native/fixed_depth.rs").is_file()
    assert (package / "training/search/native/libcircuit_planner.so").is_file()
    manifest = json.loads((package / "source-manifest.json").read_text())
    paths = {item["path"] for item in manifest["files"]}
    assert "training/search/native/fixed_depth.rs" in paths
    assert "training/search/native/libcircuit_planner.so" in paths
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False


def test_prepare_selected_dagger_checkpoint_is_hash_pinned_and_private_cpu_only(tmp_path):
    run_id = "simsearch-dagger1-run-test"
    config = {
        "experiment_id": run_id, "run_id": run_id,
        "width": 12, "height": 12, "seeds": [7000000000000011],
        "planners": ["fixed-depth", "ducb1-tuned", "regret-matching"],
        "heuristics": ["heuristic:cruise", "heuristic:local-mobility", "heuristic:interceptor"],
        "edge_budget": 9, "max_depth": 1, "max_wall_seconds": 60,
        "native_fixed_depth": True,
        "spatial_policy_checkpoint": "dagger1-selected-20261021",
        "matchups": ["native-fixed-depth-vs-dagger1-selected-20261021"],
    }
    config_path = tmp_path / "run-config-dagger1.json"
    config_path.write_text(json.dumps({"run_id": run_id, "simultaneous_search": config}))
    package = prepare("pascalcoimbra/simsearch-dagger1-test", "test-search-dagger1",
                      tmp_path / "package-dagger1", config_path)

    checkpoint = package / "training/search/assets/dagger1-selected-20261021/model.pt"
    assert kaggle_cli.file_sha256(checkpoint) == (
        "f01978c5ba9edd4f7d544a91115f3739d22c543731d90d3cdc4ea0d96f9a96af")
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == []
    manifest = json.loads((package / "source-manifest.json").read_text())
    assert any(item["path"].endswith("dagger1-selected-20261021/model.pt")
               for item in manifest["files"])


def test_prepare_frozen_search_fails_closed_if_checkpoint_hash_changes(tmp_path, monkeypatch):
    run_id = "simsearch-v15-hash-test"
    config = {
        "experiment_id": "simsearch-v15-hash-test", "run_id": run_id,
        "width": 12, "height": 12, "seeds": [7000000000000005],
        "planners": ["fixed-depth", "ducb1-tuned", "regret-matching"],
        "heuristics": ["heuristic:cruise", "heuristic:local-mobility", "heuristic:interceptor"],
        "edge_budget": 8, "max_depth": 1, "max_wall_seconds": 60,
        "frozen_opponent": "v15-step236", "matchups": ["fixed-depth-vs-v15-step236"],
    }
    config_path = tmp_path / "run-config-v15-hash.json"
    config_path.write_text(json.dumps({"run_id": run_id, "simultaneous_search": config}))
    original = kaggle_cli.file_sha256
    def altered_hash(path):
        if str(path).endswith("evaluations/g008-v15-continuation-236/checkpoints/step-236.pt"):
            return "0" * 64
        return original(path)
    monkeypatch.setattr(kaggle_cli, "file_sha256", altered_hash)
    with pytest.raises(kaggle_cli.KaggleError, match="SHA-256 differs"):
        prepare("pascalcoimbra/simsearch-v15-hash-test", "test-search-v15-hash",
                tmp_path / "package-v15-hash", config_path)
