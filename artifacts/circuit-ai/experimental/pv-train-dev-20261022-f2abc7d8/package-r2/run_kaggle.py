"""Kaggle script entrypoint; keeps Kaggle's preinstalled CUDA PyTorch intact."""
from __future__ import annotations

import importlib.util
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


SOURCE_ARCHIVE_B64 = "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"
SOURCE_TMP = tempfile.TemporaryDirectory(prefix="circuit-ai-source-")
ROOT = Path(SOURCE_TMP.name)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE_B64))) as archive:
    for item in archive.infolist():
        normalized = item.filename.replace(chr(92), "/")
        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise RuntimeError("Embedded source bundle contains an unsafe path.")
    archive.extractall(ROOT)
sys.path.insert(0, str(ROOT))


def ensure_support_dependencies(torch):
    missing = [name for name in ("onnx", "onnxscript", "onnx_ir", "ml_dtypes", "jsonschema")
               if importlib.util.find_spec(name) is None]
    if not missing:
        return
    # This requirements file intentionally does not list torch. --no-deps
    # prevents pip from replacing the image's CUDA torch or its dependencies.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-input", "--no-deps", "-r", str(ROOT / "training" / "requirements-kaggle.txt"),
    ])
    after = __import__("torch")
    if (after.__version__, after.version.cuda) != (torch.__version__, torch.version.cuda):
        raise RuntimeError("PyTorch/CUDA changed while installing support packages; refusing to train.")
    still_missing = [name for name in missing if importlib.util.find_spec(name) is None]
    if still_missing:
        raise RuntimeError(f"Kaggle image lacks support packages {still_missing}; dependencies were not resolved.")


def constrain_cuda_visibility():
    """Expose one deterministic Kaggle GPU before importing CUDA libraries."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return
    first_device = configured.split(",", 1)[0].strip()
    if not first_device:
        raise RuntimeError("Kaggle benchmark requires at least one visible CUDA device.")
    os.environ["CUDA_VISIBLE_DEVICES"] = first_device


def main():
    settings = json.loads((ROOT / "run-config.json").read_text(encoding="utf-8"))
    if "simultaneous_search" in settings:
        run_id = settings.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("Experimental search run config must contain run_id")
        output_root = Path(os.environ.get("ARENA_OUTPUT_DIR", "/kaggle/working/arena-output"))
        output = output_root / run_id
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing Kaggle output: {output}")
        source_manifest_sha256 = json.loads(
            (ROOT / "source-manifest.json").read_text(encoding="utf-8")
        )["source_manifest_sha256"]
        from training.search.tournament import run_experiment
        run_experiment(settings["simultaneous_search"], output,
                       source_manifest_sha256=source_manifest_sha256)
        return
    if "policy_value_data" in settings:
        import signal
        from training.search.distill import DistillConfig, write_dataset
        config = settings["policy_value_data"]
        run_id = settings["run_id"]
        output_root = Path(os.environ.get("ARENA_OUTPUT_DIR", "/kaggle/working/arena-output"))
        output = output_root / run_id
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing Kaggle output: {output}")
        dataset_config = DistillConfig(
            seeds=tuple(config["seeds"]), width=config["width"], height=config["height"],
            edge_budget=config["edge_budget"], max_depth=config["max_depth"],
            heuristic_styles=tuple(config["heuristic_styles"]))
        class WorkLimit(TimeoutError):
            pass
        def expired(_signum, _frame):
            raise WorkLimit("Policy-Value teacher dataset exceeded its configured wall-time budget")
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(config["max_wall_seconds"])
        try:
            manifest = write_dataset(dataset_config, output, format="npz",
                                     experiment_id=config["experiment_id"])
        finally:
            signal.alarm(0)
        manifest_path = output / "manifest.json"
        manifest["run_id"] = run_id
        manifest["kaggle_runtime"] = {
            "gpu_enabled": False, "internet_enabled": False,
            "python_version": sys.version,
            "source_manifest_sha256": json.loads(
                (ROOT / "source-manifest.json").read_text(encoding="utf-8"))["source_manifest_sha256"],
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                                       ensure_ascii=False).encode()).hexdigest(),
        }
        manifest.pop("manifest_sha256", None)
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2,
                                            ensure_ascii=False, allow_nan=False) + "\n",
                                 encoding="utf-8")
        print(f"Completed CPU-only Policy-Value dataset {run_id}; examples={manifest['sample_count']}")
        return
    if "policy_value_train" in settings:
        config = settings["policy_value_train"]
        source = config["dataset_source"]
        dataset_root = Path("/kaggle/input") / source.split("/", 1)[1]
        manifest_path = dataset_root / config["dataset_manifest_file"]
        dataset_path = dataset_root / config["dataset_file"]
        from training.ppo.protocol import file_sha256
        if (manifest_path.is_symlink() or dataset_path.is_symlink()
                or not manifest_path.is_file() or not dataset_path.is_file()
                or file_sha256(manifest_path) != config["dataset_manifest_sha256"]
                or file_sha256(dataset_path) != config["dataset_sha256"]):
            raise RuntimeError("Mounted Policy-Value dataset file or manifest differs from the pinned SHA-256.")
        try:
            dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Mounted Policy-Value dataset manifest is invalid.") from exc
        digest_document = dict(dataset_manifest)
        recorded_manifest_sha = digest_document.pop("manifest_sha256", None)
        canonical_manifest_sha = hashlib.sha256(json.dumps(
            digest_document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")).hexdigest()
        data_record = dataset_manifest.get("data", {})
        dataset_config = dataset_manifest.get("config", {})
        dataset_sample_count = dataset_manifest.get("sample_count")
        seeds = dataset_config.get("seeds") if isinstance(dataset_config, dict) else None
        protected_seed_ranges = ((0, 31), (1_000_000, 1_000_255),
                                 (2_000_000, 2_999_999), (3_000_000, 3_999_999),
                                 (4_000_000, 4_000_255), (5_000_000, 5_000_271))
        styles = dataset_config.get("heuristic_styles") if isinstance(dataset_config, dict) else None
        allowed_styles = {"cruise", "local-mobility", "interceptor"}
        width = dataset_config.get("width") if isinstance(dataset_config, dict) else None
        height = dataset_config.get("height") if isinstance(dataset_config, dict) else None
        edge_budget = dataset_config.get("edge_budget") if isinstance(dataset_config, dict) else None
        max_depth = dataset_config.get("max_depth") if isinstance(dataset_config, dict) else None
        sample_bound_ok = (
            type(width) is int and type(height) is int and width >= 2 and height >= 2
            and width <= 32 and height <= 32 and width * height <= 1024
            and type(seeds) is list and 1 <= len(seeds) <= 256
            and all(type(seed) is int and 0 <= seed < (1 << 53) for seed in seeds)
            and len(seeds) == len(set(seeds))
            and not any(lower <= seed <= upper for seed in seeds
                        for lower, upper in protected_seed_ranges)
            and type(edge_budget) is int and 8 <= edge_budget <= 512
            and type(max_depth) is int and 1 <= max_depth <= 12
            and edge_budget >= max_depth + 1
            and isinstance(styles, list) and 1 <= len(styles) <= 3
            and len(styles) == len(set(styles)) and set(styles) <= allowed_styles
            and type(dataset_sample_count) is int and dataset_sample_count > 0
            and dataset_sample_count
                <= len(seeds) * 2 * (width * height + 1) <= 50_000)
        expected_source_paths = {
            "training/arena/engine.py", "training/arena/heuristics.py",
            "training/arena/seeds.py", "training/search/bitboard.py",
            "training/search/matrix.py", "training/search/planners.py",
            "training/search/territory.py", "training/search/native/territory.py",
            "training/search/native/territory.rs", "training/search/distill.py",
        }
        source_hashes = dataset_manifest.get("source_sha256", {})
        if (dataset_manifest.get("format") != "circuit-ai-distillation-dataset/1"
                or dataset_manifest.get("experiment_id") != config["dataset_experiment_id"]
                or dataset_manifest.get("run_id") != config["dataset_experiment_id"]
                or dataset_manifest.get("protection_scope") !=
                   "development seeds only; no official/final/reserved split data accessed"
                or dataset_manifest.get("teacher", {}).get("algorithm") != "fixed-depth"
                or not sample_bound_ok
                or dataset_manifest.get("opponents") != [f"heuristic:{style}" for style in styles]
                or dataset_manifest.get("sample_schema", {}).get("id") != "circuit-ai-policy-value-sample/1"
                or not isinstance(source_hashes, dict) or set(source_hashes) != expected_source_paths
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(value))
                       for value in source_hashes.values())
                or recorded_manifest_sha != canonical_manifest_sha
                or data_record.get("file") != config["dataset_file"]
                or data_record.get("sha256") != config["dataset_sha256"]):
            raise RuntimeError("Mounted Policy-Value dataset manifest identity or embedded digest is invalid.")
        if (config["epochs"] *
                ((dataset_sample_count + config["batch_size"] - 1) // config["batch_size"]) > 20_000):
            raise RuntimeError("Policy-Value training exceeds the bounded Kaggle optimizer-step budget.")
        constrain_cuda_visibility()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Policy-Value Kaggle training requires CUDA; refusing CPU fallback.")
        torch.cuda.set_device(0)
        ensure_support_dependencies(torch)
        run_id = settings["run_id"]
        output_root = Path(os.environ.get("ARENA_OUTPUT_DIR", "/kaggle/working/arena-output"))
        output = output_root / run_id
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing Kaggle output: {output}")
        output.mkdir(parents=True)
        from training.search.train_policy_value import TrainConfig, train_from_npz
        trainer_config = {key: config[key] for key in (
            "epochs", "batch_size", "hidden_size", "learning_rate", "value_weight",
            "validation_fraction", "seed")}
        trainer_config["device"] = "cuda"
        runtime = {"gpu_enabled": True, "device": torch.cuda.get_device_name(0),
                   "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                   "source_manifest_sha256": json.loads(
                       (ROOT / "source-manifest.json").read_text(encoding="utf-8"))["source_manifest_sha256"]}
        report = train_from_npz(dataset_path, TrainConfig(**trainer_config), output,
                                run_id=run_id, export_onnx=True, dataset_source=source)
        report["dataset"] = {"source": source, "sha256": config["dataset_sha256"],
                             "manifest_sha256": config["dataset_manifest_sha256"]}
        report["runtime"] = runtime
        (output / "training-report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset"] = report["dataset"]
        manifest["runtime"] = runtime
        manifest["files"] = [{"path": path.relative_to(output).as_posix(),
                              "sha256": file_sha256(path)}
                             for path in sorted(output.rglob("*"))
                             if path.is_file() and path != manifest_path]
        manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2,
                                            ensure_ascii=False, allow_nan=False) + "\n",
                                 encoding="utf-8")
        print(f"Completed CUDA Policy-Value training {run_id}; artifacts: {output}")
        return
    # Import/check PyTorch before any optional dependency installation. Kaggle
    # training is deliberately GPU-only; local training retains CPU support.
    # Kaggle can initialize CUDA before this script runs, so changing
    # CUDA_VISIBLE_DEVICES may not affect torch.cuda.device_count(). Select
    # cuda:0 explicitly and record visible devices separately from usage.
    constrain_cuda_visibility()
    import torch
    settings = json.loads((ROOT / "run-config.json").read_text(encoding="utf-8"))
    campaign_id = (settings.get("g008_campaign_train", {}).get("campaign_id")
                   or settings.get("g008_evaluation", {}).get("campaign_id"))
    v2_nonfinal = (campaign_id == "g008-competitive-20261021-v2"
                   and settings.get("g008_evaluation", {}).get("mode") != "final")
    if v2_nonfinal:
        forbidden_sources = (
            ROOT / "training/configs/g008-split-seed-plan.json",
            ROOT / "training/eval/provenance.py",
            ROOT / "training/eval/registry.py",
            ROOT / "training/eval/lc_eval.py",
            ROOT / "training/eval/g008_final_roster.py",
            ROOT / "training/eval/g008_final_opponent.py",
            ROOT / "training/ppo/g008_benchmark.py",
            ROOT / "training/ppo/g008_incumbent",
            ROOT / "schemas/g008-split-seed-plan-3.schema.json",
        )
        if any(path.exists() for path in forbidden_sources):
            raise RuntimeError("Nonfinal G008 campaign package contains final-only source or seed data.")
        if not (ROOT / "training/eval/g008_campaign_seed.py").is_file():
            raise RuntimeError("Nonfinal G008 campaign package is missing its safe seed codec.")
    requested_device = settings.get("train_config", {}).get("device", "auto")
    if requested_device == "cpu":
        raise RuntimeError("Kaggle self-play training requires CUDA; device=cpu is not allowed.")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("Kaggle self-play training requires an available CUDA GPU; refusing CPU fallback.")
    device_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    torch.cuda.set_device(0)
    print(f"Preinstalled PyTorch={torch.__version__} CUDA={torch.version.cuda} available={cuda_available} "
          f"visible_cuda_devices={device_names}; training_device={device_names[0] if device_names else 'none'}")
    run_id = settings.get("run_id") or settings.get("g008_evaluation", {}).get("run_uuid")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("Kaggle run config must identify run_id or g008_evaluation.run_uuid")
    campaign_runtime = any(key in settings for key in
                           ("g008_principal_benchmark", "g008_train_schedule", "g008_campaign_train",
                            "g008_evaluation"))
    if campaign_runtime and (not device_names or device_names[0] != "Tesla T4"):
        raise RuntimeError(f"Registered G008 workloads require cuda:0 to be a Tesla T4, got {device_names}.")
    runtime_evidence = {
        "format": "circuit-ai-kaggle-runtime-evidence/1", "run_id": run_id,
        "python_version": sys.version, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "visible_cuda_devices": device_names,
        "visible_cuda_device_count": len(device_names),
        "training_device": device_names[0] if device_names else None,
        "training_device_index": 0, "training_cuda_device_count": 1,
        "source_manifest_sha256": json.loads(
            (ROOT / "source-manifest.json").read_text(encoding="utf-8")
        )["source_manifest_sha256"],
    }
    if "train_config" in settings:
        settings["train_config"]["device"] = "cuda"
    output_root = Path(os.environ.get("ARENA_OUTPUT_DIR", "/kaggle/working/arena-output"))
    output = output_root / run_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Kaggle output: {output}")
    output.mkdir(parents=True)
    g008_settings = settings.get("g008_principal_benchmark")
    if g008_settings is not None:
        from training.ppo.g008_benchmark import run_principal_benchmark
        report = run_principal_benchmark(output, run_ids=g008_settings["run_ids"])
        report_path = output / "g008-principal-benchmark.json"
        from training.ppo.protocol import file_sha256
        evidence = {
            "format": "circuit-ai-kaggle-g008-benchmark-evidence/1",
            "run_id": run_id,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "visible_cuda_devices": device_names,
            "visible_cuda_device_count": len(device_names),
            "training_device": device_names[0] if device_names else None,
            "training_device_index": 0, "training_cuda_device_count": 1,
            "source_manifest_sha256": runtime_evidence["source_manifest_sha256"],
            "benchmark_report_sha256": file_sha256(report_path),
            "benchmark_report": report,
        }
        (output / "kaggle-execution-evidence.json").write_text(
            json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"Completed G008 principal benchmark {run_id}; artifacts: {output}")
        return
    ensure_support_dependencies(torch)
    evaluation_settings = settings.get("g008_evaluation")
    if evaluation_settings is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Registered G008 evaluation requires the Kaggle CUDA runtime.")
        from training.eval.g008_validation import evaluate_registered
        from training.ppo.protocol import digest_json, file_sha256
        evaluation_settings = dict(evaluation_settings)
        if (campaign_id == "g008-competitive-20261021-v2"
                and evaluation_settings.get("mode") == "validation"):
            packaged_opponent = ROOT / "training/eval/assets/g007-step-300.pt"
            mounted_opponent = Path(
                "/kaggle/input/g008-frozen-candidate/g007-step-300.pt")
            if file_sha256(mounted_opponent) != evaluation_settings.get("opponent_checkpoint_sha256"):
                raise RuntimeError("Mounted G007 step-300 dataset artifact differs from the frozen opponent hash.")
            packaged_opponent.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mounted_opponent, packaged_opponent)
        evaluation_settings_sha256 = digest_json(evaluation_settings, "__no_digest_field__")
        evaluation_settings.setdefault("device", "cuda")
        report = evaluate_registered(evaluation_settings, output,
                                    config_file_sha256=file_sha256(ROOT / "run-config.json"),
                                    settings_sha256=evaluation_settings_sha256,
                                    runtime_evidence=runtime_evidence)
        print(f"Completed registered G008 {evaluation_settings['mode']} evaluation; "
              f"games={report['game_count']}; artifacts: {output}")
        return
    from training.ppo.train import TrainConfig, train
    config = TrainConfig(**settings["train_config"])
    if config.device != "cuda":
        raise RuntimeError("Kaggle training device must resolve to CUDA.")
    g008_train = settings.get("g008_train_schedule")
    if g008_train is not None:
        from training.eval.provenance import build_training_run_schedule, seed_plan_sha256
        schedule = build_training_run_schedule(
            run_id=g008_train["run_uuid"], root_seed=g008_train["root_seed"],
            arena_count=config.arena_count, total_updates=config.total_updates,
            rollout_steps=config.rollout_steps)
        if (schedule["expanded_schedule_digest"] != g008_train["expanded_schedule_digest"]
                or schedule["schedule_spec_digest"] != g008_train["schedule_spec_digest"]
                or seed_plan_sha256() != g008_train["seed_plan_sha256"]):
            raise RuntimeError("G008 training schedule differs from its frozen package binding.")
        train(config, output, export_onnx=True, run_evaluation=False,
              g008_schedule=schedule, run_id=g008_train["run_uuid"])
        runtime_evidence["run_id"] = g008_train["run_uuid"]
        (output / "kaggle-execution-evidence.json").write_text(
            json.dumps(runtime_evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"Completed G008 train-only workload {run_id}; artifacts: {output}")
        return
    g008_campaign_train = settings.get("g008_campaign_train")
    if g008_campaign_train is not None:
        report = train(config, output, export_onnx=True, run_evaluation=False,
                       g008_campaign_train=g008_campaign_train,
                       run_id=g008_campaign_train["run_uuid"])
        runtime_evidence["run_id"] = g008_campaign_train["run_uuid"]
        (output / "kaggle-execution-evidence.json").write_text(
            json.dumps(runtime_evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"Completed registered G008 campaign train {run_id}; "
              f"updates={len(report['training']['updates'])}; artifacts: {output}")
        return
    train(config, output, export_onnx=True)
    benchmark_settings = settings.get("throughput_benchmark")
    if benchmark_settings is not None:
        from dataclasses import replace
        from training.ppo.benchmark import build_candidates, run_benchmark
        from training.ppo.protocol import digest_json, file_sha256
        benchmark_root = output / ".throughput-benchmark-work"
        base = TrainConfig(**settings["train_config"])
        candidates = [replace(candidate, total_updates=benchmark_settings["updates"])
                      for candidate in build_candidates(benchmark_settings["configs"], base)]
        if any(candidate.device != "cuda" for candidate in candidates):
            raise RuntimeError("Kaggle throughput benchmark must use the same CUDA device as training.")
        benchmark_report = run_benchmark(candidates, benchmark_root)
        report_path = output / "benchmark-results.json"
        report_path.write_text(json.dumps(benchmark_report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if benchmark_root.exists():
            shutil.rmtree(benchmark_root)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative = report_path.relative_to(output).as_posix()
        manifest.setdefault("files", []).append({"path": relative, "sha256": file_sha256(report_path)})
        manifest.setdefault("benchmark_records", []).append({
            "protocol": benchmark_report["format"], "report_file": relative,
            "selected_trial": benchmark_report["selected_trial"],
            "selection_metric": benchmark_report["selection_metric"],
            "device_mode": benchmark_report["device_mode"],
            "execution_model": benchmark_report["execution_model"],
        })
        manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Completed run {run_id}; artifacts: {output}")


if __name__ == "__main__":
    main()
