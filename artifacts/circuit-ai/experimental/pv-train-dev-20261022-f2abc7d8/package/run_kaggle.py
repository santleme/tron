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
