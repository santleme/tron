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
import time
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
        generation_started = time.monotonic()
        try:
            manifest = write_dataset(dataset_config, output, format="npz",
                                     experiment_id=config["experiment_id"],
                                     workers=config.get("workers", 1))
        finally:
            signal.alarm(0)
        generation_seconds = time.monotonic() - generation_started
        manifest_path = output / "manifest.json"
        manifest["run_id"] = run_id
        manifest["kaggle_runtime"] = {
            "gpu_enabled": False, "internet_enabled": False,
            "elapsed_generation_seconds": round(generation_seconds, 3),
            "workers": config.get("workers", 1),
            "sample_cap": 50_000,
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
        input_root = Path("/kaggle/input")
        dataset_owner, dataset_slug = source.split("/", 1)
        dataset_root = input_root / dataset_slug
        from training.ppo.protocol import file_sha256
        mount_entries = []
        if input_root.is_dir():
            mount_entries = sorted(entry.name for entry in input_root.iterdir())
        mount_roots = [dataset_root, input_root / dataset_owner / dataset_slug,
                       input_root / "datasets" / dataset_owner / dataset_slug,
                       input_root / "datasets" / dataset_slug]
        mount_roots.extend(input_root / name for name in mount_entries if name != dataset_slug)
        if input_root.is_dir():
            mount_roots.extend(path.parent for path in input_root.rglob(
                config["dataset_manifest_file"]))
        mount_roots = list(dict.fromkeys(mount_roots))
        resolved_pair = None
        for candidate_root in mount_roots:
            try:
                mounted_root = candidate_root.resolve(strict=True)
                candidate_manifest = candidate_root / config["dataset_manifest_file"]
                candidate_data = candidate_root / config["dataset_file"]
                mounted_manifest = candidate_manifest.resolve(strict=True)
                mounted_data = candidate_data.resolve(strict=True)
            except OSError:
                continue
            if (mounted_manifest.is_relative_to(mounted_root)
                    and mounted_data.is_relative_to(mounted_root)
                    and mounted_manifest.is_file() and mounted_data.is_file()
                    and file_sha256(mounted_manifest) == config["dataset_manifest_sha256"]
                    and file_sha256(mounted_data) == config["dataset_sha256"]):
                resolved_pair = (candidate_manifest, candidate_data)
                break
        if resolved_pair is None:
            raise RuntimeError(
                "No mounted Policy-Value dataset matched its pinned hashes; "
                f"source={source!r}, input_entries={mount_entries!r}.")
        manifest_path, dataset_path = resolved_pair
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
            and type(dataset_sample_count) is int and 0 < dataset_sample_count <= 50_000
            and dataset_sample_count <= len(seeds) * 2 * (width * height + 1))
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
        training_started = time.monotonic()
        report = train_from_npz(dataset_path, TrainConfig(**trainer_config), output,
                                run_id=run_id, export_onnx=True, dataset_source=source)
        runtime["elapsed_training_seconds"] = round(time.monotonic() - training_started, 3)
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
