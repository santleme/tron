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
