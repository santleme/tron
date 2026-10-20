from __future__ import annotations

import argparse
import base64
from dataclasses import fields, replace
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence
import zipfile

from training.arena.engine import ACTION_ORDER, Agent, State, initialize_episode, step
from training.ppo.protocol import digest_json, file_sha256, safe_relative_path


class KaggleError(RuntimeError):
    """An actionable, safe-to-display Kaggle wrapper error."""


def _run(args: Sequence[str], *, timeout: int = 120) -> str:
    """Call Kaggle without a shell and avoid echoing credential-bearing output."""
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise KaggleError("Kaggle CLI not found. Install/configure it, then retry this explicit command.") from exc
    except subprocess.TimeoutExpired as exc:
        raise KaggleError("Kaggle CLI command timed out; no automatic retry was attempted.") from exc
    if result.returncode:
        # Kaggle's error text can contain account details. Keep the command boundary
        # useful without copying arbitrary stderr into logs or artifacts.
        raise KaggleError(f"Kaggle CLI failed (exit {result.returncode}); check Kaggle CLI authentication and kernel access.")
    return result.stdout.strip()


def _kernel_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value or ""):
        raise KaggleError("Kernel ID must be an explicit owner/slug, for example owner/light-cycles.")
    return value


def _version(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise KaggleError("Kernel version must be an explicit positive integer.") from exc
    if parsed < 1 or str(parsed) != str(value):
        raise KaggleError("Kernel version must be an explicit positive integer.")
    return parsed


def _status_state(text: str) -> str:
    normalized = text.strip().lower()
    if any(token in normalized for token in ("complete", "completed", "success", "succeeded")):
        return "success"
    if any(token in normalized for token in ("error", "failed", "cancel", "canceled")):
        return "error"
    if any(token in normalized for token in ("running", "queued", "pending", "starting")):
        return "running"
    raise KaggleError("Kaggle returned an unrecognized kernel status; inspect it with the Kaggle CLI.")


def poll_status(kernel_id: str, version: int, *, max_attempts: int = 30,
                interval_seconds: float = 10, run: Callable[[Sequence[str]], str] = _run,
                sleep: Callable[[float], None] = time.sleep) -> tuple[str, str]:
    if max_attempts < 1 or max_attempts > 360 or interval_seconds < 0 or interval_seconds > 300:
        raise KaggleError("Polling limits are invalid (max 360 attempts, interval 0..300 seconds).")
    latest = ""
    for attempt in range(max_attempts):
        # Version-qualified kernel IDs avoid accidentally observing a newer run.
        latest = run(["kaggle", "kernels", "status", f"{kernel_id}/{version}"])
        state = _status_state(latest)
        if state != "running":
            return state, latest
        if attempt + 1 < max_attempts:
            sleep(interval_seconds)
    return "timeout", latest


def _metadata(source: Path, kernel_id: str, source_revision: str) -> None:
    if not source.is_dir():
        raise KaggleError("Selected kernel source directory does not exist.")
    metadata_path = source / "kernel-metadata.json"
    revision_path = source / ".arena-source-revision"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded_revision = revision_path.read_text(encoding="utf-8").strip()
    except (OSError, json.JSONDecodeError) as exc:
        raise KaggleError("Source must contain valid kernel-metadata.json and .arena-source-revision files.") from exc
    if metadata.get("id") != kernel_id:
        raise KaggleError("kernel-metadata.json id does not match the explicitly selected kernel ID.")
    if not source_revision or recorded_revision != source_revision:
        raise KaggleError("Selected source revision does not match .arena-source-revision in the kernel directory.")


def _kernel_list_contains(output: str, kernel_id: str) -> bool:
    """Parse Kaggle's table-only kernels list conservatively; malformed output is not absence."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    lowered = "\n".join(lines).lower()
    if any(marker in lowered for marker in ("no kernels found", "no results found", "no results", "not found")):
        return False
    for index, line in enumerate(lines):
        columns = line.lower().split()
        if not columns or columns[0].strip("|:") not in {"ref", "reference"}:
            continue
        for row in lines[index + 1:]:
            first = row.split()[0].strip("|:")
            if re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", first) and first == kernel_id:
                return True
        return False
    raise KaggleError("Kaggle kernel list returned an unrecognized response; submission was not attempted.")


def _load_run_settings(path: Path | None) -> dict:
    if path is None:
        return {"run_id": "kaggle-run", "train_config": {"device": "auto"}}
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KaggleError("Run config must be readable JSON.") from exc
    if not isinstance(settings, dict) or set(settings) - {"run_id", "train_config", "throughput_benchmark", "g008_principal_benchmark"}:
        raise KaggleError("Run config contains unsupported fields.")
    run_id = settings.get("run_id", "kaggle-run")
    config = settings.get("train_config", {})
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", run_id):
        raise KaggleError("Run config run_id must match the artifact ID contract.")
    allowed_config = {field.name for field in fields(__import__("training.ppo.train", fromlist=["TrainConfig"]).TrainConfig)}
    if not isinstance(config, dict) or set(config) - allowed_config:
        raise KaggleError("Run config train_config contains unknown fields or is not an object.")
    settings_out = {"run_id": run_id, "train_config": config}
    if "g008_principal_benchmark" in settings:
        benchmark = settings["g008_principal_benchmark"]
        if not isinstance(benchmark, dict) or set(benchmark) != {"run_ids"}:
            raise KaggleError("g008_principal_benchmark must contain exactly run_ids.")
        try:
            from training.ppo.g008_benchmark import prepare_benchmark_schedules
            prepare_benchmark_schedules(benchmark["run_ids"])
        except (TypeError, ValueError) as exc:
            raise KaggleError(f"Invalid preregistered G008 benchmark schedule: {exc}") from exc
        if "throughput_benchmark" in settings:
            raise KaggleError("G008 principal benchmark cannot be combined with throughput_benchmark.")
        settings_out["g008_principal_benchmark"] = benchmark
    if "throughput_benchmark" in settings:
        benchmark = settings["throughput_benchmark"]
        if not isinstance(benchmark, dict) or set(benchmark) != {"configs", "updates"}:
            raise KaggleError("throughput_benchmark must contain exactly configs and updates.")
        from training.ppo.benchmark import MAX_BENCHMARK_UPDATES, build_candidates
        from training.ppo.train import TrainConfig
        configs = benchmark["configs"]
        updates = benchmark["updates"]
        if type(updates) is not int or not 1 <= updates <= MAX_BENCHMARK_UPDATES:
            raise KaggleError(f"throughput_benchmark updates must be an integer in [1,{MAX_BENCHMARK_UPDATES}].")
        try:
            # Validate candidate dimensions without requiring the local machine
            # to have CUDA; the generated Kaggle runner enforces CUDA at runtime.
            base = replace(TrainConfig(**config), device="cpu")
            candidates = [replace(candidate, total_updates=updates)
                          for candidate in build_candidates(configs, base)]
            for candidate in candidates:
                candidate.validate()
        except (TypeError, ValueError) as exc:
            raise KaggleError(f"Invalid bounded throughput_benchmark configuration: {exc}") from exc
        settings_out["throughput_benchmark"] = {"configs": configs, "updates": updates}
    return settings_out


def prepare(kernel_id: str, source_revision: str, output_dir: Path | None = None,
            run_config: Path | None = None) -> Path:
    """Build a deterministic, local Kaggle kernel package without contacting Kaggle."""
    kernel_id = _kernel_id(kernel_id)
    if not source_revision or "\n" in source_revision or "\r" in source_revision:
        raise KaggleError("Source revision must be an explicit non-empty single-line value.")
    repo = Path(__file__).resolve().parents[2]
    destination = (output_dir or repo / "training" / "kaggle").resolve()
    if destination.exists():
        raise KaggleError("Prepare destination already exists; refusing to overwrite it.")
    settings = _load_run_settings(run_config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.prepare-", dir=destination.parent))
    try:
        shutil.copytree(repo / "training", stage / "training",
                        ignore=shutil.ignore_patterns("kaggle", "__pycache__", ".pytest_cache",
                                                      ".kaggle.prepare-*"))
        shutil.copytree(repo / "schemas", stage / "schemas",
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        for relative in ("training/README.md", "training/requirements-kaggle.txt"):
            shutil.copy2(repo / relative, stage / relative)
        (stage / "run-config.json").write_text(
            json.dumps(settings, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (stage / ".arena-source-revision").write_text(source_revision + "\n", encoding="utf-8")
        metadata = {
            "id": kernel_id,
            "title": kernel_id.split("/", 1)[1],
            "code_file": "run_kaggle.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            # Request the Kaggle dual-T4 shape for the G008 benchmark. Ordinary
            # kernels keep the default accelerator shape; G008 uses one CUDA
            # device and does not enable DDP.
            "machine_shape": "NvidiaTeslaT4" if "g008_principal_benchmark" in settings else "",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (stage / "kernel-metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        entrypoint = '''"""Kaggle script entrypoint; keeps Kaggle's preinstalled CUDA PyTorch intact."""
from __future__ import annotations

import importlib.util
import base64
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


SOURCE_ARCHIVE_B64 = "__SOURCE_ARCHIVE_B64__"
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


def main():
    # Import/check PyTorch before any optional dependency installation. Kaggle
    # training is deliberately GPU-only; local training retains CPU support.
    import torch
    settings = json.loads((ROOT / "run-config.json").read_text(encoding="utf-8"))
    requested_device = settings["train_config"].get("device", "auto")
    if requested_device == "cpu":
        raise RuntimeError("Kaggle self-play training requires CUDA; device=cpu is not allowed.")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("Kaggle self-play training requires an available CUDA GPU; refusing CPU fallback.")
    device_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    print(f"Preinstalled PyTorch={torch.__version__} CUDA={torch.version.cuda} available={cuda_available} "
          f"visible_cuda_devices={device_names}; training_device={device_names[0] if device_names else 'none'}")
    settings["train_config"]["device"] = "cuda"
    from training.ppo.train import TrainConfig, train
    run_id = settings["run_id"]
    output_root = Path(os.environ.get("ARENA_OUTPUT_DIR", "/kaggle/working/arena-output"))
    output = output_root / run_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Kaggle output: {output}")
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
            "benchmark_report_sha256": file_sha256(report_path),
            "benchmark_report": report,
        }
        (output / "kaggle-execution-evidence.json").write_text(
            json.dumps(evidence, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
        print(f"Completed G008 principal benchmark {run_id}; artifacts: {output}")
        return
    ensure_support_dependencies(torch)
    config = TrainConfig(**settings["train_config"])
    if config.device != "cuda":
        raise RuntimeError("Kaggle training device must resolve to CUDA.")
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
        report_path.write_text(json.dumps(benchmark_report, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
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
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
    print(f"Completed run {run_id}; artifacts: {output}")


if __name__ == "__main__":
    main()
'''
        # Kaggle script kernels execute the selected code file under /kaggle/src;
        # sibling package files are not guaranteed to be mounted beside it.
        # Embed the exact run config, trainer, schemas and support requirements
        # so the generated entrypoint is genuinely self-contained.
        source_buffer = io.BytesIO()
        with zipfile.ZipFile(source_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in ("run-config.json", "training", "schemas"):
                source_path = stage / relative
                if source_path.is_file():
                    archive.write(source_path, relative)
                else:
                    for child in sorted(source_path.rglob("*")):
                        if child.is_file():
                            archive.write(child, child.relative_to(stage).as_posix())
        embedded_source = base64.b64encode(source_buffer.getvalue()).decode("ascii")
        entrypoint = entrypoint.replace("__SOURCE_ARCHIVE_B64__", embedded_source)
        (stage / "run_kaggle.py").write_text(entrypoint, encoding="utf-8")
        # Atomic publication and fail-closed race behavior for the package directory.
        if destination.exists():
            raise KaggleError("Prepare destination appeared while staging; refusing to overwrite it.")
        stage.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def submit(kernel_id: str, source_revision: str, source_dir: Path, *, confirm: bool,
           dry_run: bool = False, run: Callable[[Sequence[str]], str] = _run) -> str:
    kernel_id = _kernel_id(kernel_id)
    _metadata(source_dir, kernel_id, source_revision)
    command = ["kaggle", "kernels", "push", "-p", str(source_dir.resolve())]
    if dry_run:
        return f"DRY RUN: kaggle kernels push -p <selected-source-dir> (kernel={kernel_id}, revision={source_revision})"
    if not confirm:
        raise KaggleError("Submission requires --confirm. No Kaggle command was run.")
    # Resolve whether this is a new kernel before querying status: Kaggle reports
    # an error for status on an ID that has not yet been created. Restrict the
    # search to its exact owner and slug, and fail closed on unparseable output.
    owner, slug = kernel_id.split("/", 1)
    listed = run(["kaggle", "kernels", "list", "--user", owner, "--search", slug, "--page-size", "100"])
    if _kernel_list_contains(listed, kernel_id):
        observed = run(["kaggle", "kernels", "status", kernel_id])
        if _status_state(observed) == "running":
            raise KaggleError("Selected kernel is already running; submission was not attempted.")
    return run(command)


def _schema_validate(doc: dict, schema_file: str) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise KaggleError("Artifact validation requires jsonschema; install training/requirements-kaggle.txt.") from exc
    schema = json.loads((Path(__file__).resolve().parents[2] / "schemas" / schema_file).read_text(encoding="utf-8"))
    try:
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(doc)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "document"
        raise KaggleError(f"Artifact schema validation failed at {location}.") from exc


def validate_run_artifacts(root: Path, expected_run_id: str) -> None:
    """Validate manifest/schema/file hashes and linked evaluation/replays."""
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KaggleError("Downloaded output is missing a valid manifest.json.") from exc
    _schema_validate(manifest, "lc-run-1.schema.json")
    if manifest["run_id"] != expected_run_id:
        raise KaggleError("Downloaded run_id does not match the explicitly requested run ID.")
    if digest_json(manifest, "manifest_sha256") != manifest["manifest_sha256"]:
        raise KaggleError("Run manifest digest is invalid.")
    checkpoint_groups: dict[tuple[str, int], list[str]] = {}
    for checkpoint in manifest["checkpoints"]:
        group = (checkpoint["checkpoint_id"], checkpoint["step"])
        checkpoint_groups.setdefault(group, []).append(checkpoint["format"])
    for (checkpoint_id, step_number), formats in checkpoint_groups.items():
        if len(formats) != 2 or set(formats) != {"pytorch", "onnx"}:
            raise KaggleError(
                f"Checkpoint {checkpoint_id} at step {step_number} must have exactly one PyTorch and one ONNX reference."
            )
    listed: dict[str, str] = {}
    for entry in manifest["files"]:
        rel = safe_relative_path(entry["path"])
        if rel in listed:
            raise KaggleError(f"Run manifest lists an artifact more than once: {rel}")
        path = root / rel
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve() and root.resolve() not in path.resolve().parents:
            raise KaggleError(f"Run artifact is missing or unsafe: {rel}")
        if file_sha256(path) != entry["sha256"]:
            raise KaggleError(f"Run artifact SHA-256 mismatch: {rel}")
        listed[rel] = entry["sha256"]
    for checkpoint in manifest["checkpoints"]:
        checkpoint_rel = safe_relative_path(checkpoint["file"])
        if listed.get(checkpoint_rel) != checkpoint["sha256"]:
            raise KaggleError(f"Checkpoint is missing from the run file hashes or has a mismatched digest: {checkpoint_rel}")
        # The viewer's checkpoint contract is one-file ONNX. Older PyTorch
        # exports may put tensor weights in a sibling `<model>.onnx.data` file;
        # reject that unsupported format even if a manifest hashes the sidecar.
        if checkpoint["format"] == "onnx":
            sidecar_rel = f"{checkpoint_rel}.data"
            sidecar_path = root / sidecar_rel
            if sidecar_path.exists():
                raise KaggleError(f"Viewer requires single-file ONNX; external-data sidecar is unsupported: {sidecar_rel}")
        metadata_rel = str(Path(checkpoint_rel).parent / f"{checkpoint['checkpoint_id']}.metadata.json")
        if metadata_rel not in listed:
            raise KaggleError(f"Checkpoint metadata is missing from the run file hashes: {metadata_rel}")
        metadata_path = root / metadata_rel
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KaggleError(f"Checkpoint metadata is missing or invalid JSON: {metadata_rel}") from exc
        _schema_validate(metadata, "checkpoint-metadata-1.schema.json")
        if digest_json(metadata, "metadata_sha256") != metadata["metadata_sha256"]:
            raise KaggleError(f"Checkpoint metadata digest is invalid: {metadata_rel}")
        if metadata["checkpoint_id"] != checkpoint["checkpoint_id"] or metadata["step"] != checkpoint["step"]:
            raise KaggleError(f"Checkpoint metadata identity does not match manifest: {checkpoint_rel}")
        paired = metadata["onnx"] if checkpoint["format"] == "onnx" else metadata["pytorch"]
        if paired["file"] != checkpoint_rel or paired["sha256"] != checkpoint["sha256"]:
            raise KaggleError(f"Checkpoint metadata file/hash does not match manifest: {checkpoint_rel}")
        for artifact in metadata["artifacts"]:
            rel = safe_relative_path(artifact["file"])
            if listed.get(rel) != artifact["sha256"]:
                raise KaggleError(f"Paired checkpoint artifact is missing or has a mismatched digest: {rel}")
    # Validate each paired metadata file once even when both native and ONNX refs
    # point to it.
    evaluation_ref = manifest["evaluation"]
    metrics_rel = safe_relative_path(evaluation_ref["metrics_file"])
    metrics_path = root / metrics_rel
    if metrics_rel not in listed:
        raise KaggleError("Evaluation report is not protected by the run manifest file hashes.")
    try:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KaggleError("Evaluation report is missing or invalid JSON.") from exc
    _schema_validate(report, "lc-eval-1.schema.json")
    if report["run_id"] != expected_run_id:
        raise KaggleError("Evaluation report run_id does not match the requested run ID.")
    if digest_json(report, "report_sha256") != report["report_sha256"]:
        raise KaggleError("Evaluation report digest is invalid.")
    for rel in evaluation_ref["replay_files"]:
        rel = safe_relative_path(rel)
        path = root / rel
        if rel not in listed or path.is_symlink() or not path.is_file():
            raise KaggleError(f"Evaluation replay is missing, unsafe, or unhashed: {rel}")
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KaggleError(f"Evaluation replay is invalid JSON: {rel}") from exc
        _schema_validate(replay, "lc-replay-1.schema.json")
        if replay["run_id"] != expected_run_id or digest_json(replay, "replay_sha256") != replay["replay_sha256"]:
            raise KaggleError(f"Evaluation replay identity or digest is invalid: {rel}")
        _validate_replay(replay)


def validate_principal_benchmark_artifacts(root: Path, expected_run_id: str) -> None:
    """Validate compact G008 benchmark outputs (which intentionally have no training manifest)."""
    try:
        report_path = root / "g008-principal-benchmark.json"
        evidence_path = root / "kaggle-execution-evidence.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KaggleError("Downloaded G008 benchmark is missing a valid report or execution evidence.") from exc
    if (not isinstance(report, dict) or report.get("format") != "circuit-ai-g008-principal-benchmark/1"
            or not isinstance(evidence, dict)
            or evidence.get("format") != "circuit-ai-kaggle-g008-benchmark-evidence/1"
            or evidence.get("run_id") != expected_run_id or evidence.get("benchmark_report") != report
            or evidence.get("benchmark_report_sha256") != file_sha256(report_path)):
        raise KaggleError("G008 benchmark report/evidence identity or digest is invalid.")
    device = report.get("device")
    repeats = report.get("repeats")
    if (not isinstance(device, dict) or device.get("device") != "cuda"
            or device.get("cuda_available") is not True
            or not isinstance(device.get("cuda_name"), str) or not device["cuda_name"].strip()
            or not isinstance(repeats, list) or len(repeats) != 5
            or report.get("measured_repeats") != 5):
        raise KaggleError("G008 benchmark lacks the frozen five-repeat CUDA evidence.")
    for index, repeat in enumerate(repeats):
        metrics_path = root / "measured" / f"repeat-{index:02d}" / "training-metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KaggleError(f"G008 measured repeat {index} is missing valid training metrics.") from exc
        run_device = metrics.get("device", {})
        updates = repeat.get("optimizer_updates")
        if (repeat.get("repeat") != index or repeat.get("training_metrics_sha256") != file_sha256(metrics_path)
                or run_device.get("device") != "cuda" or run_device.get("cuda_available") is not True
                or run_device.get("cuda_optimizer_updates") != run_device.get("optimizer_updates")
                or run_device.get("optimizer_updates") != updates
                or repeat.get("cuda_optimizer_updates") != updates):
            raise KaggleError(f"G008 measured repeat {index} failed CUDA metrics/hash validation.")
    for index in range(2):
        metrics_path = root / "warmups" / f"warmup-{index:02d}" / "training-metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KaggleError(f"G008 warmup {index} is missing valid training metrics.") from exc
        run_device = metrics.get("device", {})
        if (run_device.get("device") != "cuda" or run_device.get("cuda_available") is not True
                or run_device.get("cuda_optimizer_updates") != run_device.get("optimizer_updates")):
            raise KaggleError(f"G008 warmup {index} failed CUDA metrics validation.")


def _wire_state(state: State) -> dict:
    return {"width": state.width, "height": state.height,
            "positions": [{"x": a.position[0], "y": a.position[1]} for a in state.agents],
            "headings": [ACTION_ORDER[a.heading] for a in state.agents],
            "trails": [[{"x": x, "y": y} for x, y in agent.trail] for agent in state.agents]}


def _validate_replay(replay: dict) -> None:
    """Rebuild deterministic initial state and replay every simultaneous transition."""
    try:
        seed, arena_index = int(replay["master_seed"]), replay["arena_index"]
        initial, derived = initialize_episode(replay["initial_state"]["width"],
                                              replay["initial_state"]["height"], seed, arena_index)
        expected_streams = [{"label": item.label, "initstate": str(item.initstate), "initseq": str(item.initseq)}
                            for item in derived]
        if replay["derived_streams"] != expected_streams or replay["initial_state"] != _wire_state(initial):
            raise KaggleError("Replay initial state/derived streams do not match its deterministic seed.")
        current = initial
        for index, frame in enumerate(replay["ticks"]):
            if current.result is not None or frame["tick"] != index:
                raise KaggleError("Replay contains ticks after terminal state or a non-sequential tick index.")
            actions = [None if action == "INVALID" else action for action in frame["joint_actions"]]
            transition = step(current, actions)
            expected_causes = [list(causes) for causes in transition.causes]
            if frame["collision_causes"] != expected_causes:
                raise KaggleError(f"Replay collision causes differ from the engine at tick {index}.")
            terminal_causes = ([f"agent_{i}_dead" for i, agent in enumerate(transition.state.agents) if not agent.alive])
            if frame["terminal_causes"] != terminal_causes:
                raise KaggleError(f"Replay terminal causes differ from the engine at tick {index}.")
            current = transition.state
        if replay["tick_count"] != len(replay["ticks"]):
            raise KaggleError("Replay tick_count does not match its transitions.")
        if current.result is None or replay["result"] != current.result:
            raise KaggleError("Replay does not reach its declared terminal result.")
    except KaggleError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise KaggleError("Replay cannot be reconstructed by the deterministic engine.") from exc


def download(kernel_id: str, version: int, run_id: str, output_dir: Path, *,
             max_attempts: int = 30, interval_seconds: float = 10,
             run: Callable[[Sequence[str]], str] = _run,
             sleep: Callable[[float], None] = time.sleep) -> Path:
    kernel_id = _kernel_id(kernel_id)
    version = _version(str(version))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", run_id or ""):
        raise KaggleError("Run ID must be explicit and match the artifact ID contract.")
    state, status = poll_status(kernel_id, version, max_attempts=max_attempts,
                                interval_seconds=interval_seconds, run=run, sleep=sleep)
    if state == "timeout":
        raise KaggleError(f"Kernel is still running after bounded polling for version {version}; no download was attempted.")
    if state != "success":
        raise KaggleError(f"Kernel did not succeed for selected version {version}; no download was attempted.")
    destination = output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise KaggleError("Final artifact directory already exists; refusing to overwrite it.")
    staging_parent = destination.parent
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=staging_parent))
    try:
        run(["kaggle", "kernels", "output", f"{kernel_id}/{version}", "-p", str(stage)])
        # Kaggle script kernels preserve the notebook's `/kaggle/working`
        # directory hierarchy in downloaded outputs (for example,
        # `arena-output/<run-id>/manifest.json`). Promote only the exact run
        # subtree, not Kaggle's wrapper files or sibling runs.
        artifact_root = stage
        if not (artifact_root / "manifest.json").is_file():
            matches = []
            for manifest_path in stage.rglob("manifest.json"):
                try:
                    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(candidate, dict) and candidate.get("run_id") == run_id:
                    matches.append(manifest_path.parent)
            if matches:
                if len(matches) != 1:
                    raise KaggleError("Downloaded output must contain exactly one manifest for the requested run.")
                artifact_root = matches[0]
                validate_run_artifacts(artifact_root, run_id)
            else:
                benchmark_matches = []
                for evidence_path in stage.rglob("kaggle-execution-evidence.json"):
                    try:
                        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if (isinstance(evidence, dict) and evidence.get("format") ==
                            "circuit-ai-kaggle-g008-benchmark-evidence/1"
                            and evidence.get("run_id") == run_id):
                        benchmark_matches.append(evidence_path.parent)
                if len(benchmark_matches) != 1:
                    raise KaggleError("Downloaded output must contain exactly one validated run or G008 benchmark.")
                artifact_root = benchmark_matches[0]
                validate_principal_benchmark_artifacts(artifact_root, run_id)
        else:
            validate_run_artifacts(artifact_root, run_id)
        # Atomic promotion on the same filesystem; the destination was checked
        # before download and is checked again to fail closed on a race.
        if destination.exists():
            raise KaggleError("Final artifact directory appeared during download; refusing to overwrite it.")
        if artifact_root == stage:
            stage.rename(destination)
        else:
            artifact_root.rename(destination)
            shutil.rmtree(stage, ignore_errors=True)
        return destination
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arena", description="Explicit Kaggle lifecycle for Light Cycles.")
    group = parser.add_subparsers(dest="group", required=True)
    kaggle = group.add_parser("kaggle", help="explicit Kaggle submit/status/download")
    commands = kaggle.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="build a deterministic Kaggle kernel package locally")
    prep.add_argument("--kernel-id", required=True)
    prep.add_argument("--source-revision", required=True)
    prep.add_argument("--output-dir", type=Path)
    prep.add_argument("--run-config", type=Path)
    sub = commands.add_parser("submit", help="submit one selected kernel revision")
    sub.add_argument("--kernel-id", required=True)
    sub.add_argument("--source-revision", required=True)
    sub.add_argument("--source-dir", type=Path, required=True)
    sub.add_argument("--confirm", action="store_true", help="explicitly authorize remote submission")
    sub.add_argument("--dry-run", action="store_true", help="validate and print a redacted plan without Kaggle access")
    stat = commands.add_parser("status", help="poll a selected kernel/version")
    stat.add_argument("--kernel-id", required=True)
    stat.add_argument("--version", required=True)
    stat.add_argument("--max-attempts", type=int, default=30)
    stat.add_argument("--interval-seconds", type=float, default=10)
    down = commands.add_parser("download", help="download and validate a selected successful run")
    down.add_argument("--kernel-id", required=True)
    down.add_argument("--version", required=True)
    down.add_argument("--run-id", required=True)
    down.add_argument("--output-dir", type=Path, required=True)
    down.add_argument("--max-attempts", type=int, default=30)
    down.add_argument("--interval-seconds", type=float, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            path = prepare(args.kernel_id, args.source_revision, args.output_dir, args.run_config)
            print(f"Prepared Kaggle kernel source locally: {path}")
        elif args.command == "submit":
            print(submit(args.kernel_id, args.source_revision, args.source_dir,
                         confirm=args.confirm, dry_run=args.dry_run))
        elif args.command == "status":
            state, _ = poll_status(_kernel_id(args.kernel_id), _version(args.version),
                                   max_attempts=args.max_attempts,
                                   interval_seconds=args.interval_seconds)
            print(f"version={args.version} state={state}")
            if state == "timeout":
                print("Polling deadline reached; use status again explicitly to continue.", file=sys.stderr)
                return 2
            if state == "error":
                print("Kaggle reported a terminal failure; inspect logs with `kaggle kernels logs owner/slug/version`.", file=sys.stderr)
            return 1 if state == "error" else 0
        else:
            path = download(args.kernel_id, _version(args.version), args.run_id, args.output_dir,
                            max_attempts=args.max_attempts, interval_seconds=args.interval_seconds)
            print(f"Validated run {args.run_id} from {args.kernel_id} version {args.version}: {path}")
    except (KaggleError, ValueError, OSError) as exc:
        print(f"arena kaggle: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
