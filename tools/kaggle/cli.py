from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
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


def package_tree_sha256(source: Path) -> str:
    """Hash a package tree by normalized relative paths and file-byte hashes; reject symlinks."""
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise KaggleError("Frozen package must be a real directory, not a symlink.")
    entries = []
    for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        if path.is_symlink():
            raise KaggleError("Frozen package trees cannot contain symlinks.")
        if path.is_file():
            entries.append({"path": path.relative_to(source).as_posix(), "sha256": file_sha256(path)})
        elif not path.is_dir():
            raise KaggleError("Frozen package tree contains a non-regular filesystem entry.")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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


def register_recovery_workloads(registry, amendment: dict, workloads: Sequence[dict]) -> list[dict]:
    """Append the reviewed recovery amendment and its two pre-frozen registrations."""
    if not isinstance(workloads, Sequence) or isinstance(workloads, (str, bytes)) or len(workloads) != 2:
        raise KaggleError("Recovery registration requires exactly the approved benchmark and training workloads.")
    named = amendment.get("recovery_registrations") if isinstance(amendment, dict) else None
    if not isinstance(named, list) or len(named) != 2:
        raise KaggleError("Recovery amendment must name exactly two workloads before registry mutation.")
    workload_map = {item.get("registration_id"): item for item in workloads if isinstance(item, dict)}
    expected_map = {item.get("registration_id"): item for item in named if isinstance(item, dict)}
    if len(workload_map) != 2 or workload_map.keys() != expected_map.keys():
        raise KaggleError("Recovery workload records do not match the amendment's exact registration IDs.")
    for registration_id, expected in expected_map.items():
        record = workload_map[registration_id]
        kind = record.get("registration_kind")
        details = record.get("record")
        cap = 3600 if kind == "benchmark" else 7200 if kind == "train" else 0
        timeout = details.get("session_timeout_seconds") if isinstance(details, dict) else None
        if (kind != expected.get("registration_kind") or not cap or type(timeout) is not int
                or not 0 < timeout <= cap):
            raise KaggleError("Recovery registrations need the exact amendment kind and a positive bounded timeout.")
    try:
        amendment_event = registry.register_recovery_amendment(amendment)
        events = []
        for item in workloads:
            events.append(registry.register_recovery_workload(
                item["registration_id"], item["registration_kind"], item["record"],
            ))
        return [amendment_event, *events]
    except Exception as exc:
        raise KaggleError(f"Recovery registry append failed; inspect the append-only head before continuing: {exc}") from exc


def register_recovery_benchmark_artifact(registry, attempt_event: dict,
                                         artifact_root: Path, envelope_path: Path) -> dict:
    """Write and register an immutable post-download envelope for a successful /3 attempt."""
    if (not isinstance(attempt_event, dict) or attempt_event.get("event_type") != "workload_attempt"
            or attempt_event.get("event_data", {}).get("outcome") != "terminal"
            or attempt_event.get("event_data", {}).get("terminal_status") != "COMPLETE"):
        raise KaggleError("Recovery benchmark artifact requires a successful terminal attempt event.")
    data = attempt_event["event_data"]
    artifact_root, envelope_path = Path(artifact_root), Path(envelope_path)
    report_path = artifact_root / "g008-principal-benchmark.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_hash = file_sha256(report_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise KaggleError("Recovery benchmark report or artifact is missing/invalid.") from exc
    run_id = None
    try:
        evidence = json.loads((artifact_root / "kaggle-execution-evidence.json").read_text(encoding="utf-8"))
        run_id = evidence.get("run_id") if isinstance(evidence, dict) else None
        validate_principal_benchmark_artifacts(artifact_root, run_id)
    except (OSError, json.JSONDecodeError, KaggleError) as exc:
        raise KaggleError("Downloaded recovery benchmark does not pass full /3 artifact validation.") from exc
    selected = report.get("selected_variant_id") if isinstance(report, dict) else None
    implementation = report.get("implementations", {}).get(selected, {}) if isinstance(report, dict) else {}
    if (not isinstance(report, dict) or report.get("format") != "circuit-ai-g008-principal-benchmark/3"
            or report.get("source_sha256") != data.get("source_sha256")
            or report.get("config_sha256") != data.get("config_sha256")
            or report.get("seed_plan_sha256") != data.get("seed_plan_sha256")
            or selected not in {"incumbent", "candidate"}
            or not implementation):
        raise KaggleError("Recovery benchmark report does not bind its successful terminal attempt.")
    events = registry.events()
    if (attempt_event.get("event_index", -1) >= len(events)
            or events[attempt_event["event_index"]].get("event_hash") != attempt_event.get("event_hash")):
        raise KaggleError("Recovery benchmark attempt is not the exact event in the current registry chain.")
    raw_metrics = {}
    for pair in report["pairs"]:
        for role, arm in pair["arms"].items():
            raw_metrics[arm["training_metrics_file"]] = arm["training_metrics_sha256"]
    for item in report["warmups"]["runs"]:
        raw_metrics[item["training_metrics_file"]] = item["training_metrics_sha256"]
    selection_evidence = _recovery_selection_evidence(report)
    envelope = {
        "format": "circuit-ai-g008-recovery-benchmark-artifact/1",
        "attempt": {"attempt_id": data["attempt_id"], "event_index": attempt_event["event_index"],
                    "event_hash": attempt_event["event_hash"], "assigned_version": data["assigned_version"],
                    "terminal_status": data["terminal_status"], "terminal_metadata": data["terminal_metadata"],
                    "status_evidence": data["status_evidence"]},
        "hashes": {key: data[key] for key in ("package_sha256", "source_sha256", "trainer_sha256",
                                               "backend_sha256", "engine_sha256", "config_sha256",
                                               "seed_plan_sha256")},
        "benchmark_report": report,
        "benchmark_report_sha256": report_hash,
        "raw_metrics_sha256": dict(sorted(raw_metrics.items())),
        "selection_evidence": selection_evidence,
    }
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    if envelope_path.exists():
        raise KaggleError("Recovery artifact envelope destination already exists; refusing to overwrite.")
    envelope_path.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
                             encoding="utf-8")
    artifact_hash = file_sha256(envelope_path)
    return registry.register_benchmark_artifact(
        artifact_hash, data["registration_event_index"], data["registration_event_hash"],
        attempt_event_index=attempt_event["event_index"], attempt_event_hash=attempt_event["event_hash"],
        report_sha256=report_hash, selection_evidence=selection_evidence,
    )


def _recovery_selection_evidence(report: dict) -> dict:
    """Project the paired report's decision into the registry's exact allowed shape."""
    return {
        "parity": {"passed": report["parity"]["passed"]},
        "pairs": [{"candidate_speedup_ratio": pair["candidate_speedup_ratio"]} for pair in report["pairs"]],
        "summary": {"candidate_speedup_ratio": {"median": report["summary"]["candidate_speedup_ratio"]["median"]}},
        "selection": report["selection"],
        "selected_variant_id": report["selected_variant_id"],
    }


@contextmanager
def _recovery_kernel_lock(registry_path: Path, kernel_id: str):
    lock_tag = hashlib.sha256(kernel_id.encode("utf-8")).hexdigest()[:16]
    lock_path = Path(registry_path).parent / f".g008-kaggle-{lock_tag}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise KaggleError("Another project process holds the exclusive lock for this Kaggle kernel.") from exc
        lock_id = str(uuid.uuid4())
        try:
            yield lock_id
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sdk_metadata(api, kernel_id: str) -> dict:
    """Call the pinned SDK get_kernel request for exactly owner and slug."""
    if hasattr(api, "get_kernel"):
        response = api.get_kernel(kernel_id)
    else:
        try:
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
            owner, slug = kernel_id.split("/", 1)
            request = ApiGetKernelRequest()
            request.user_name = owner
            request.kernel_slug = slug
            response = api.kernels.kernels_api_client.get_kernel(request)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            raise KaggleError("Pinned Kaggle SDK get_kernel metadata interface is unavailable.") from exc
    metadata = getattr(response, "metadata", response)
    if isinstance(metadata, dict):
        values = metadata
        observed_id = values.get("kernel_id", values.get("ref"))
        if observed_id is None and values.get("author") and values.get("slug"):
            observed_id = f"{values['author']}/{values['slug']}"
        current = values.get("current_version_number")
    else:
        observed_id = getattr(metadata, "ref", None)
        if not observed_id and getattr(metadata, "author", None) and getattr(metadata, "slug", None):
            observed_id = f"{metadata.author}/{metadata.slug}"
        current = getattr(metadata, "current_version_number", None)
    if observed_id != kernel_id or type(current) is not int or current <= 0:
        raise KaggleError("Kaggle exact-kernel metadata is missing identity or a positive current version.")
    return {"kernel_id": kernel_id, "current_version_number": current}


def _sdk_status(api, kernel_id: str) -> tuple[str, dict]:
    """Read latest status for an exact kernel; this deliberately sends no version selector."""
    try:
        response = api.kernels_status(kernel_id)
    except AttributeError as exc:
        raise KaggleError("Pinned Kaggle SDK latest-status interface is unavailable.") from exc
    raw_status = response.get("status") if isinstance(response, dict) else getattr(response, "status", response)
    status = getattr(raw_status, "name", raw_status)
    if not isinstance(status, str):
        raise KaggleError("Kaggle returned an unknown status representation; attempt remains pending.")
    normalized = status.upper().replace("-", "_").replace(" ", "_")
    if normalized in {"COMPLETE", "COMPLETED", "SUCCESS", "SUCCEEDED"}:
        return "COMPLETE", {"status": "COMPLETE"}
    if normalized in {"ERROR", "FAILED"}:
        return "ERROR", {"status": "ERROR"}
    if normalized in {"CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCELED"}:
        return "CANCELLED", {"status": "CANCELLED"}
    if normalized == "TIMEOUT":
        return "TIMEOUT", {"status": "TIMEOUT"}
    if normalized in {"QUEUED", "RUNNING", "CANCEL_REQUESTED", "NEW_SCRIPT", "PENDING", "STARTING"}:
        return "RUNNING", {"status": normalized}
    raise KaggleError("Kaggle returned an unrecognized latest status; attempt remains pending.")


def _quota_seconds(api) -> tuple[int, int]:
    try:
        response = api.quota_view()
        quota = getattr(response, "gpu_quota", None)
        if quota is None and isinstance(response, dict):
            quota = response.get("gpu_quota", response)
        if isinstance(quota, dict):
            remaining = quota.get("remaining_gpu_seconds")
            used = quota.get("used_gpu_seconds")
        else:
            total = getattr(quota, "total_time_allowed", None)
            consumed = getattr(quota, "time_used", None)
            remaining = (total - consumed).total_seconds() if total is not None and consumed is not None else None
            used = consumed.total_seconds() if consumed is not None else None
        if (isinstance(remaining, bool) or not isinstance(remaining, (int, float))
                or not math.isfinite(remaining) or remaining < 0
                or isinstance(used, bool) or not isinstance(used, (int, float))
                or not math.isfinite(used) or used < 0):
            raise ValueError("missing or invalid GPU quota seconds")
        return math.floor(remaining), math.ceil(used)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise KaggleError("Kaggle GPU quota response is missing usable GPU seconds.") from exc


def _recovery_record(registration_event: dict, variant_id: str) -> tuple[dict, dict]:
    if registration_event.get("event_type") != "registration":
        raise KaggleError("Recovery workload must be an exact prior registration event.")
    data = registration_event.get("event_data", {})
    record = data.get("record")
    kind = data.get("registration_kind")
    if not isinstance(record, dict) or kind not in {"benchmark", "train"}:
        raise KaggleError("Recovery registration must bind a benchmark or training workload.")
    maximum = 3600 if kind == "benchmark" else 7200
    timeout = record.get("session_timeout_seconds")
    if type(timeout) is not int or not 0 < timeout <= maximum:
        raise KaggleError("Registered Kaggle timeout must be a positive integer within the workload cap.")
    if record.get("accelerator_count") != 1 or record.get("machine_shape") != "NvidiaTeslaT4":
        raise KaggleError("Registered recovery workload must use exactly one Tesla T4.")
    variants = record.get("implementation_variants" if kind == "benchmark" else "allowed_implementation_variants")
    expected_variant_id = "paired" if kind == "benchmark" else variant_id
    variant = next((item for item in variants or [] if item.get("variant_id") == expected_variant_id), None)
    if kind == "train" and expected_variant_id not in {"incumbent", "candidate"}:
        raise KaggleError("Training workload requires a pre-registered incumbent or candidate variant.")
    if kind == "benchmark":
        variant = record
    if not isinstance(variant, dict):
        raise KaggleError("Selected implementation variant is not frozen in the registration.")
    return record, {key: variant[key] for key in (
        "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
        "backend_sha256", "trainer_sha256", "engine_sha256",
    )}


def _validate_recovery_budget(registry, registration_event: dict, timeout: int,
                              quota_remaining: int, *, resuming_attempt_id: str | None = None) -> dict:
    events = registry.events()
    amendment_event = next((event for event in events if event.get("event_type") == "recovery_amendment"), None)
    if amendment_event is None:
        raise KaggleError("Recovery amendment is absent; workload reservation cannot be checked.")
    kind = registration_event["event_data"]["registration_kind"]
    amendment = amendment_event["event_data"]
    reservation = amendment["reservation_metadata"]
    actual = {"benchmark": 0, "train": 0}
    output = 0
    pending_ids = set()
    states = {}
    for event in events:
        data = event.get("event_data", {})
        typ = event.get("event_type")
        attempt_id = data.get("attempt_id")
        if typ == "workload_preflight":
            states[attempt_id] = "preflight"
        elif typ == "workload_push_dispatched":
            states[attempt_id] = "dispatched"
        elif typ == "workload_push_accepted":
            states[attempt_id] = "accepted"
        elif typ == "workload_attempt":
            states[attempt_id] = "terminal"
            registration_index = data["registration_event_index"]
            registered = events[registration_index]["event_data"]
            actual[registered["registration_kind"]] += data["actual_gpu_seconds"]
            output += data["output_bytes"]
    pending_ids = {attempt_id for attempt_id, state in states.items() if state != "terminal"}
    if resuming_attempt_id is not None:
        if states.get(resuming_attempt_id) not in {"preflight", "accepted"}:
            raise KaggleError("Only this exact preflight or accepted attempt may be excluded while revalidating its reservation.")
        pending_ids.remove(resuming_attempt_id)
    if pending_ids:
        raise KaggleError("An unresolved registry tail blocks all new recovery events and pushes.")
    cap = 3600 if kind == "benchmark" else 7200
    workload_used = actual[kind]
    if workload_used + timeout > cap:
        raise KaggleError("Full registered timeout reservation exceeds the remaining workload GPU budget.")
    new_total = reservation["campaign_gpu_seconds_before"] + sum(actual.values()) + timeout
    final_gpu = reservation["reserved_gpu_seconds"]["validation"] + reservation["reserved_gpu_seconds"]["final"]
    if new_total + final_gpu > 21600 or new_total + final_gpu > 28800:
        raise KaggleError("Full timeout reservation exceeds project compute budget or protected final reserve.")
    future_train = 7200 if kind == "benchmark" else 0
    if quota_remaining < timeout + future_train + 7200:
        raise KaggleError("Kaggle GPU quota cannot cover this attempt, remaining training, and final reserve.")
    output_reserved = sum(reservation["reserved_output_bytes"].values())
    if reservation["campaign_output_bytes_before"] + output + output_reserved > 5_000_000_000:
        raise KaggleError("Full registered output reservation exceeds the project output cap.")
    return {"actual": actual, "output_bytes": output, "campaign_gpu_seconds_projected": new_total,
            "quota_gpu_seconds_remaining": quota_remaining, "protected_final_reserve_gpu_seconds": 7200}


def run_recovery_workload(registry, registration_event_index: int, source_dir: Path, *, attempt_id: str,
                          implementation_variant_id: str = "paired", no_external_writer_confirmed: bool,
                          api=None, output_bytes: Callable[[str, int], int] | None = None,
                          terminal_callback: Callable[[dict], None] | None = None,
                          monotonic: Callable[[], float] = time.monotonic,
                          sleep: Callable[[float], None] = time.sleep,
                          poll_interval_seconds: float = 10, reconciliation_slack_seconds: int = 600) -> dict:
    """Run one frozen recovery attempt. Ambiguous tails remain reserved and are never retried here."""
    if no_external_writer_confirmed is not True:
        raise KaggleError("Recovery preflight requires explicit confirmation that no outside writer will update the kernel.")
    if (not isinstance(attempt_id, str) or not attempt_id
            or type(reconciliation_slack_seconds) is not int or not 0 <= reconciliation_slack_seconds <= 600
            or not isinstance(poll_interval_seconds, (int, float)) or isinstance(poll_interval_seconds, bool)
            or not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0):
        raise KaggleError("Recovery attempt ID, poll interval, or reconciliation slack is invalid.")
    events = registry.events()
    if type(registration_event_index) is not int or not 0 <= registration_event_index < len(events):
        raise KaggleError("Recovery registration index is not present in the verified registry.")
    registration = events[registration_event_index]
    reg_data = registration.get("event_data", {})
    record, hashes = _recovery_record(registration, implementation_variant_id)
    kernel_id = _kernel_id(record["kernel_id"])
    timeout = record["session_timeout_seconds"]
    expected_cap = 3600 if reg_data["registration_kind"] == "benchmark" else 7200
    if timeout > expected_cap:
        raise KaggleError("Registered timeout exceeds the recovery workload cap.")
    marker = Path(source_dir) / ".arena-source-revision"
    try:
        source_revision = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise KaggleError("Frozen Kaggle package is missing its source revision marker.") from exc
    _metadata(Path(source_dir), kernel_id, source_revision)
    if api is None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        except (ImportError, AttributeError) as exc:
            raise KaggleError("Pinned Kaggle CLI SDK is not installed/configured.") from exc

    registry_path = getattr(registry, "path", Path.cwd() / ".omx/evidence/g008-experiment-registry.jsonl")
    with _recovery_kernel_lock(Path(registry_path), kernel_id) as lock_id:
        observed_package_sha256 = package_tree_sha256(Path(source_dir))
        if observed_package_sha256 != hashes["package_sha256"]:
            raise KaggleError("Frozen Kaggle package bytes differ from the registered package_sha256.")
        events = registry.events()
        prior_attempt_events = [event for event in events if event.get("event_type") in {
            "workload_preflight", "workload_push_dispatched", "workload_push_accepted", "workload_attempt",
        }]
        existing_attempt = [event for event in prior_attempt_events
                            if event.get("event_data", {}).get("attempt_id") == attempt_id]
        if existing_attempt and existing_attempt[-1]["event_type"] == "workload_attempt":
            raise KaggleError("Attempt ID already has a terminal event; use a fresh unique ID for an approved retry.")
        phase = existing_attempt[-1]["event_type"] if existing_attempt else None
        if phase in {"workload_push_dispatched", "workload_preflight"} and phase == "workload_push_dispatched":
            raise KaggleError("A dispatch-only tail has unknown assignment; it remains pending and cannot be polled or pushed again.")
        if any(event.get("event_type") == "workload_preflight"
               and event.get("event_data", {}).get("attempt_id") != attempt_id
               and not any(item.get("event_type") == "workload_attempt"
                           and item.get("event_data", {}).get("attempt_id") == event["event_data"]["attempt_id"]
                           for item in events) for event in events if event.get("event_type") == "workload_preflight"):
            raise KaggleError("A prior unresolved attempt blocks all new workload events.")
        previous = [event for event in events if event.get("event_type") == "workload_attempt"
                    and event.get("event_data", {}).get("registration_event_index") == registration_event_index]
        if phase != "workload_push_accepted":
            if phase == "workload_preflight":
                if len(existing_attempt) != 1:
                    raise KaggleError("Preflight-only attempt tail is not the exact single registered event.")
                preflight_event = existing_attempt[0]
                preflight_data = preflight_event["event_data"]
                expected_version = preflight_data.get("expected_current_version")
                if (preflight_data.get("registration_event_index") != registration_event_index
                        or preflight_data.get("registration_event_hash") != registration.get("event_hash")
                        or preflight_data.get("kernel_id") != kernel_id
                        or preflight_data.get("implementation_variant_id") != implementation_variant_id
                        or preflight_data.get("session_timeout_seconds") != timeout
                        or preflight_data.get("no_outside_writer_confirmed") is not True
                        or preflight_data.get("reserved_gpu_seconds") != timeout
                        or any(preflight_data.get(key) != value for key, value in hashes.items())
                        or preflight_data.get("lock_head_hash") != (registry.lock_head_hash
                            if reg_data["registration_kind"] == "train" else None)):
                    raise KaggleError("Existing preflight tail does not match its exact registration, package, timeout, or operator evidence.")
            else:
                expected_version = (previous[-1]["event_data"].get("assigned_version")
                                    if previous and previous[-1]["event_data"].get("assigned_version") is not None
                                    else next((event["event_data"]["historical_overrun_evidence"]["current_version_number"]
                                               for event in events if event.get("event_type") == "recovery_amendment"), None))
            if type(expected_version) is not int or expected_version <= 0:
                raise KaggleError("No exact reconciled kernel version is available for recovery preflight.")
            before = _sdk_metadata(api, kernel_id)
            if before["current_version_number"] != expected_version:
                raise KaggleError("Exact Kaggle kernel version differs from the registry's reconciled head.")
            prior_status, prior_evidence = _sdk_status(api, kernel_id)
            after = _sdk_metadata(api, kernel_id)
            if after != before or prior_status == "RUNNING":
                raise KaggleError("Latest Kaggle run is active or the kernel version changed during preflight.")
            if prior_status not in {"COMPLETE", "ERROR", "CANCELLED", "TIMEOUT"}:
                raise KaggleError("Prior latest Kaggle status is not terminal.")
            quota_remaining, quota_used_before = _quota_seconds(api)
            budget = _validate_recovery_budget(
                registry, registration, timeout, quota_remaining,
                resuming_attempt_id=attempt_id if phase == "workload_preflight" else None,
            )
            if phase is None:
                same_registration = [event for event in events if event.get("event_type") == "workload_attempt"
                                     and event.get("event_data", {}).get("registration_event_index") == registration_event_index]
                prior = [{"attempt_id": event["event_data"]["attempt_id"],
                          "terminal_event_index": event["event_index"], "terminal_event_hash": event["event_hash"]}
                         for event in same_registration]
                preflight_data = {
                    "attempt_id": attempt_id, "registration_event_index": registration_event_index,
                    "registration_event_hash": registration["event_hash"], "registration_id": reg_data["registration_id"],
                    "kernel_id": kernel_id, **hashes, "session_timeout_seconds": timeout,
                    "accelerator_count": 1, "reserved_gpu_seconds": timeout,
                    "expected_current_version": expected_version,
                    "quota_evidence": {"remaining_gpu_seconds": quota_remaining,
                                       "used_gpu_seconds": quota_used_before, "source": "pinned Kaggle SDK quota_view"},
                    "policy_evidence": {"paid_services": False, "billing_changes": False,
                                         "machine_shape": "NvidiaTeslaT4", "prior_runs_terminal": True,
                                         "no_queued_or_running": True, "preflight_status": prior_status,
                                         "status_evidence": prior_evidence},
                    "prior_attempts": prior, "implementation_variant_id": implementation_variant_id,
                    "no_outside_writer_confirmed": True, "exclusive_lock_id": lock_id,
                    "reconciliation_slack_seconds": reconciliation_slack_seconds,
                    "lock_head_hash": registry.lock_head_hash if reg_data["registration_kind"] == "train" else None,
                }
                preflight_event = registry.preflight_workload(preflight_data)
            else:
                preflight_event = existing_attempt[0]
                preflight_data = preflight_event["event_data"]
                if (preflight_data.get("registration_event_hash") != registration["event_hash"]
                        or preflight_data.get("expected_current_version") != expected_version
                        or preflight_data.get("session_timeout_seconds") != timeout
                        or preflight_data.get("no_outside_writer_confirmed") is not True):
                    raise KaggleError("Existing preflight tail does not match current exact registration and metadata.")
            dispatch_data = {
                key: preflight_data[key] for key in (
                    "attempt_id", "registration_event_index", "registration_event_hash", "kernel_id",
                    "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                    "backend_sha256", "trainer_sha256", "engine_sha256", "session_timeout_seconds",
                    "accelerator_count", "reserved_gpu_seconds", "implementation_variant_id", "lock_head_hash",
                )
            }
            dispatch_data.update(preflight_event_index=preflight_event["event_index"],
                                 preflight_event_hash=preflight_event["event_hash"],
                                 dispatch_timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                                 request_identity=f"{kernel_id}:{attempt_id}:{timeout}")
            if package_tree_sha256(Path(source_dir)) != hashes["package_sha256"]:
                raise KaggleError("Frozen Kaggle package changed after preflight; no dispatch or SDK push was attempted.")
            dispatch_event = registry.dispatch_workload_push(dispatch_data)
            push_started = monotonic()
            try:
                response = api.kernels_push(str(Path(source_dir).resolve()), timeout=str(timeout), acc="NvidiaTeslaT4")
            except Exception:
                # A generic exception cannot prove that Kaggle did not assign a version.
                raise KaggleError("Kaggle push response is ambiguous; dispatch remains pending and reserved.")
            assigned = getattr(response, "version_number", None)
            if isinstance(response, dict):
                assigned = response.get("version_number")
            if type(assigned) is not int or assigned <= 0:
                response_error = getattr(response, "error", None)
                if isinstance(response, dict):
                    response_error = response.get("error")
                if response_error:
                    rejection_metadata = _sdk_metadata(api, kernel_id)
                    if rejection_metadata["current_version_number"] == expected_version:
                        rejection_data = {
                            "attempt_id": attempt_id, "outcome": "pre_assignment_rejected",
                            "registration_event_index": registration_event_index,
                            "registration_event_hash": registration["event_hash"],
                            "preflight_event_index": preflight_event["event_index"],
                            "preflight_event_hash": preflight_event["event_hash"],
                            "dispatch_event_index": dispatch_event["event_index"],
                            "dispatch_event_hash": dispatch_event["event_hash"],
                            "accepted_event_index": None, "accepted_event_hash": None,
                            "assigned_version": None, "terminal_status": None,
                            "terminal_metadata": None, "status_evidence": [],
                            "actual_gpu_seconds": 0, "output_bytes": 0, "result": {},
                            "artifact_sha256": None,
                            "rejection_evidence": {"server_rejected_before_assignment": True,
                                                   "response_error_present": True},
                            "no_assignment_metadata": {"kernel_id": kernel_id,
                                                       "pre_push_current_version": expected_version,
                                                       "post_push_current_version": rejection_metadata["current_version_number"],
                                                       "no_version_assigned": True},
                            "implementation_variant_id": implementation_variant_id,
                            **hashes, "lock_head_hash": preflight_data["lock_head_hash"],
                        }
                        return registry.resolve_workload_attempt(rejection_data)
                raise KaggleError("Kaggle push did not return an exact positive assigned version; dispatch remains pending.")
            confirming_metadata = _sdk_metadata(api, kernel_id)
            if confirming_metadata["current_version_number"] != assigned:
                raise KaggleError("Kaggle assigned version is not the exact current kernel version; attempt remains pending.")
            accepted_data = {
                "attempt_id": attempt_id, "registration_event_index": registration_event_index,
                "registration_event_hash": registration["event_hash"],
                "preflight_event_index": preflight_event["event_index"], "preflight_event_hash": preflight_event["event_hash"],
                "dispatch_event_index": dispatch_event["event_index"], "dispatch_event_hash": dispatch_event["event_hash"],
                "assigned_version": assigned, "implementation_variant_id": implementation_variant_id,
                "push_response": {"version_number": assigned, "kernel_id": kernel_id,
                                  "accepted_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")},
                "confirming_metadata": confirming_metadata,
                "lock_head_hash": preflight_data["lock_head_hash"],
            }
            accepted_event = registry.accept_workload_push(accepted_data)
        else:
            if phase != "workload_push_accepted":
                raise KaggleError("Unresolved recovery event tail blocks push or polling.")
            preflight_event, dispatch_event, accepted_event = existing_attempt[0], existing_attempt[1], existing_attempt[2]
            preflight_data = preflight_event["event_data"]
            assigned = accepted_event["event_data"]["assigned_version"]
            _same = _sdk_metadata(api, kernel_id)
            if _same["current_version_number"] != assigned:
                raise KaggleError("Accepted attempt version drifted; attempt remains pending.")
            quota_remaining, quota_used_before = _quota_seconds(api)
            preflight_quota = preflight_data.get("quota_evidence", {}).get("remaining_gpu_seconds")
            if type(preflight_quota) is not int or preflight_quota < 0:
                raise KaggleError("Accepted attempt preflight lacks its original quota reservation evidence.")
            budget = _validate_recovery_budget(
                registry, registration, timeout, preflight_quota,
                resuming_attempt_id=attempt_id,
            )
            accepted_at = accepted_event["event_data"].get("push_response", {}).get("accepted_at_utc")
            try:
                accepted_time = datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - accepted_time).total_seconds()
            except (AttributeError, TypeError, ValueError) as exc:
                raise KaggleError("Accepted attempt lacks a usable durable start time; keep its reservation pending.") from exc
            remaining_deadline = timeout + reconciliation_slack_seconds - age
            if remaining_deadline <= 0:
                raise KaggleError("Accepted attempt is past its registered reconciliation deadline; keep it pending.")
            push_started = monotonic() - max(0, age)

        deadline = monotonic() + (remaining_deadline if phase == "workload_push_accepted"
                                  else timeout + reconciliation_slack_seconds)
        status_evidence = []
        terminal_metadata = None
        terminal_status = None
        while True:
            if monotonic() >= deadline:
                raise KaggleError("Kaggle attempt remains nonterminal at the registered monotonic timeout deadline.")
            metadata_before = _sdk_metadata(api, kernel_id)
            if metadata_before["current_version_number"] != assigned:
                raise KaggleError("Kernel current version drifted before latest-status poll; attempt remains pending.")
            status, status_item = _sdk_status(api, kernel_id)
            metadata_after = _sdk_metadata(api, kernel_id)
            if metadata_after != metadata_before:
                raise KaggleError("Kernel version changed across latest-status poll; attempt remains pending.")
            item = {**status_item, "kernel_id": kernel_id,
                    "current_version_number": assigned,
                    "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
            status_evidence.append(item)
            if status in {"COMPLETE", "ERROR", "CANCELLED", "TIMEOUT"}:
                terminal_status, terminal_metadata = status, {**metadata_after, "status": status}
                break
            now = monotonic()
            if now >= deadline:
                raise KaggleError("Kaggle attempt remains nonterminal at the registered monotonic timeout deadline.")
            sleep(min(float(poll_interval_seconds), max(0.0, deadline - now)))
        elapsed_seconds = max(0, math.ceil(monotonic() - push_started))
        quota_remaining_after, quota_used_after = _quota_seconds(api)
        quota_delta = quota_used_after - quota_used_before
        if quota_delta < 0 or quota_delta > timeout:
            raise KaggleError("Kaggle quota delta is contradictory or exceeds the reserved attempt timeout.")
        actual_gpu_seconds = min(timeout, max(elapsed_seconds, quota_delta))
        if output_bytes is None:
            raise KaggleError("Terminal Kaggle output size must be measured before releasing its reservation.")
        measured_output_bytes = output_bytes(kernel_id, assigned)
        if type(measured_output_bytes) is not int or measured_output_bytes < 0:
            raise KaggleError("Measured Kaggle output size is invalid; attempt remains pending.")
        terminal_data = {
            "attempt_id": attempt_id, "outcome": "terminal",
            "registration_event_index": registration_event_index,
            "registration_event_hash": registration["event_hash"],
            "preflight_event_index": preflight_event["event_index"], "preflight_event_hash": preflight_event["event_hash"],
            "dispatch_event_index": dispatch_event["event_index"], "dispatch_event_hash": dispatch_event["event_hash"],
            "accepted_event_index": accepted_event["event_index"], "accepted_event_hash": accepted_event["event_hash"],
            "assigned_version": assigned, "terminal_status": terminal_status,
            "terminal_metadata": terminal_metadata, "status_evidence": status_evidence,
            "actual_gpu_seconds": actual_gpu_seconds, "output_bytes": measured_output_bytes,
            "result": {"quota_gpu_seconds_before": quota_used_before,
                       "quota_gpu_seconds_after": quota_used_after,
                       "quota_gpu_seconds_remaining_after": quota_remaining_after,
                       "projected_budget": budget},
            "artifact_sha256": None, "rejection_evidence": None, "no_assignment_metadata": None,
            "implementation_variant_id": implementation_variant_id,
            **hashes, "lock_head_hash": preflight_data["lock_head_hash"],
        }
        terminal_event = registry.resolve_workload_attempt(terminal_data)
        if terminal_callback is not None:
            terminal_callback(terminal_event)
        return terminal_event


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
    if isinstance(report, dict) and report.get("format") == "circuit-ai-g008-principal-benchmark/3":
        _validate_principal_benchmark_v3(root, expected_run_id, report, evidence, report_path)
        return
    if (not isinstance(report, dict) or report.get("format") != "circuit-ai-g008-principal-benchmark/2"
            or not isinstance(evidence, dict)
            or evidence.get("format") != "circuit-ai-kaggle-g008-benchmark-evidence/1"
            or evidence.get("run_id") != expected_run_id or evidence.get("benchmark_report") != report
            or evidence.get("benchmark_report_sha256") != file_sha256(report_path)):
        raise KaggleError("G008 benchmark report/evidence identity or digest is invalid.")
    python_version = evidence.get("python_version")
    torch_version = evidence.get("torch_version")
    cuda_version = evidence.get("cuda_version")
    visible_cuda_devices = evidence.get("visible_cuda_devices")
    if (not isinstance(python_version, str) or not python_version.strip()
            or not isinstance(torch_version, str) or not torch_version.strip()
            or not isinstance(cuda_version, str) or not cuda_version.strip()
                or not isinstance(visible_cuda_devices, list) or len(visible_cuda_devices) != 1
            or any(not isinstance(name, str) or not name.strip() for name in visible_cuda_devices)):
        raise KaggleError("G008 Kaggle execution evidence lacks required Python/PyTorch/CUDA or visible GPU metadata.")
    try:
        from training.eval.provenance import (benchmark_run_keys, schedule_spec_digest,
                                              seed_plan_sha256)
        from training.ppo.g008_benchmark import (CONFIG_PATH, load_principal_config,
                                                 prepare_benchmark_schedules)

        run_keys = benchmark_run_keys()
        roster = report.get("schedule_roster")
        if (not isinstance(roster, list) or len(roster) != len(run_keys)
                or any(not isinstance(item, dict)
                       or set(item) != {"run_key", "run_id", "expanded_schedule_digest"}
                       for item in roster)
                or tuple(item["run_key"] for item in roster) != run_keys):
            raise KaggleError("G008 benchmark schedule roster is missing, altered, or out of frozen order.")
        run_ids = {item["run_key"]: item["run_id"] for item in roster}
        schedules = prepare_benchmark_schedules(run_ids)
        expected_plan_sha = seed_plan_sha256()
        expected_spec_digest = schedule_spec_digest(expected_plan_sha)
        principal_config, frozen_config_document = load_principal_config()
        reported_config = report.get("config")
        effective_config = principal_config.__dict__
        if (not isinstance(reported_config, dict)
                or json.dumps(reported_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                != json.dumps(effective_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                or report.get("source") != frozen_config_document["source"]
                or report.get("config_sha256") != frozen_config_document["config_sha256"]
                or report.get("config_artifact_sha256") != file_sha256(CONFIG_PATH)):
            raise KaggleError("G008 benchmark frozen config document or artifact binding is missing, malformed, or stale.")
        expected_minibatches = (principal_config.arena_count * principal_config.rollout_steps
                                + principal_config.minibatch_size - 1) // principal_config.minibatch_size
        expected_optimizer_steps = expected_minibatches * principal_config.update_epochs
        if (report.get("seed_plan_sha256") != expected_plan_sha
                or report.get("schedule_spec_digest") != expected_spec_digest):
            raise KaggleError("G008 benchmark seed-plan or schedule-spec binding is missing or stale.")
        for item in roster:
            schedule = schedules[item["run_key"]]
            if (item["expanded_schedule_digest"] != schedule["expanded_schedule_digest"]
                    or schedule["schedule_spec_digest"] != expected_spec_digest):
                raise KaggleError("G008 benchmark schedule roster digest is stale.")
    except KaggleError:
        raise
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise KaggleError(f"G008 frozen config or schedule binding is invalid: {exc}") from exc

    roster_by_key = {item["run_key"]: item for item in roster}
    device = report.get("device")
    repeats = report.get("repeats")
    warmups = report.get("warmups")
    if (not isinstance(device, dict) or device.get("device") != "cuda"
            or device.get("cuda_available") is not True
            or not isinstance(device.get("cuda_name"), str) or not device["cuda_name"].strip()
            or not isinstance(repeats, list) or len(repeats) != 5
            or not isinstance(warmups, dict) or warmups.get("count") != 2
            or not isinstance(warmups.get("runs"), list) or len(warmups["runs"]) != 2
            or report.get("measured_repeats") != 5):
        raise KaggleError("G008 benchmark lacks the frozen five-repeat CUDA evidence.")
    if visible_cuda_devices[0] != device["cuda_name"]:
        raise KaggleError("G008 visible training GPU does not match the benchmark report device.")
    if device["cuda_name"] != "Tesla T4":
        raise KaggleError("G008 benchmark must run on the approved Tesla T4 device.")

    def validate_update_evidence(
        metrics: dict, expected_updates: int, label: str, expected_cuda_name: str,
    ) -> int:
        run_device = metrics.get("device")
        training = metrics.get("training")
        updates = training.get("updates") if isinstance(training, dict) else None
        if not isinstance(run_device, dict):
            raise KaggleError(f"G008 {label} device evidence is invalid.")
        if not isinstance(updates, list) or len(updates) != expected_updates:
            raise KaggleError(f"G008 {label} must contain exactly {expected_updates} training updates.")

        total_updates = run_device.get("optimizer_updates")
        cuda_total_updates = run_device.get("cuda_optimizer_updates")
        if (type(total_updates) is not int or total_updates <= 0
                or type(cuda_total_updates) is not int or cuda_total_updates != total_updates):
            raise KaggleError(f"G008 {label} optimizer update totals are invalid.")
        update_total = 0
        for update_index, update in enumerate(updates):
            if not isinstance(update, dict):
                raise KaggleError(f"G008 {label} update {update_index} evidence is invalid.")
            count = update.get("optimizer_updates")
            cuda_count = update.get("cuda_updates")
            if (update.get("device") != "cuda" or type(count) is not int or count <= 0
                    or type(cuda_count) is not int or cuda_count != count
                    or count != expected_optimizer_steps):
                raise KaggleError(f"G008 {label} update {update_index} lacks valid CUDA optimizer evidence.")
            update_total += count
        if update_total != total_updates:
            raise KaggleError(f"G008 {label} optimizer update total disagrees with its training updates.")
        run_cuda_name = run_device.get("cuda_name")
        if (run_device.get("device") != "cuda" or run_device.get("cuda_available") is not True
                or not isinstance(run_cuda_name, str) or not run_cuda_name.strip()
                or run_cuda_name != expected_cuda_name):
            raise KaggleError(f"G008 {label} device evidence is invalid.")
        return total_updates

    if report.get("updates_per_repeat") != 3 or warmups.get("updates_each") != 1:
        raise KaggleError("G008 benchmark training update counts differ from the frozen protocol.")
    if (type(device.get("optimizer_updates")) is not int
            or device["optimizer_updates"] != expected_optimizer_steps
            or type(device.get("cuda_optimizer_updates")) is not int
            or device["cuda_optimizer_updates"] != expected_optimizer_steps):
        raise KaggleError("G008 top-level device optimizer counts do not match the frozen warmup run.")
    expected_transitions = (principal_config.arena_count * principal_config.rollout_steps
                            * principal_config.total_updates)

    def finite_positive(value: object, label: str, *, allow_zero: bool = False) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0)):
            raise KaggleError(f"G008 {label} is missing or is not a valid finite duration/rate.")
        return float(value)

    def percentiles(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        middle = len(ordered) // 2
        q1_values = ordered[:middle]
        q3_values = ordered[(len(ordered) + 1) // 2:]
        q1 = (q1_values[0] + q1_values[-1]) / 2
        q3 = (q3_values[0] + q3_values[-1]) / 2
        return {"median": ordered[middle], "min": ordered[0], "max": ordered[-1], "iqr": q3 - q1}

    derived_repeats = []
    measured_optimizer_updates_total = 0
    for index, repeat in enumerate(repeats):
        run_key = f"pair-{index:02d}"
        roster_item = roster_by_key[run_key]
        metrics_path = root / "measured" / f"repeat-{index:02d}" / "training-metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KaggleError(f"G008 measured repeat {index} is missing valid training metrics.") from exc
        if not isinstance(repeat, dict) or not isinstance(metrics, dict):
            raise KaggleError(f"G008 measured repeat {index} report entry is invalid.")
        optimizer_updates = validate_update_evidence(
            metrics, 3, f"measured repeat {index}", device["cuda_name"],
        )
        measured_optimizer_updates_total += optimizer_updates
        training = metrics.get("training")
        updates = training.get("updates") if isinstance(training, dict) else None
        if (metrics.get("format") != "circuit-ai-g008-benchmark-run-metrics/1"
                or not isinstance(training, dict) or not isinstance(updates, list)):
            raise KaggleError(f"G008 measured repeat {index} training metrics format is invalid.")
        rollout_seconds = finite_positive(training.get("mean_rollout_seconds"),
                                          f"measured repeat {index} rollout duration") * 3
        update_elapsed = finite_positive(training.get("update_elapsed_seconds"),
                                         f"measured repeat {index} update duration")
        wall_seconds = finite_positive(training.get("elapsed_seconds"),
                                       f"measured repeat {index} wall duration")
        optimizer_seconds = 0.0
        checkpoint_seconds = 0.0
        update_only_seconds = 0.0
        for update_index, update in enumerate(updates):
            optimizer_seconds += finite_positive(update.get("optimizer_seconds"),
                                                 f"measured repeat {index} optimizer duration")
            checkpoint = finite_positive(update.get("checkpoint_seconds"),
                                         f"measured repeat {index} checkpoint duration", allow_zero=True)
            latency = finite_positive(update.get("update_latency_seconds"),
                                      f"measured repeat {index} update latency")
            if checkpoint > latency:
                raise KaggleError(f"G008 measured repeat {index} checkpoint duration exceeds update latency.")
            checkpoint_seconds += checkpoint
            update_only_seconds += latency - checkpoint
        if update_elapsed + 1e-12 < update_only_seconds:
            raise KaggleError(f"G008 measured repeat {index} update duration is shorter than derived update-only time.")
        durations = {
            "rollout_seconds": rollout_seconds,
            "optimizer_seconds": optimizer_seconds,
            "update_only_seconds": update_only_seconds,
            "checkpoint_inclusive_update_elapsed_seconds": update_elapsed,
            "checkpoint_seconds": checkpoint_seconds,
            "wall_seconds": wall_seconds,
        }
        if any(value <= 0 for key, value in durations.items() if key != "checkpoint_seconds"):
            raise KaggleError(f"G008 measured repeat {index} has a non-positive derived duration.")
        derived = {
            "completed_joint_arena_transitions": expected_transitions,
            "agent_decisions": expected_transitions * 2,
            "update_only_joint_transitions_per_second": expected_transitions / update_only_seconds,
            "wall_inclusive_joint_transitions_per_second": expected_transitions / wall_seconds,
            "update_only_agent_decisions_per_second": expected_transitions * 2 / update_only_seconds,
            "wall_inclusive_agent_decisions_per_second": expected_transitions * 2 / wall_seconds,
            **durations,
        }
        for field, expected_value in derived.items():
            actual_value = repeat.get(field)
            if (type(actual_value) not in (int, float) or not math.isfinite(actual_value)
                    or actual_value != expected_value):
                raise KaggleError(f"G008 measured repeat {index} {field} disagrees with hashed training metrics.")
        derived_repeats.append(derived)
        if (repeat.get("repeat") != index or repeat.get("run_key") != run_key
                or repeat.get("run_id") != roster_item["run_id"]
                or repeat.get("expanded_schedule_digest") != roster_item["expanded_schedule_digest"]
                or repeat.get("training_metrics_sha256") != file_sha256(metrics_path)
                or type(repeat.get("optimizer_updates")) is not int
                or repeat.get("optimizer_updates") <= 0
                or type(repeat.get("cuda_optimizer_updates")) is not int
                or repeat.get("optimizer_updates") != optimizer_updates
                or repeat.get("cuda_optimizer_updates") != optimizer_updates):
            raise KaggleError(f"G008 measured repeat {index} failed CUDA metrics/hash validation.")
    if (type(report.get("optimizer_updates_total")) is not int
            or report["optimizer_updates_total"] != measured_optimizer_updates_total
            or type(report.get("cuda_optimizer_updates_total")) is not int
            or report["cuda_optimizer_updates_total"] != measured_optimizer_updates_total):
        raise KaggleError("G008 benchmark aggregate optimizer update totals are missing or invalid.")
    summary = report.get("summary")
    summary_fields = tuple(derived_repeats[0])
    if not isinstance(summary, dict) or set(summary) != set(summary_fields):
        raise KaggleError("G008 benchmark summary percentiles are missing or contain unexpected fields.")
    for field in summary_fields:
        expected_summary = percentiles([repeat[field] for repeat in derived_repeats])
        actual_summary = summary.get(field)
        if (not isinstance(actual_summary, dict) or set(actual_summary) != set(expected_summary)
                or any(type(actual_summary.get(key)) not in (int, float)
                       or not math.isfinite(actual_summary[key])
                       or actual_summary[key] != value
                       for key, value in expected_summary.items())):
            raise KaggleError(f"G008 benchmark summary {field} percentiles disagree with measured repeats.")
    for index, warmup in enumerate(warmups["runs"]):
        run_key = f"warmup-python-oracle-incumbent-{index:02d}"
        roster_item = roster_by_key[run_key]
        metrics_path = root / "warmups" / f"warmup-{index:02d}" / "training-metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KaggleError(f"G008 warmup {index} is missing valid training metrics.") from exc
        if not isinstance(warmup, dict) or not isinstance(metrics, dict):
            raise KaggleError(f"G008 warmup {index} report entry is invalid.")
        warmup_optimizer_updates = validate_update_evidence(metrics, 1, f"warmup {index}", device["cuda_name"])
        if (warmup_optimizer_updates != device["optimizer_updates"]
                or warmup_optimizer_updates != device["cuda_optimizer_updates"]):
            raise KaggleError(f"G008 warmup {index} optimizer counts disagree with top-level device evidence.")
        if (warmup.get("run_key") != run_key or warmup.get("run_id") != roster_item["run_id"]
                or warmup.get("expanded_schedule_digest") != roster_item["expanded_schedule_digest"]
                or warmup.get("training_metrics_sha256") != file_sha256(metrics_path)):
            raise KaggleError(f"G008 warmup {index} failed CUDA metrics validation.")


def _validate_principal_benchmark_v3(root: Path, expected_run_id: str, report: dict,
                                     evidence: object, report_path: Path) -> None:
    """Validate paired /3 evidence and every hashed arm in the frozen 21-run roster."""
    if (not isinstance(evidence, dict)
            or evidence.get("format") != "circuit-ai-kaggle-g008-benchmark-evidence/1"
            or evidence.get("run_id") != expected_run_id or evidence.get("benchmark_report") != report
            or evidence.get("benchmark_report_sha256") != file_sha256(report_path)):
        raise KaggleError("G008 /3 report/evidence identity or digest is invalid.")
    visible = evidence.get("visible_cuda_devices")
    device = report.get("device")
    if (not all(isinstance(evidence.get(field), str) and evidence[field].strip()
                for field in ("python_version", "torch_version", "cuda_version"))
            or not isinstance(visible, list) or len(visible) != 1
            or visible != ["Tesla T4"] or not isinstance(device, dict)
            or device.get("device") != "cuda" or device.get("cuda_available") is not True
            or device.get("cuda_name") != "Tesla T4"):
        raise KaggleError("G008 /3 requires complete execution metadata for exactly one Tesla T4.")
    try:
        from training.eval.provenance import benchmark_run_keys, schedule_spec_digest, seed_plan_sha256
        from training.ppo.g008_benchmark import CONFIG_PATH, load_principal_config, prepare_benchmark_schedules
        keys = benchmark_run_keys()
        roster = report.get("schedule_roster")
        if (not isinstance(roster, list) or len(roster) != 21
                or tuple(item.get("run_key") for item in roster if isinstance(item, dict)) != keys
                or any(not isinstance(item, dict) or set(item) != {"run_key", "run_id", "expanded_schedule_digest"}
                       for item in roster)):
            raise KaggleError("G008 /3 must bind all five pairs and sixteen warmups in frozen order.")
        run_ids = {item["run_key"]: item["run_id"] for item in roster}
        schedules = prepare_benchmark_schedules(run_ids)
        plan_sha = seed_plan_sha256()
        spec_digest = schedule_spec_digest(plan_sha)
        config, config_doc = load_principal_config()
        if (report.get("seed_plan_sha256") != plan_sha or report.get("schedule_spec_digest") != spec_digest
                or report.get("config") != config.__dict__ or report.get("source") != config_doc["source"]
                or report.get("config_sha256") != config_doc["config_sha256"]
                or report.get("config_artifact_sha256") != file_sha256(CONFIG_PATH)):
            raise KaggleError("G008 /3 config, seed plan, or schedule specification is stale.")
        for item in roster:
            if item["expanded_schedule_digest"] != schedules[item["run_key"]]["expanded_schedule_digest"]:
                raise KaggleError("G008 /3 schedule roster digest is stale.")
    except KaggleError:
        raise
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise KaggleError(f"G008 /3 frozen schedule/config validation failed: {exc}") from exc
    roster_by_key = {item["run_key"]: item for item in roster}
    implementations = report.get("implementations")
    if (not isinstance(implementations, dict) or set(implementations) != {"incumbent", "candidate"}
            or any(not isinstance(item, dict) or set(item) != {"trainer_sha256", "engine_sha256"}
                   or any(not re.fullmatch(r"[0-9a-f]{64}", item.get(field, ""))
                          for field in ("trainer_sha256", "engine_sha256")) for item in implementations.values())
            or report.get("source_sha256") != hashlib.sha256(
                json.dumps(implementations, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()):
        raise KaggleError("G008 /3 implementation identity map or source hash is invalid.")
    source_commits = report.get("implementation_source_commits")
    if (not isinstance(source_commits, dict) or set(source_commits) != {"incumbent", "candidate"}
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value)
                   for value in source_commits.values())):
        raise KaggleError("G008 /3 implementation source commits are missing or malformed.")
    parity = report.get("parity")
    pairs = report.get("pairs")
    if (not isinstance(parity, dict) or set(parity) != {"passed", "pairs", "implementation_hashes"}
            or type(parity.get("passed")) is not bool
            or not isinstance(pairs, list) or len(pairs) != 5 or report.get("measured_repeats") != 5
            or parity.get("implementation_hashes") != implementations):
        raise KaggleError("G008 /3 requires exact parity evidence, implementation hashes, and five measured pairs.")
    parity_pairs = parity.get("pairs")
    if (not isinstance(parity_pairs, dict) or set(parity_pairs) != {f"pair-{i:02d}" for i in range(5)}
            or any(not isinstance(item, dict) for item in parity_pairs.values())
            or parity["passed"] != all(item.get("passed") is True for item in parity_pairs.values()
                                         if isinstance(item, dict))):
        raise KaggleError("G008 /3 parity summary differs from its five pair-level traces.")
    expected_parity_transitions = config.arena_count * config.rollout_steps
    expected_parity_action_source = "g008-trainer-model-sampled-collector/1"
    for parity_index in range(5):
        parity_key = f"pair-{parity_index:02d}"
        pair_parity = parity_pairs[parity_key]
        if (not isinstance(pair_parity, dict)
                or set(pair_parity) != {"passed", "incumbent", "candidate", "implementation_hashes"}
                or type(pair_parity.get("passed")) is not bool
                or pair_parity["implementation_hashes"] != implementations):
            raise KaggleError(f"G008 {parity_key} parity trace shape or implementation identity is invalid.")
        traces = []
        for role in ("incumbent", "candidate"):
            trace = pair_parity[role]
            if (not isinstance(trace, dict)
                    or set(trace) != {"sha256", "transition_count", "reset_count", "action_source"}
                    or not isinstance(trace.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", trace["sha256"])
                    or type(trace.get("transition_count")) is not int
                    or trace["transition_count"] != expected_parity_transitions
                    or type(trace.get("reset_count")) is not int
                    or not 0 <= trace["reset_count"] <= expected_parity_transitions
                    or trace.get("action_source") != expected_parity_action_source):
                raise KaggleError(f"G008 {parity_key}/{role} parity trace digest/count/action source is invalid.")
            traces.append(trace)
        if pair_parity["passed"] is not (traces[0]["sha256"] == traces[1]["sha256"]):
            raise KaggleError(f"G008 {parity_key} passed flag contradicts its two parity trace hashes.")
        if pair_parity["passed"] and traces[0]["reset_count"] != traces[1]["reset_count"]:
            raise KaggleError(f"G008 {parity_key} passed parity arms report contradictory reset counts.")
    expected_transitions = config.arena_count * config.rollout_steps * config.total_updates
    expected_decisions = expected_transitions * 2
    expected_minibatches = (config.arena_count * config.rollout_steps + config.minibatch_size - 1) // config.minibatch_size
    expected_steps = expected_minibatches * config.update_epochs

    def metrics_at(relative: object, expected_hash: object, label: str) -> dict:
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not isinstance(expected_hash, str)):
            raise KaggleError(f"G008 {label} metrics path/hash is malformed.")
        path = root / relative
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KaggleError(f"G008 {label} metrics are missing or invalid.") from exc
        if file_sha256(path) != expected_hash or not isinstance(data, dict):
            raise KaggleError(f"G008 {label} metrics hash or object is invalid.")
        return data

    def positive(value: object, label: str, zero: bool = False) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (value < 0 if zero else value <= 0)):
            raise KaggleError(f"G008 {label} is not a finite {'nonnegative' if zero else 'positive'} number.")
        return float(value)

    def validate_run(metrics: dict, updates_count: int, label: str) -> dict:
        training = metrics.get("training")
        run_device = metrics.get("device")
        updates = training.get("updates") if isinstance(training, dict) else None
        if (metrics.get("format") != "circuit-ai-g008-benchmark-run-metrics/1"
                or not isinstance(run_device, dict) or run_device.get("device") != "cuda"
                or run_device.get("cuda_available") is not True or run_device.get("cuda_name") != "Tesla T4"
                or not isinstance(updates, list) or len(updates) != updates_count):
            raise KaggleError(f"G008 {label} run/device/update metrics are invalid.")
        total = 0
        for update in updates:
            count = update.get("optimizer_updates") if isinstance(update, dict) else None
            cuda = update.get("cuda_updates") if isinstance(update, dict) else None
            if (update.get("device") != "cuda" or type(count) is not int or count != expected_steps
                    or type(cuda) is not int or cuda != count):
                raise KaggleError(f"G008 {label} optimizer update evidence differs from frozen config.")
            total += count
        if (type(run_device.get("optimizer_updates")) is not int or run_device["optimizer_updates"] != total
                or type(run_device.get("cuda_optimizer_updates")) is not int
                or run_device["cuda_optimizer_updates"] != total):
            raise KaggleError(f"G008 {label} device optimizer totals disagree with measured updates.")
        return {"total": total, "training": training, "updates": updates}

    rates: list[float] = []
    derived_by_role = {role: {field: [] for field in (
        "wall_inclusive_joint_transitions_per_second", "update_only_seconds", "wall_seconds",
        "optimizer_seconds", "rollout_seconds")} for role in ("incumbent", "candidate")}
    measured_optimizer_total = 0
    for index, pair in enumerate(pairs):
        key = f"pair-{index:02d}"
        if (not isinstance(pair, dict) or pair.get("repeat") != index or pair.get("run_key") != key
                or pair.get("run_id") != roster_by_key[key]["run_id"]
                or pair.get("seed") != config.seed
                or pair.get("order") != (["incumbent", "candidate"] if index % 2 == 0
                                           else ["candidate", "incumbent"])
                or pair.get("expanded_schedule_digest") != roster_by_key[key]["expanded_schedule_digest"]
                or pair.get("config_sha256") != report["config_sha256"]
                or pair.get("parity_sha256") != parity_pairs[key]["incumbent"]["sha256"]
                or not isinstance(pair.get("arms"), dict) or set(pair["arms"]) != {"incumbent", "candidate"}):
            raise KaggleError(f"G008 /3 measured pair {index} does not match its frozen schedule.")
        role_rates = {}
        for role, arm in pair["arms"].items():
            if (not isinstance(arm, dict) or arm.get("run_key") != key
                    or arm.get("run_id") != pair["run_id"] or arm.get("seed") != config.seed
                    or arm.get("config_sha256") != report["config_sha256"]
                    or arm.get("expanded_schedule_digest") != pair["expanded_schedule_digest"]):
                raise KaggleError(f"G008 /3 {key}/{role} arm identity is invalid.")
            identity = arm.get("implementation")
            hashes = implementations[role]
            if (not isinstance(identity, dict) or identity.get("role") != role
                    or identity.get("variant_id") != role or identity.get("trainer_sha256") != hashes["trainer_sha256"]
                    or identity.get("engine_sha256") != hashes["engine_sha256"]
                    or identity.get("source_commit") != source_commits[role]):
                raise KaggleError(f"G008 /3 {key}/{role} implementation identity is invalid.")
            metrics = metrics_at(arm.get("training_metrics_file"), arm.get("training_metrics_sha256"), f"{key}/{role}")
            measured = validate_run(metrics, config.total_updates, f"{key}/{role}")
            measured_optimizer_total += measured["total"]
            training, updates = measured["training"], measured["updates"]
            rollout = positive(training.get("mean_rollout_seconds"), f"{key}/{role} rollout") * config.total_updates
            update_elapsed = positive(training.get("update_elapsed_seconds"), f"{key}/{role} update")
            wall = positive(training.get("elapsed_seconds"), f"{key}/{role} wall")
            optimizer = sum(positive(item.get("optimizer_seconds"), f"{key}/{role} optimizer") for item in updates)
            checkpoint = sum(positive(item.get("checkpoint_seconds"), f"{key}/{role} checkpoint", True) for item in updates)
            update_only = sum(positive(item.get("update_latency_seconds"), f"{key}/{role} latency") -
                              positive(item.get("checkpoint_seconds"), f"{key}/{role} checkpoint", True) for item in updates)
            derived = {"completed_joint_arena_transitions": expected_transitions,
                       "agent_decisions": expected_decisions,
                       "rollout_seconds": rollout, "optimizer_seconds": optimizer,
                       "update_only_seconds": update_only,
                       "checkpoint_inclusive_update_elapsed_seconds": update_elapsed,
                       "checkpoint_seconds": checkpoint, "wall_seconds": wall,
                       "update_only_joint_transitions_per_second": expected_transitions / update_only,
                       "wall_inclusive_joint_transitions_per_second": expected_transitions / wall,
                       "update_only_agent_decisions_per_second": expected_decisions / update_only,
                       "wall_inclusive_agent_decisions_per_second": expected_decisions / wall,
                       "optimizer_updates": measured["total"], "cuda_optimizer_updates": measured["total"]}
            if update_only <= 0 or update_elapsed < update_only:
                raise KaggleError(f"G008 {key}/{role} derived update duration is invalid.")
            if (optimizer <= 0 or rollout <= 0 or checkpoint < 0
                    or any(item.get("checkpoint_seconds", 0) > item.get("update_latency_seconds", 0)
                           for item in updates)):
                raise KaggleError(f"G008 {key}/{role} component durations are contradictory.")
            if any(arm.get(name) != value for name, value in derived.items()):
                raise KaggleError(f"G008 {key}/{role} throughput/duration fields differ from hashed metrics.")
            for field in derived_by_role[role]:
                derived_by_role[role][field].append(derived[field])
            role_rates[role] = derived["update_only_joint_transitions_per_second"]
        ratio = role_rates["candidate"] / role_rates["incumbent"]
        if pair.get("candidate_speedup_ratio") != ratio:
            raise KaggleError(f"G008 pair {index} speedup ratio differs from hashed metrics.")
        rates.append(ratio)
    warmup = report.get("warmups")
    warmup_runs = warmup.get("runs") if isinstance(warmup, dict) else None
    if (not isinstance(warmup_runs, list) or len(warmup_runs) != 16
            or warmup.get("count_per_role") != 2 or warmup.get("discarded") is not True):
        raise KaggleError("G008 /3 requires exactly sixteen discarded frozen warmup runs.")
    expected_warmup_keys = [key for key in keys if key.startswith("warmup-")]
    for index, (item, run_key) in enumerate(zip(warmup_runs, expected_warmup_keys)):
        if (not isinstance(item, dict) or item.get("run_key") != run_key
                or item.get("run_id") != roster_by_key[run_key]["run_id"]
                or item.get("expanded_schedule_digest") != roster_by_key[run_key]["expanded_schedule_digest"]
                or item.get("completed_joint_arena_transitions") != config.arena_count * config.rollout_steps
                or item.get("discarded") is not True):
            raise KaggleError(f"G008 warmup slot {index} does not match the frozen 16-run order.")
        backend, role = run_key.removeprefix("warmup-").rsplit("-", 2)[0], run_key.removesuffix(f"-{run_key[-2:]}").rsplit("-", 1)[-1]
        if item.get("backend_id") != backend or item.get("role") != role:
            raise KaggleError(f"G008 warmup slot {index} backend or implementation role is wrong.")
        metrics = metrics_at(item.get("training_metrics_file"), item.get("training_metrics_sha256"), run_key)
        measured = validate_run(metrics, 1, run_key)
        identity = item.get("implementation")
        if (not isinstance(identity, dict) or identity.get("role") != role or identity.get("variant_id") != role
                or identity.get("trainer_sha256") != implementations[role]["trainer_sha256"]
                or identity.get("engine_sha256") != implementations[role]["engine_sha256"]
                or identity.get("source_commit") != source_commits[role]
                or measured["total"] != sum(positive(u.get("optimizer_seconds"), f"{run_key} optimizer") > 0
                                             for u in measured["updates"]) * expected_steps):
            raise KaggleError(f"G008 warmup {run_key} implementation or optimizer counts are invalid.")
    summary = report.get("summary")
    ratio_summary = summary.get("candidate_speedup_ratio") if isinstance(summary, dict) else None
    median = statistics.median(rates)
    def pct(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {"median": statistics.median(ordered), "min": min(ordered), "max": max(ordered),
                "iqr": statistics.median(ordered[(len(ordered) + 1) // 2:])
                - statistics.median(ordered[:len(ordered) // 2])}
    expected_summaries = {
        "candidate_update_only_joint_transitions_per_second": pct(
            [value for pair in pairs for value in [pair["arms"]["candidate"]["update_only_joint_transitions_per_second"]]]),
        "incumbent_update_only_joint_transitions_per_second": pct(
            [pair["arms"]["incumbent"]["update_only_joint_transitions_per_second"] for pair in pairs]),
        "candidate_speedup_ratio": pct(rates),
    }
    for role in ("incumbent", "candidate"):
        for field, values in derived_by_role[role].items():
            expected_summaries[f"{role}_{field}"] = pct(values)
    if (not isinstance(summary, dict) or set(summary) != set(expected_summaries)
            or any(not isinstance(summary.get(field), dict) or summary[field] != expected
                   for field, expected in expected_summaries.items())
            or not isinstance(ratio_summary, dict) or ratio_summary.get("median") != median
            or report.get("selection") != report.get("selected_variant_id")
            or report.get("selection") not in {"incumbent", "candidate"}):
        raise KaggleError("G008 /3 summary or selected implementation is invalid.")
    expected_selection = ("candidate" if parity["passed"] and median >= 1.10
                          and sum(value >= 1.10 for value in rates) >= 4 else "incumbent")
    if report["selection"] != expected_selection:
        raise KaggleError("G008 /3 selection does not follow parity and paired speedup evidence.")
    if (type(device.get("optimizer_updates")) is not int or device["optimizer_updates"] != expected_steps
            or type(device.get("cuda_optimizer_updates")) is not int
            or device["cuda_optimizer_updates"] != expected_steps
            or report.get("optimizer_updates_total") != measured_optimizer_total
            or report.get("cuda_optimizer_updates_total") != measured_optimizer_total):
        raise KaggleError("G008 /3 top-level optimizer counts disagree with frozen runs and hashed metrics.")


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
    recovery = commands.add_parser("recovery-run", help="run one registered, budgeted G008 recovery workload")
    recovery.add_argument("--registry", type=Path, required=True)
    recovery.add_argument("--registration-event-index", type=int, required=True)
    recovery.add_argument("--attempt-id", required=True)
    recovery.add_argument("--variant", choices=("incumbent", "candidate"), default="incumbent")
    recovery.add_argument("--source-dir", type=Path, required=True)
    recovery.add_argument("--output-dir", type=Path, required=True)
    recovery.add_argument("--no-external-writer-confirmed", action="store_true",
                          help="assert the operator has disabled all other writers for this kernel")
    recovery.add_argument("--confirm", action="store_true", help="explicitly authorize the remote recovery push")
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
        elif args.command == "recovery-run":
            if not args.confirm:
                raise KaggleError("Recovery push requires --confirm. No Kaggle command was run.")
            if not args.no_external_writer_confirmed:
                raise KaggleError("Recovery preflight requires --no-external-writer-confirmed.")
            from training.eval.registry import Registry
            registry = Registry(args.registry)
            events = registry.events()
            if (type(args.registration_event_index) is not int or not 0 <= args.registration_event_index < len(events)
                    or events[args.registration_event_index].get("event_type") != "registration"):
                raise KaggleError("Recovery registration event index is not an exact prior registration.")
            registration = events[args.registration_event_index]["event_data"]
            record = registration.get("record", {})
            kernel_id = _kernel_id(record.get("kernel_id"))
            if args.output_dir.exists():
                raise KaggleError("Recovery output directory already exists; refusing to overwrite.")
            args.output_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
            except (ImportError, AttributeError) as exc:
                raise KaggleError("Pinned Kaggle CLI SDK is not installed/configured.") from exc

            def download_outputs(exact_kernel: str, version: int) -> int:
                args.output_dir.mkdir(parents=True, exist_ok=False)
                api.kernels_output(f"{exact_kernel}/{version}", path=str(args.output_dir), force=False, quiet=True)
                return sum(path.stat().st_size for path in args.output_dir.rglob("*") if path.is_file())

            def bind_complete_benchmark(terminal_event: dict) -> None:
                if (terminal_event.get("event_data", {}).get("terminal_status") == "COMPLETE"
                        and registration.get("registration_kind") == "benchmark"):
                    register_recovery_benchmark_artifact(
                        registry, terminal_event, args.output_dir,
                        args.output_dir / "g008-recovery-benchmark-artifact.json",
                    )

            terminal = run_recovery_workload(
                registry, args.registration_event_index, args.source_dir,
                attempt_id=args.attempt_id,
                implementation_variant_id=("paired" if registration.get("registration_kind") == "benchmark"
                                            else args.variant),
                no_external_writer_confirmed=args.no_external_writer_confirmed,
                api=api, output_bytes=download_outputs, terminal_callback=bind_complete_benchmark,
            )
            print(f"Recovery attempt {args.attempt_id} resolved: {terminal['event_data'].get('terminal_status') or terminal['event_data']['outcome']}")
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
