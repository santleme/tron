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


_CAMPAIGN_TRAIN_FLOAT_FIELDS = frozenset({
    "learning_rate", "gamma", "gae_lambda", "clip_ratio", "entropy_coef",
    "value_coef", "snapshot_opponent_probability", "heuristic_opponent_probability",
})


def _campaign_train_config_wire(config: dict) -> dict:
    """Encode TrainConfig for registry canonical JSON, which disallows floats."""
    result = dict(config)
    for key in _CAMPAIGN_TRAIN_FLOAT_FIELDS:
        if key not in result:
            continue
        value = result.get(key)
        if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
            raise KaggleError(f"G008 TrainConfig field {key} must be a finite number.")
        result[key] = repr(float(value))
    return result


def _campaign_train_config_native(config: dict) -> dict:
    """Decode the registry's stringified float fields into a native TrainConfig."""
    result = dict(config)
    for key in _CAMPAIGN_TRAIN_FLOAT_FIELDS:
        value = result.get(key)
        if not isinstance(value, str):
            raise KaggleError(f"Registered G008 TrainConfig field {key} is not a canonical float string.")
        try:
            number = float(value)
        except ValueError as exc:
            raise KaggleError(f"Registered G008 TrainConfig field {key} is malformed.") from exc
        if not math.isfinite(number) or repr(number) != value:
            raise KaggleError(f"Registered G008 TrainConfig field {key} is not canonical.")
        result[key] = number
    return result


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
    if not isinstance(settings, dict) or set(settings) - {
            "run_id", "train_config", "throughput_benchmark", "g008_principal_benchmark",
            "g008_train_schedule", "g008_campaign_train", "g008_evaluation"}:
        raise KaggleError("Run config contains unsupported fields.")
    run_id = settings.get("run_id", "kaggle-run")
    config = settings.get("train_config", {})
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", run_id):
        raise KaggleError("Run config run_id must match the artifact ID contract.")
    allowed_config = {field.name for field in fields(__import__("training.ppo.train", fromlist=["TrainConfig"]).TrainConfig)}
    if not isinstance(config, dict) or set(config) - allowed_config:
        raise KaggleError("Run config train_config contains unknown fields or is not an object.")
    settings_out = {"run_id": run_id, "train_config": config}
    if "g008_evaluation" in settings:
        evaluation = settings["g008_evaluation"]
        if not isinstance(evaluation, dict):
            raise KaggleError("g008_evaluation must be an object.")
        mode = evaluation.get("mode")
        expected_keys = {
            "campaign_id", "registration_kind", "mode", "run_uuid", "candidate_dataset_source",
            "candidate_checkpoint_path", "roster_sha256", "opponent_code_sha256", "game_count",
            "seat_order", "seed_plan_sha256", "opponent_id",
        }
        if mode == "validation":
            expected_keys |= {"opponent_asset_path", "opponent_checkpoint_sha256"}
        if mode not in {"validation", "final"} or set(evaluation) != expected_keys:
            raise KaggleError("g008_evaluation fields differ from the frozen validation/final contract.")
        if (evaluation.get("campaign_id") not in _CAMPAIGN_IDS
                or evaluation.get("registration_kind") != mode
                or evaluation.get("game_count") != 512 or evaluation.get("seat_order") != [0, 1]
                or evaluation.get("candidate_dataset_source") != "pascalcoimbra/g008-frozen-candidate"
                or evaluation.get("candidate_checkpoint_path") != "/kaggle/input/g008-frozen-candidate/candidate.pt"):
            raise KaggleError("g008_evaluation changed the frozen campaign, roster size, seat order, or candidate dataset binding.")
        run_uuid = evaluation.get("run_uuid")
        if (not isinstance(run_uuid, str) or not re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", run_uuid)
                or run_id not in {"kaggle-run", run_uuid}):
            raise KaggleError("g008_evaluation run_uuid must be canonical and match run_id when supplied.")
        if run_id == "kaggle-run":
            run_id = run_uuid
            settings_out["run_id"] = run_uuid
        if any(key in settings for key in ("g008_train_schedule", "g008_principal_benchmark", "throughput_benchmark")):
            raise KaggleError("Registered G008 evaluation cannot be combined with training or benchmark workloads.")
        try:
            from training.eval.provenance import seed_plan_sha256
            if mode == "final":
                from training.eval.g008_final_roster import EXPECTED_FINAL_ROSTER
                expected = EXPECTED_FINAL_ROSTER
            else:
                from training.eval.g008_validation import _EXPECTED_ROSTERS
                expected = _EXPECTED_ROSTERS["validation"]
            expected_plan_sha = seed_plan_sha256()
        except (ImportError, KeyError, OSError, ValueError) as exc:
            raise KaggleError(f"Could not load frozen G008 evaluation identities: {exc}") from exc
        if (evaluation.get("roster_sha256") != expected["roster_sha256"]
                or evaluation.get("opponent_code_sha256") != expected["opponent_code_sha256"]
                or evaluation.get("opponent_id") != expected["opponent_id"]
                or evaluation.get("seed_plan_sha256") != expected_plan_sha):
            raise KaggleError("g008_evaluation roster, opponent, or seed plan differs from its frozen campaign commitment.")
        if mode == "validation":
            expected_opponent_path = "training/eval/assets/g007-step-300.pt"
            opponent_source = Path(__file__).resolve().parents[2] / "artifacts/circuit-ai/runs/lc-circuit-ai-selfplay-300-viewer-ready/checkpoints/step-300.pt"
            from training.ppo.protocol import file_sha256
            if (evaluation.get("opponent_asset_path") != expected_opponent_path
                    or evaluation.get("opponent_checkpoint_sha256") != file_sha256(opponent_source)):
                raise KaggleError("g008 validation must bind the packaged frozen G007 step-300 checkpoint bytes.")
        settings_out["g008_evaluation"] = evaluation
    if "g008_train_schedule" in settings:
        training = settings["g008_train_schedule"]
        v2_schedule = isinstance(training, dict) and training.get("campaign_id") == "g008-competitive-20261021-v2"
        expected_schedule_keys = ({"campaign_id", "pair_id", "ppo_seed", "run_uuid", "root_seed"}
                                  if v2_schedule else {"run_uuid", "root_seed"})
        if not isinstance(training, dict) or set(training) != expected_schedule_keys:
            raise KaggleError("g008_train_schedule fields differ from the frozen v1/v2 contract.")
        run_uuid = training["run_uuid"]
        if (not isinstance(run_uuid, str) or not re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                run_uuid)):
            raise KaggleError("g008_train_schedule run_uuid must be a canonical lowercase RFC 4122 UUID.")
        if v2_schedule and (not isinstance(training.get("pair_id"), str) or not training["pair_id"]
                            or type(training.get("ppo_seed")) is not int
                            or not 0 <= training["ppo_seed"] <= 2**32 - 1
                            or type(training.get("root_seed")) is not int
                            or not 2_000_000 <= training["root_seed"] <= 2_999_999):
            raise KaggleError("G008 v2 training schedule requires a frozen pair ID, PPO seed, and environment root seed.")
        if run_id != run_uuid:
            raise KaggleError("G008 training run_id must equal its preregistered run_uuid.")
        if "throughput_benchmark" in settings or "g008_principal_benchmark" in settings:
            raise KaggleError("G008 train-only schedule cannot be combined with benchmark workloads.")
        try:
            from training.eval.provenance import _read_seed_plan, build_training_run_schedule
            from training.ppo.g008_benchmark import load_principal_config
            from training.ppo.train import TrainConfig
            if config.get("device", "auto") not in {"auto", "cuda"}:
                raise ValueError("G008 train-only workload requires CUDA execution")
            train_config = TrainConfig(**{**config, "device": "auto"})
            train_config.validate()
            frozen_config, _ = load_principal_config()
            frozen_config = replace(frozen_config, device="auto")
            changed_fields = [field.name for field in fields(TrainConfig)
                              if getattr(train_config, field.name) != getattr(frozen_config, field.name)]
            if v2_schedule:
                changed_fields = [name for name in changed_fields if name != "seed"]
            if changed_fields:
                raise ValueError(
                    "G008 train-only TrainConfig differs from the frozen principal config in "
                    f"{', '.join(changed_fields)}")
            plan, plan_sha = _read_seed_plan()
            expected_ppo_seed = training["ppo_seed"] if v2_schedule else plan["model_ppo_seed"]
            if train_config.seed != expected_ppo_seed:
                raise ValueError("G008 TrainConfig.seed differs from its frozen model/PPO seed")
            schedule = build_training_run_schedule(
                run_id=run_uuid, root_seed=training["root_seed"],
                arena_count=train_config.arena_count, total_updates=train_config.total_updates,
                rollout_steps=train_config.rollout_steps, seed_plan_sha=plan_sha)
        except (TypeError, ValueError) as exc:
            raise KaggleError(f"Invalid preregistered G008 training schedule: {exc}") from exc
        settings_out["g008_train_schedule"] = {
            "run_uuid": run_uuid,
            "root_seed": training["root_seed"],
            "expanded_schedule_digest": schedule["expanded_schedule_digest"],
            "schedule_spec_digest": schedule["schedule_spec_digest"],
            "seed_plan_sha256": plan_sha,
        }
        if v2_schedule:
            settings_out["g008_train_schedule"].update({
                "campaign_id": training["campaign_id"], "pair_id": training["pair_id"],
                "ppo_seed": training["ppo_seed"],
            })
    if "g008_campaign_train" in settings:
        campaign = settings["g008_campaign_train"]
        expected_campaign_keys = {
            "campaign_id", "registration_id", "run_uuid", "pair_id", "strategy_role",
            "root_seed", "ppo_seed", "update_budget", "seed_plan_sha256",
            "opponent_probabilities", "config_sha256", "shared_config_sha256",
            "schedule_descriptor_sha256",
        }
        if (not isinstance(campaign, dict) or set(campaign) != expected_campaign_keys
                or campaign.get("campaign_id") != "g008-competitive-20261021-v2"
                or campaign.get("strategy_role") not in {"baseline", "mixed"}
                or campaign.get("pair_id") not in {"seed-pair-1", "seed-pair-2", "seed-pair-3"}
                or campaign.get("update_budget") != 1_000
                or type(campaign.get("ppo_seed")) is not int
                or type(campaign.get("root_seed")) is not int
                or not all(re.fullmatch(r"[0-9a-f]{64}", str(campaign.get(key, "")))
                           for key in ("seed_plan_sha256", "config_sha256", "shared_config_sha256",
                                       "schedule_descriptor_sha256"))):
            raise KaggleError("g008_campaign_train fields differ from the frozen v2 training contract.")
        expected_seed_by_pair = {"seed-pair-1": 3301, "seed-pair-2": 3302, "seed-pair-3": 3303}
        expected_root_by_pair = {"seed-pair-1": 2_100_001, "seed-pair-2": 2_200_001,
                                 "seed-pair-3": 2_300_001}
        if (campaign["ppo_seed"] != expected_seed_by_pair[campaign["pair_id"]]
                or campaign["root_seed"] != expected_root_by_pair[campaign["pair_id"]]
                or run_id != campaign["run_uuid"]
                or settings.get("g008_train_schedule") is not None
                or any(key in settings for key in ("g008_evaluation", "g008_principal_benchmark",
                                                   "throughput_benchmark"))):
            raise KaggleError("g008_campaign_train changed its frozen paired seed or workload binding.")
        role_probabilities = {
            "baseline": {"latest": 10_000, "step0": 0, "cruise": 0,
                         "local_mobility": 0, "interceptor": 0},
            "mixed": {"latest": 6_000, "step0": 1_000, "cruise": 1_000,
                      "local_mobility": 1_000, "interceptor": 1_000},
        }
        if campaign["opponent_probabilities"] != role_probabilities[campaign["strategy_role"]]:
            raise KaggleError("g008_campaign_train opponent pool differs from the preregistered strategy arm.")
        try:
            from training.eval.provenance import seed_plan_sha256
            from training.ppo.g008_benchmark import load_principal_config
            from training.ppo.protocol import digest_json
            from training.ppo.train import TrainConfig, g008_campaign_schedule_descriptor

            if campaign["seed_plan_sha256"] != seed_plan_sha256():
                raise ValueError("campaign seed plan hash differs from the frozen seed plan")
            frozen_config, _ = load_principal_config()
            expected_config = replace(frozen_config, device="auto", seed=campaign["ppo_seed"],
                                      total_updates=1_000, checkpoint_every=50,
                                      snapshot_opponent_probability=(0.1 if campaign["strategy_role"] == "mixed" else 0.0),
                                      heuristic_opponent_probability=(1 / 3 if campaign["strategy_role"] == "mixed" else 0.0))
            train_config = TrainConfig(**{**config, "device": "auto"})
            train_config.validate()
            if train_config != expected_config:
                raise ValueError("TrainConfig differs from the frozen v2 training baseline")
            if digest_json(train_config.__dict__, "") != campaign["config_sha256"]:
                raise ValueError("full TrainConfig hash differs from the frozen v2 workload identity")
            shared = {name: value for name, value in train_config.__dict__.items()
                      if name not in {"device", "seed", "snapshot_opponent_probability",
                                      "heuristic_opponent_probability"}}
            if digest_json(shared, "") != campaign["shared_config_sha256"]:
                raise ValueError("shared TrainConfig hash differs from the frozen v2 paired config")
            descriptor = g008_campaign_schedule_descriptor(
                run_uuid=campaign["run_uuid"], root_seed=campaign["root_seed"],
                arena_count=train_config.arena_count, total_updates=train_config.total_updates,
                rollout_steps=train_config.rollout_steps, seed_plan_sha=campaign["seed_plan_sha256"])
            if descriptor["schedule_descriptor_sha256"] != campaign["schedule_descriptor_sha256"]:
                raise ValueError("compact environment schedule descriptor hash differs from registration")
        except (ImportError, TypeError, ValueError) as exc:
            raise KaggleError(f"Invalid preregistered G008 v2 campaign training config: {exc}") from exc
        settings_out["g008_campaign_train"] = campaign
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
    campaign_id = (settings.get("g008_campaign_train", {}).get("campaign_id")
                   or settings.get("g008_evaluation", {}).get("campaign_id"))
    evaluation_mode = settings.get("g008_evaluation", {}).get("mode")
    v2_nonfinal_package = campaign_id == "g008-competitive-20261021-v2" and evaluation_mode != "final"
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.prepare-", dir=destination.parent))
    try:
        excluded_training_files = {
            "training/configs/g008-split-seed-plan.json",
            "training/eval/provenance.py",
            "training/eval/registry.py",
            "training/eval/lc_eval.py",
            "training/eval/g008_final_roster.py",
            "training/eval/g008_final_opponent.py",
            "training/README.md",
            "training/ppo/g008_benchmark.py",
        }

        def ignore_training(directory: str, names: list[str]) -> set[str]:
            ignored = {name for name in names
                       if name in {"kaggle", "__pycache__", ".pytest_cache"}
                       or name.startswith(".kaggle.prepare-")}
            if not v2_nonfinal_package:
                return ignored
            base = Path(directory).resolve()
            relative_dir = base.relative_to((repo / "training").resolve()).as_posix()
            for name in names:
                relative_file = (Path(relative_dir) / name).as_posix()
                if (f"training/{relative_file}" in excluded_training_files
                        or relative_file == "ppo/g008_incumbent"):
                    ignored.add(name)
            return ignored

        shutil.copytree(repo / "training", stage / "training", ignore=ignore_training)
        if ("g008_evaluation" in settings and settings["g008_evaluation"]["mode"] == "validation"
                and not v2_nonfinal_package):
            opponent_source = repo / "artifacts/circuit-ai/runs/lc-circuit-ai-selfplay-300-viewer-ready/checkpoints/step-300.pt"
            if not opponent_source.is_file():
                raise KaggleError("Frozen G007 step-300 opponent checkpoint is missing from the local source tree.")
            opponent_asset = stage / "training/eval/assets/g007-step-300.pt"
            opponent_asset.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(opponent_source, opponent_asset)
        if v2_nonfinal_package:
            def ignore_schemas(directory: str, names: list[str]) -> set[str]:
                ignored = {name for name in names if name in {"__pycache__", ".pytest_cache"}}
                if Path(directory).resolve() == (repo / "schemas").resolve():
                    ignored.add("g008-split-seed-plan-3.schema.json")
                return ignored
            shutil.copytree(repo / "schemas", stage / "schemas", ignore=ignore_schemas)
        else:
            shutil.copytree(repo / "schemas", stage / "schemas",
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        if not v2_nonfinal_package:
            shutil.copy2(repo / "training/README.md", stage / "training/README.md")
        for relative in ("training/requirements-kaggle.txt",):
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
            # Both frozen G008 workloads use a single CUDA device on the T4;
            # ordinary kernels retain Kaggle's default accelerator shape.
            "machine_shape": "NvidiaTeslaT4" if any(
                key in settings for key in ("g008_principal_benchmark", "g008_train_schedule",
                                            "g008_campaign_train", "g008_evaluation")) else "",
            "dataset_sources": ([settings["g008_evaluation"]["candidate_dataset_source"]]
                                if "g008_evaluation" in settings else []),
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
    }
    if "train_config" in settings:
        settings["train_config"]["device"] = "cuda"
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
            "visible_cuda_device_count": len(device_names),
            "training_device": device_names[0] if device_names else None,
            "training_device_index": 0, "training_cuda_device_count": 1,
            "benchmark_report_sha256": file_sha256(report_path),
            "benchmark_report": report,
        }
        (output / "kaggle-execution-evidence.json").write_text(
            json.dumps(evidence, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
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
            json.dumps(runtime_evidence, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
        print(f"Completed G008 train-only workload {run_id}; artifacts: {output}")
        return
    g008_campaign_train = settings.get("g008_campaign_train")
    if g008_campaign_train is not None:
        report = train(config, output, export_onnx=True, run_evaluation=False,
                       g008_campaign_train=g008_campaign_train,
                       run_id=g008_campaign_train["run_uuid"])
        runtime_evidence["run_id"] = g008_campaign_train["run_uuid"]
        (output / "kaggle-execution-evidence.json").write_text(
            json.dumps(runtime_evidence, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
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


def register_recovery_cycle2_workloads(registry, amendment: dict, workloads: Sequence[dict]) -> list[dict]:
    """Append the separately reviewed and user-authorized Cycle-2 amendment plus its two workloads."""
    if not isinstance(workloads, Sequence) or isinstance(workloads, (str, bytes)) or len(workloads) != 2:
        raise KaggleError("Cycle 2 requires exactly its approved benchmark and training workload records.")
    if not isinstance(amendment, dict) or amendment.get("cycle_id") != "g008-recovery-cycle-2":
        raise KaggleError("Cycle-2 amendment must bind the exact reviewed cycle ID.")
    authorization = amendment.get("user_authorization")
    if (not isinstance(authorization, dict)
            or authorization.get("proposal_sha256") != amendment.get("proposal_sha256")
            or not isinstance(authorization.get("statement"), str) or not authorization["statement"].strip()
            or not isinstance(authorization.get("confirmed_at_utc"), str)
            or not authorization["confirmed_at_utc"].endswith("Z")):
        raise KaggleError("Cycle-2 registration requires durable user authorization bound to the proposal hash.")
    named = amendment.get("recovery_registrations")
    if not isinstance(named, list) or len(named) != 2:
        raise KaggleError("Cycle-2 amendment must name exactly two workloads before registry mutation.")
    workload_map = {item.get("registration_id"): item for item in workloads if isinstance(item, dict)}
    expected_map = {item.get("registration_id"): item for item in named if isinstance(item, dict)}
    if len(workload_map) != 2 or workload_map.keys() != expected_map.keys():
        raise KaggleError("Cycle-2 workload records differ from the amendment's exact registration IDs.")
    for registration_id, expected in expected_map.items():
        item = workload_map[registration_id]
        kind, record = item.get("registration_kind"), item.get("record")
        expected_timeout = 2800 if kind == "benchmark" else 7200 if kind == "train" else None
        expected_cap = 5600 if kind == "benchmark" else 7200 if kind == "train" else None
        expected_output = 100_000_000 if kind == "benchmark" else 1_000_000_000 if kind == "train" else None
        if (kind != expected.get("registration_kind") or not isinstance(record, dict)
                or record.get("cycle_id") != amendment["cycle_id"]
                or record.get("session_timeout_seconds") != expected_timeout
                or record.get("workload_gpu_seconds_cap") != expected_cap
                or record.get("workload_output_bytes_cap") != expected_output
                or record.get("kernel_id") != amendment.get("kernel_id")
                or record.get("accelerator_count") != 1
                or record.get("machine_shape") != "NvidiaTeslaT4"):
            raise KaggleError("Cycle-2 workload record differs from its frozen kind, package budget, or kernel identity.")
        for key in ("package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
                    "backend_sha256", "trainer_sha256", "engine_sha256"):
            if not isinstance(record.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", record[key]) is None:
                raise KaggleError(f"Cycle-2 registration has no exact frozen {key}.")
        if record.get("run_uuid") != expected.get("run_uuid"):
            raise KaggleError("Cycle-2 workload record differs from its amendment-bound run UUID.")
    ordered_workloads = sorted(workloads, key=lambda item: 0 if item["registration_kind"] == "benchmark" else 1)
    try:
        amendment_event = registry.register_recovery_cycle_amendment(amendment)
        events = []
        for item in ordered_workloads:
            events.append(registry.register_recovery_cycle_workload(
                item["registration_id"], item["registration_kind"], item["record"],
            ))
        return [amendment_event, *events]
    except Exception as exc:
        raise KaggleError(f"Cycle-2 registry append failed; inspect the append-only head before continuing: {exc}") from exc


def register_recovery_cycle3_workloads(registry, amendment: dict, workloads: Sequence[dict], *, api=None) -> list[dict]:
    """Bind authorization to frozen C3 records, then append amendment and derived-pointer registrations."""
    if not isinstance(workloads, Sequence) or isinstance(workloads, (str, bytes)) or len(workloads) != 2:
        raise KaggleError("Cycle 3 requires exactly its approved benchmark and training workload records.")
    if not isinstance(amendment, dict) or amendment.get("cycle_id") != "g008-recovery-cycle-3":
        raise KaggleError("Cycle-3 amendment must bind the exact reviewed cycle ID.")
    try:
        from training.eval.registry import (
            cycle3_authorization_statement,
            cycle3_preflight_context_sha256,
            cycle3_registration_bindings_sha256,
            cycle3_registration_record_sha256,
        )
    except ImportError as exc:
        raise KaggleError("Cycle-3 registration binding helpers are unavailable.") from exc

    named = amendment.get("recovery_registrations")
    if not isinstance(named, list) or len(named) != 2:
        raise KaggleError("Cycle-3 amendment must name exactly two records before registry mutation.")
    expected_kinds = ["benchmark", "train"]
    if [row.get("registration_kind") if isinstance(row, dict) else None for row in named] != expected_kinds:
        raise KaggleError("Cycle-3 amendment bindings must order benchmark then train.")
    workload_map = {item.get("registration_id"): item for item in workloads if isinstance(item, dict)}
    expected_map = {row.get("registration_id"): row for row in named if isinstance(row, dict)}
    if len(workload_map) != 2 or workload_map.keys() != expected_map.keys():
        raise KaggleError("Cycle-3 workloads differ from the amendment's exact registration IDs.")

    ordered_workloads = []
    projected_rows = []
    for expected in named:
        item = workload_map[expected["registration_id"]]
        kind, record = item.get("registration_kind"), item.get("record")
        if kind != expected["registration_kind"] or not isinstance(record, dict):
            raise KaggleError("Cycle-3 workload kind or record differs from its authorized binding.")
        if ("recovery_cycle_amendment_event_index" in record
                or "recovery_cycle_amendment_event_hash" in record):
            raise KaggleError("Cycle-3 records must be frozen before adding derived amendment pointers.")
        if record.get("cycle_id") != "g008-recovery-cycle-3" or record.get("run_uuid") != expected.get("run_uuid"):
            raise KaggleError("Cycle-3 workload record differs from its authorized cycle or run UUID.")
        record_hash = cycle3_registration_record_sha256(record)
        if expected.get("record_sha256") != record_hash:
            raise KaggleError("Cycle-3 workload record hash differs from the user-authorized binding.")
        projected_rows.append({key: expected[key] for key in (
            "registration_id", "registration_kind", "run_uuid", "record_sha256")})
        ordered_workloads.append(item)

    proposal_hash = amendment.get("proposal_sha256")
    bindings_hash = cycle3_registration_bindings_sha256(projected_rows)
    preflight_context_hash = cycle3_preflight_context_sha256(amendment)
    authorization = amendment.get("user_authorization")
    if (amendment.get("registration_bindings_sha256") != bindings_hash
            or amendment.get("preflight_context_sha256") != preflight_context_hash
            or not isinstance(authorization, dict)
            or authorization.get("proposal_sha256") != proposal_hash
            or authorization.get("registration_bindings_sha256") != bindings_hash
            or authorization.get("preflight_context_sha256") != preflight_context_hash
            or authorization.get("statement") != cycle3_authorization_statement(
                proposal_hash, bindings_hash, preflight_context_hash)
            or not isinstance(authorization.get("confirmed_at_utc"), str)
            or not authorization["confirmed_at_utc"].endswith("Z")):
        raise KaggleError("Cycle-3 authorization must bind the exact proposal and ordered workload records.")

    if api is None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        except (ImportError, AttributeError) as exc:
            raise KaggleError("Pinned Kaggle SDK is required to verify the live Cycle-3 preflight context.") from exc
    kernel_id = _kernel_id(amendment.get("kernel_id"))
    try:
        before = _sdk_metadata(api, kernel_id)
        live_status, _ = _sdk_status(api, kernel_id)
        after = _sdk_metadata(api, kernel_id)
    except KaggleError as exc:
        raise KaggleError("Live Cycle-3 kernel identity/status could not be verified; no registry events were appended.") from exc
    if before != after or live_status not in {"COMPLETE", "ERROR", "CANCELLED", "TIMEOUT"}:
        raise KaggleError("Live Cycle-3 kernel is changing or nonterminal; no registry events were appended.")
    quota_remaining, quota_used = _quota_seconds(api)
    live_context = {
        "kernel_id": kernel_id,
        "current_version_number": after["current_version_number"],
        "current_terminal_status": live_status,
        "current_status_evidence": {
            "status": live_status,
            "current_version_number": after["current_version_number"],
            "kernel_id": kernel_id,
        },
        "quota_gpu_seconds_remaining_before": quota_remaining,
        "quota_evidence": {
            "remaining_gpu_seconds": quota_remaining,
            "used_gpu_seconds": quota_used,
            "source": "pinned Kaggle SDK quota_view",
        },
    }
    if cycle3_preflight_context_sha256(live_context) != preflight_context_hash:
        raise KaggleError("Live kernel/quota context differs from the user-authorized Cycle-3 snapshot; no registry events were appended.")

    try:
        batch_workloads = [{"registration_id": item["registration_id"],
                            "registration_kind": item["registration_kind"],
                            "record": dict(item["record"])} for item in ordered_workloads]
        return registry.register_recovery_cycle3_batch(amendment, batch_workloads)
    except Exception as exc:
        raise KaggleError(f"Cycle-3 registry append failed; inspect the append-only head before continuing: {exc}") from exc


def register_recovery_cycle3_retry(registry, amendment: dict, workload: dict, *, api=None) -> list[dict]:
    """Verify exact retry authorization and live Kaggle context before the atomic C3 continuation append."""
    if (not isinstance(amendment, dict) or not isinstance(workload, dict)
            or set(workload) != {"registration_id", "registration_kind", "record"}
            or workload.get("registration_kind") != "benchmark"
            or not isinstance(workload.get("record"), dict)):
        raise KaggleError("Cycle-3 retry requires one exact benchmark workload record.")
    try:
        from training.eval import registry as registry_module
        from training.eval.registry import (
            cycle3_retry_authorization_statement,
            cycle3_retry_binding_sha256,
            cycle3_retry_context_sha256,
            cycle3_retry_registration_record_sha256,
        )
    except ImportError as exc:
        raise KaggleError("Cycle-3 retry authorization helpers are unavailable.") from exc

    record = workload["record"]
    registration_id = workload["registration_id"]
    run_uuid = record.get("run_uuid")
    record_sha = cycle3_retry_registration_record_sha256(record)
    context_sha = cycle3_retry_context_sha256(amendment)
    binding_sha = cycle3_retry_binding_sha256(registration_id, "benchmark", run_uuid, record_sha)
    authorization = amendment.get("user_authorization")
    proposal_sha = amendment.get("proposal_sha256")
    if (amendment.get("schema_id") != "g008-kaggle-recovery-cycle-retry-amendment"
            or amendment.get("schema_version") != 1
            or amendment.get("cycle_id") != "g008-recovery-cycle-3-retry"
            or proposal_sha != registry_module._CYCLE3_RETRY_PROPOSAL_SHA256
            or amendment.get("retry_registration_id") != registration_id
            or amendment.get("registration_kind") != "benchmark"
            or amendment.get("retry_run_uuid") != run_uuid
            or amendment.get("retry_record_sha256") != record_sha
            or amendment.get("registration_binding_sha256") != binding_sha
            or amendment.get("preflight_context_sha256") != context_sha
            or record.get("cycle_id") != "g008-recovery-cycle-3"
            or record.get("kernel_id") != amendment.get("kernel_id")
            or any(key in record for key in (
                "cycle3_retry_amendment_event_index", "cycle3_retry_amendment_event_hash",
                "recovery_cycle_amendment_event_index", "recovery_cycle_amendment_event_hash"))
            or not isinstance(authorization, dict)
            or authorization.get("proposal_sha256") != proposal_sha
            or authorization.get("record_sha256") != record_sha
            or authorization.get("preflight_context_sha256") != context_sha
            or authorization.get("statement") != cycle3_retry_authorization_statement(
                proposal_sha, record_sha, context_sha)
            or not isinstance(authorization.get("confirmed_at_utc"), str)
            or not authorization["confirmed_at_utc"].endswith("Z")):
        raise KaggleError("Cycle-3 retry authorization does not bind the exact proposal, record, and live context.")

    events = registry.events()
    if (len(events) != 34 or events[-1].get("event_index") != 33
            or events[-1].get("event_hash") != amendment.get("prior_head_hash")
            or amendment.get("prior_final_event_index") != 33
            or events[-1].get("event_hash") != registry_module._CYCLE3_RETRY_PARENT_EVENT_HASH
            or amendment.get("cycle3_amendment_event_index") != 27
            or amendment.get("cycle3_amendment_event_hash") != events[27].get("event_hash")):
        raise KaggleError("Cycle-3 retry requires the exact immutable registry head through failed event 33.")
    failed = events[33].get("event_data", {})
    if (failed.get("outcome") != "terminal" or failed.get("terminal_status") not in {"ERROR", "CANCELLED", "TIMEOUT"}
            or failed.get("assigned_version") != amendment.get("current_version_number")
            or failed.get("terminal_metadata", {}).get("kernel_id") != amendment.get("kernel_id")
            or failed.get("terminal_status") != amendment.get("current_terminal_status")):
        raise KaggleError("Cycle-3 retry amendment does not bind the reconciled terminal benchmark failure.")

    if api is None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        except (ImportError, AttributeError) as exc:
            raise KaggleError("Pinned Kaggle SDK is required to verify the retry preflight context.") from exc
    kernel_id = _kernel_id(amendment.get("kernel_id"))
    try:
        before = _sdk_metadata(api, kernel_id)
        live_status, _ = _sdk_status(api, kernel_id)
        after = _sdk_metadata(api, kernel_id)
        quota_remaining, quota_used = _quota_seconds(api)
    except KaggleError as exc:
        raise KaggleError("Live retry kernel identity/status/quota could not be verified; no events were appended.") from exc
    if before != after or live_status not in {"COMPLETE", "ERROR", "CANCELLED", "TIMEOUT"}:
        raise KaggleError("Live retry kernel is changing or nonterminal; no events were appended.")
    live_context = {
        "kernel_id": kernel_id,
        "current_version_number": after["current_version_number"],
        "current_terminal_status": live_status,
        "current_status_evidence": {
            "status": live_status,
            "current_version_number": after["current_version_number"],
            "kernel_id": kernel_id,
        },
        "quota_gpu_seconds_remaining_before": quota_remaining,
        "quota_evidence": {
            "remaining_gpu_seconds": quota_remaining,
            "used_gpu_seconds": quota_used,
            "source": "pinned Kaggle SDK quota_view",
        },
    }
    if cycle3_retry_context_sha256(live_context) != context_sha:
        raise KaggleError("Live retry status/quota differs from the user-authorized context; no events were appended.")
    try:
        return registry.register_recovery_cycle3_retry_batch(amendment, workload)
    except Exception as exc:
        raise KaggleError(f"Cycle-3 retry registry append failed; inspect the append-only head before continuing: {exc}") from exc


def register_recovery_benchmark_artifact(registry, attempt_event: dict,
                                         artifact_root: Path, envelope_path: Path) -> dict:
    """Write and register an immutable post-download envelope for a successful /3 attempt."""
    if (not isinstance(attempt_event, dict) or attempt_event.get("event_type") != "workload_attempt"
            or attempt_event.get("event_data", {}).get("outcome") != "terminal"
            or attempt_event.get("event_data", {}).get("terminal_status") != "COMPLETE"):
        raise KaggleError("Recovery benchmark artifact requires a successful terminal attempt event.")
    data = attempt_event["event_data"]
    artifact_root, envelope_path = Path(artifact_root), Path(envelope_path)
    report_paths = sorted(artifact_root.rglob("g008-principal-benchmark.json"))
    if len(report_paths) != 1:
        raise KaggleError("Downloaded recovery benchmark must contain exactly one principal report artifact directory.")
    report_path = report_paths[0]
    artifact_dir = report_path.parent
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_hash = file_sha256(report_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise KaggleError("Recovery benchmark report or artifact is missing/invalid.") from exc
    run_id = None
    try:
        evidence = json.loads((artifact_dir / "kaggle-execution-evidence.json").read_text(encoding="utf-8"))
        run_id = evidence.get("run_id") if isinstance(evidence, dict) else None
        validate_principal_benchmark_artifacts(artifact_dir, run_id)
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
            with api.build_kaggle_client() as kaggle:
                response = kaggle.kernels.kernels_api_client.get_kernel(request)
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


def _status_is_waiting_for_worker(status_item: dict) -> bool:
    """Kaggle queue states do not consume the registered workload session budget."""
    return status_item.get("status") in {"QUEUED", "PENDING", "STARTING", "NEW_SCRIPT"}


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


_CAMPAIGN_V1 = "g008-competitive-campaign-v1"
_CAMPAIGN_V2 = "g008-competitive-campaign-v2"
_CAMPAIGN_CYCLES = {_CAMPAIGN_V1, _CAMPAIGN_V2}
_CAMPAIGN_IDS = {"g008-competitive-20261020-v1", "g008-competitive-20261021-v2"}


def _is_competitive_campaign(cycle_id: str | None) -> bool:
    return cycle_id in _CAMPAIGN_CYCLES


def _recovery_record(registration_event: dict, variant_id: str,
                     events: Sequence[dict] | None = None) -> tuple[dict, dict]:
    if registration_event.get("event_type") != "registration":
        raise KaggleError("Recovery workload must be an exact prior registration event.")
    data = registration_event.get("event_data", {})
    record = data.get("record")
    kind = data.get("registration_kind")
    cycle_id = data.get("cycle_id")
    campaign = _is_competitive_campaign(cycle_id)
    allowed_kinds = ({"train", "validation", "final"} if cycle_id == _CAMPAIGN_V2 else
                     {"benchmark", "train", "validation", "final"} if campaign else {"benchmark", "train"})
    if not isinstance(record, dict) or kind not in allowed_kinds:
        raise KaggleError("Recovery registration must bind a supported, exact workload.")
    maximum = record.get("workload_gpu_seconds_cap", 3600 if kind == "benchmark" else 7200)
    timeout = record.get("session_timeout_seconds")
    if type(timeout) is not int or not 0 < timeout <= maximum:
        raise KaggleError("Registered Kaggle timeout must be a positive integer within the workload cap.")
    if record.get("accelerator_count") != 1 or record.get("machine_shape") != "NvidiaTeslaT4":
        raise KaggleError("Registered recovery workload must use exactly one Tesla T4.")
    if campaign:
        expected_variant_id = ("paired" if kind == "benchmark" else
                               record.get("strategy_role") if kind == "train" else "frozen-candidate")
        if variant_id != expected_variant_id:
            raise KaggleError(f"Campaign workload requires its exact registered implementation variant: {expected_variant_id}.")
        variant = record
    else:
        variants = record.get("implementation_variants" if kind == "benchmark" else "allowed_implementation_variants")
        expected_variant_id = "paired" if kind == "benchmark" else variant_id
        variant = next((item for item in variants or [] if item.get("variant_id") == expected_variant_id), None)
        if kind == "train" and expected_variant_id not in {"incumbent", "candidate"}:
            raise KaggleError("Training workload requires a pre-registered incumbent or candidate variant.")
        if kind == "benchmark":
            variant = record
    if not isinstance(variant, dict):
        raise KaggleError("Selected implementation variant is not frozen in the registration.")
    hashes = {key: variant[key] for key in (
        "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
        "backend_sha256", "trainer_sha256", "engine_sha256",
    )}
    if cycle_id == _CAMPAIGN_V2 and kind in {"train", "validation", "final"} and events is not None:
        if kind == "final":
            revisions = [event for event in events
                         if event.get("event_type") == "g008_competitive_campaign_v2_final_package_revision_v1"
                         and event.get("event_data", {}).get("campaign_id") == record.get("campaign_id")]
            if len(revisions) > 1:
                raise KaggleError("Campaign v2 final package has more than one rebinding event.")
            if revisions:
                data = revisions[0]["event_data"]
                binding = data.get("package_binding")
                freeze = next((event for event in events
                               if event.get("event_type") == "competitive_candidate_freeze_v2"
                               and event.get("event_data", {}).get("campaign_id") == record.get("campaign_id")), None)
                candidate_train = next((event for event in events
                                        if event.get("event_index") == data.get("candidate_train_attempt_event_index")), None)
                if (not isinstance(binding, dict) or freeze is None
                        or data.get("candidate_freeze_event_index") != freeze.get("event_index")
                        or data.get("candidate_freeze_event_hash") != freeze.get("event_hash")
                        or data.get("candidate_checkpoint_sha256") != freeze["event_data"].get("checkpoint_sha256")
                        or data.get("candidate_train_attempt_event_index") != freeze["event_data"].get("train_attempt_event_index")
                        or data.get("candidate_train_attempt_event_hash") != freeze["event_data"].get("train_attempt_event_hash")
                        or candidate_train is None
                        or candidate_train.get("event_hash") != data.get("candidate_train_attempt_event_hash")
                        or data.get("candidate_trainer_sha256") != candidate_train.get("event_data", {}).get("trainer_sha256")
                        or binding.get("trainer_sha256") != data.get("candidate_trainer_sha256")
                        or binding.get("registration_event_index") != registration_event.get("event_index")
                        or binding.get("registration_event_hash") != registration_event.get("event_hash")
                        or data.get("registration_event_index") != registration_event.get("event_index")
                        or data.get("registration_event_hash") != registration_event.get("event_hash")
                        or data.get("original_package_sha256") != hashes["package_sha256"]
                        or data.get("original_source_sha256") != hashes["source_sha256"]
                        or any(binding.get(key) != hashes[key] for key in hashes
                               if key not in {"package_sha256", "source_sha256", "trainer_sha256"})):
                    raise KaggleError("Campaign v2 final package revision does not preserve the frozen row, candidate, and configuration.")
                hashes["package_sha256"] = binding["package_sha256"]
                hashes["source_sha256"] = binding["source_sha256"]
                hashes["trainer_sha256"] = binding["trainer_sha256"]
        else:
            allowed_revisions = ({"g008_competitive_campaign_v2_package_revision_v4",
                                  "g008_competitive_campaign_v2_package_revision_v5"}
                                 if kind == "validation" else {
                                     "g008_competitive_campaign_v2_package_revision_v1",
                                     "g008_competitive_campaign_v2_package_revision_v2",
                                     "g008_competitive_campaign_v2_package_revision_v3"})
            revisions = [event for event in events
                         if event.get("event_type") in allowed_revisions
                         and event.get("event_data", {}).get("campaign_id") == record.get("campaign_id")]
            for revision in revisions:
                data = revision.get("event_data", {})
                if revision["event_type"].endswith("_v1"):
                    binding = next((item for item in data.get("package_bindings", [])
                                    if item.get("registration_event_index") == registration_event.get("event_index")), None)
                    if (binding is None or binding.get("registration_event_hash") != registration_event.get("event_hash")
                            or any(binding.get(key) != hashes[key] for key in hashes if key != "package_sha256")):
                        raise KaggleError("Campaign v2 package revision is not bound to this exact registration and component hashes.")
                else:
                    if revision["event_type"].endswith("_v2"):
                        if data.get("failed_registration_event_index") != registration_event.get("event_index"):
                            continue
                        binding = data.get("package_binding")
                    else:
                        binding = next((item for item in data.get("package_bindings", [])
                                        if item.get("registration_event_index") == registration_event.get("event_index")), None)
                        if binding is None:
                            continue
                    if (not isinstance(binding, dict)
                            or binding.get("registration_event_hash") != registration_event.get("event_hash")
                            or any(binding.get(key, hashes[key]) != hashes[key]
                                   for key in hashes if key not in {"package_sha256", "source_sha256", "trainer_sha256"})):
                        raise KaggleError("Campaign v2 package repair changed this registration's frozen configuration.")
                for key in ("package_sha256", "source_sha256", "trainer_sha256"):
                    if key in binding:
                        hashes[key] = binding[key]
    return record, hashes


def _cycle3_amendment(events: list[dict], registration_event: dict) -> dict:
    """Resolve the exact C3 amendment pinned by a workload registration."""
    data = registration_event.get("event_data", {})
    record = data.get("record", {})
    if data.get("cycle_id") != "g008-recovery-cycle-3" or record.get("cycle_id") != data.get("cycle_id"):
        raise KaggleError("Cycle-3 recovery requires an exact current-cycle registration.")
    index = record.get("recovery_cycle_amendment_event_index")
    expected_hash = record.get("recovery_cycle_amendment_event_hash")
    if type(index) is not int or not 0 <= index < len(events):
        raise KaggleError("Cycle-3 registration does not pin an exact prior amendment event.")
    amendment_event = events[index]
    if (amendment_event.get("event_type") != "recovery_cycle_amendment_v2"
            or amendment_event.get("event_hash") != expected_hash
            or amendment_event.get("event_data", {}).get("cycle_id") != data.get("cycle_id")
            or amendment_event.get("event_data", {}).get("schema_id") != "g008-kaggle-recovery-cycle-amendment"
            or amendment_event.get("event_data", {}).get("schema_version") != 2):
        raise KaggleError("Cycle-3 registration amendment identity does not match the verified registry chain.")
    return amendment_event["event_data"]


def _campaign_amendment(events: list[dict], registration_event: dict) -> dict:
    """Resolve the exact immutable campaign amendment named by a registration."""
    data = registration_event.get("event_data", {})
    record = data.get("record", {})
    cycle_id = data.get("cycle_id")
    pointer = "competitive_campaign_v2_event" if cycle_id == _CAMPAIGN_V2 else "competitive_campaign_event"
    index, expected_hash = record.get(f"{pointer}_index"), record.get(f"{pointer}_hash")
    if (not _is_competitive_campaign(cycle_id) or type(index) is not int
            or index < 0 or index >= len(events)):
        raise KaggleError("Campaign workload does not pin an exact campaign amendment.")
    event = events[index]
    amendment = event.get("event_data", {})
    expected_type = ("g008_competitive_campaign_amendment_v2" if cycle_id == _CAMPAIGN_V2
                     else "g008_competitive_campaign_amendment_v1")
    frozen_records = amendment.get("records")
    expected_count = 13 if cycle_id == _CAMPAIGN_V2 else 9
    registration_count = len(frozen_records) if isinstance(frozen_records, list) else -1
    if registration_count != expected_count:
        raise KaggleError(f"Campaign amendment must freeze exactly {expected_count} workload registrations.")
    if (event.get("event_type") != expected_type
            or event.get("event_hash") != expected_hash
            or amendment.get("campaign_id") != record.get("campaign_id")
            or (cycle_id == _CAMPAIGN_V2
                and (amendment.get("schema_id") != "g008-competitive-campaign"
                     or amendment.get("schema_version") != 2))
            or (cycle_id == _CAMPAIGN_V2 and amendment.get("campaign_id") != "g008-competitive-20261021-v2")):
        raise KaggleError("Campaign registration amendment identity differs from the verified registry chain.")
    registrations = events[index + 1:index + registration_count + 1]
    if (len(registrations) != registration_count
            or any(item.get("event_type") != "registration" for item in registrations)
            or registration_event.get("event_index") not in range(index + 1, index + registration_count + 1)):
        raise KaggleError(f"Campaign workload is outside its exact {registration_count}-registration block.")
    if cycle_id == _CAMPAIGN_V2:
        _validate_campaign_v2_benchmark_lineage(events, event, amendment)
    return amendment


def _validate_campaign_v2_benchmark_lineage(events: list[dict], amendment_event: dict,
                                            amendment: dict) -> None:
    """Require v2 to reuse the exact attested v1 benchmark without launching another one."""
    pointers = (
        ("benchmark_artifact_event_index", "benchmark_artifact_event_hash", "benchmark_artifact"),
        ("benchmark_lock_event_index", "benchmark_lock_event_hash", "registration_closed"),
        ("benchmark_attestation_event_index", "benchmark_attestation_event_hash", "benchmark_attestation"),
    )
    for expected_index, (index_key, hash_key, event_type) in zip((48, 49, 50), pointers):
        index, digest = amendment.get(index_key), amendment.get(hash_key)
        if (type(index) is not int or index != expected_index or not 0 <= index < amendment_event.get("event_index", -1)
                or events[index].get("event_type") != event_type
                or events[index].get("event_hash") != digest):
            raise KaggleError("Campaign v2 must bind the exact existing benchmark artifact, lock, and attestation.")
    head_index = amendment.get("prior_final_event_index")
    if (type(head_index) is not int or head_index != amendment_event.get("event_index") - 1
            or amendment.get("prior_head_hash") != events[head_index].get("event_hash")):
        raise KaggleError("Campaign v2 must append at its exact prior registry head.")
    registration_index = amendment.get("selected_training_registration_event_index")
    attempt_index = amendment.get("selected_training_attempt_event_index")
    if (type(registration_index) is not int or type(attempt_index) is not int
            or not 0 <= registration_index < head_index or not 0 <= attempt_index <= head_index):
        raise KaggleError("Campaign v2 must bind the selected training registration and attempt.")
    registration = events[registration_index]
    attempt = events[attempt_index]
    registration_data = registration.get("event_data", {})
    attempt_data = attempt.get("event_data", {})
    if (registration.get("event_type") != "registration"
            or registration.get("event_hash") != amendment.get("selected_training_registration_event_hash")
            or registration_data.get("cycle_id") != "g008-recovery-cycle-3"
            or registration_data.get("registration_kind") != "train"
            or not any(item.get("variant_id") == "incumbent"
                       for item in registration_data.get("record", {}).get("allowed_implementation_variants", []))
            or registration.get("event_index") != 29
            or attempt.get("event_type") != "workload_attempt"
            or attempt.get("event_hash") != amendment.get("selected_training_attempt_event_hash")
            or attempt_data.get("registration_event_index") != registration_index
            or attempt_data.get("registration_event_hash") != registration.get("event_hash")
            or attempt_data.get("implementation_variant_id") != "incumbent"
            or attempt_data.get("terminal_status") != "COMPLETE"
            or attempt_data.get("assigned_version") != 17
            or attempt.get("event_index") != 54
            or registration_data.get("record", {}).get("run_uuid") != amendment.get("selected_training_run_uuid")
            or amendment.get("current_version_number") != 17):
        raise KaggleError("Campaign v2 selected incumbent is not the exact registered COMPLETE v17 training run.")
    rows = amendment.get("records", [])
    if (len(rows) != 13
            or any("open-space" in json.dumps(row.get("record", {}), sort_keys=True)
                   for row in rows[6:])):
        raise KaggleError("Campaign v2 amendment must keep all validation and final opponent identity hidden before freeze.")


def _validate_campaign_v2_phase(events: list[dict], registration_event: dict) -> None:
    """Gate v2 phases on attested training, comparative validation, and final freeze."""
    data = registration_event.get("event_data", {})
    if data.get("cycle_id") != _CAMPAIGN_V2:
        return
    amendment = _campaign_amendment(events, registration_event)
    campaign_id = amendment["campaign_id"]
    kind = data.get("registration_kind")
    training_regs = [event for event in events if event.get("event_type") == "registration"
                     and event.get("event_data", {}).get("cycle_id") == _CAMPAIGN_V2
                     and event.get("event_data", {}).get("registration_kind") == "train"]
    training_attempts = [event for event in events if event.get("event_type") == "workload_attempt"
                         and event.get("event_data", {}).get("cycle_id") == _CAMPAIGN_V2
                         and event.get("event_data", {}).get("terminal_status") == "COMPLETE"
                         and event.get("event_data", {}).get("registration_event_index") in {
                             item.get("event_index") for item in training_regs}]
    artifacts = [event for event in events if event.get("event_type") == "competitive_training_artifact_v2"
                 and event.get("event_data", {}).get("campaign_id") == campaign_id]
    if kind == "validation" and (len(training_regs) != 6 or len(training_attempts) != 6
                                  or len(artifacts) != 6):
        raise KaggleError("Campaign v2 validation requires six COMPLETE training runs with artifact attestations.")
    comparison = next((event for event in events
                       if event.get("event_type") == "competitive_strategy_comparison_v2"
                       and event.get("event_data", {}).get("campaign_id") == campaign_id), None)
    freeze = next((event for event in events
                   if event.get("event_type") == "competitive_candidate_freeze_v2"
                   and event.get("event_data", {}).get("campaign_id") == campaign_id), None)
    campaign_attempts = [event.get("event_data", {}) for event in events
                         if event.get("event_type") in {"workload_preflight", "workload_push_dispatched",
                                                         "workload_push_accepted", "workload_attempt"}
                         and event.get("event_data", {}).get("cycle_id") == _CAMPAIGN_V2]
    final_row = amendment["records"][12]
    final_registration = next((event for event in events if event.get("event_type") == "registration"
                               and event.get("event_data", {}).get("registration_id")
                               == final_row.get("registration_id")), None)
    final_index = final_registration.get("event_index") if final_registration else None
    if kind != "final" and any(item.get("registration_event_index") == final_index
                                for item in campaign_attempts):
        raise KaggleError("Campaign v2 open-space final attempt cannot run before candidate freeze.")
    if kind != "final" and any("open-space" in json.dumps(event.get("event_data", {}), sort_keys=True)
                                for event in events
                                if event.get("event_type") in {"competitive_validation_result_v2",
                                                                "competitive_strategy_comparison_v2",
                                                                "competitive_candidate_freeze_v2"}
                                and event.get("event_data", {}).get("campaign_id") == campaign_id):
        raise KaggleError("Campaign v2 selection events cannot disclose final open-space evidence.")
    if kind == "final":
        validations = [event for event in events
                       if event.get("event_type") == "competitive_validation_result_v2"
                       and event.get("event_data", {}).get("campaign_id") == campaign_id]
        if len(validations) != 6 or comparison is None or freeze is None:
            raise KaggleError("Campaign v2 final evaluation requires six per-arm results, a strategy comparison, and a frozen candidate.")


def _campaign_json(root: Path, relative: str) -> tuple[dict, Path]:
    """Read one regular campaign artifact, rejecting traversal and symlinks."""
    try:
        rel = safe_relative_path(relative)
        path = root / rel
        if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
            raise KaggleError(f"Campaign artifact is missing or unsafe: {relative}.")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise KaggleError(f"Campaign artifact is missing or invalid JSON: {relative}.") from exc
    if not isinstance(value, dict):
        raise KaggleError(f"Campaign artifact must be a JSON object: {relative}.")
    return value, path


def _validate_campaign_evaluation_output(registration_event: dict, attempt_event: dict,
                                         output_root: Path, candidate_checkpoint_sha256: str,
                                         source_root: Path) -> dict:
    """Validate the 512-game report, exact roster, manifest, and sampled replay bytes."""
    from training.ppo.protocol import digest_json

    registration = registration_event["event_data"]
    kind = registration["registration_kind"]
    record = registration["record"]
    if kind == "final":
        from training.eval.g008_final_roster import EXPECTED_FINAL_ROSTER
        expected = EXPECTED_FINAL_ROSTER
    elif kind == "validation":
        from training.eval.g008_validation import _EXPECTED_ROSTERS
        expected = _EXPECTED_ROSTERS["validation"]
    else:
        raise KaggleError("Campaign evaluation output kind must be validation or final.")
    report, report_path = _campaign_json(output_root, "g008-evaluation.json")
    manifest, manifest_path = _campaign_json(output_root, "g008-evaluation-manifest.json")
    runtime, runtime_path = _campaign_json(output_root, "kaggle-execution-evidence.json")
    run_config, run_config_path = _campaign_json(source_root, "run-config.json")
    evaluation_config = run_config.get("g008_evaluation")
    if (report.get("format") != "g008-registered-evaluation/1"
            or report.get("status") != "complete" or report.get("game_count") != 512
            or report.get("campaign_id") != record.get("campaign_id")
            or report.get("registration_kind") != kind or report.get("mode") != kind
            or report.get("run_uuid") != record.get("run_uuid")
            or report.get("roster_sha256") != record.get("roster_sha256")
            or report.get("opponent_code_sha256") != record.get("opponent_code_sha256")
            or report.get("roster_sha256") != expected["roster_sha256"]
            or report.get("opponent_code_sha256") != expected["opponent_code_sha256"]
            or report.get("opponent_id") != expected["opponent_id"]
            or report.get("seed_plan_sha256") != record.get("seed_plan_sha256")
            or report.get("candidate_checkpoint_sha256") != candidate_checkpoint_sha256
            or report.get("runtime_evidence") != runtime
            or runtime.get("format") != "circuit-ai-kaggle-runtime-evidence/1"
            or runtime.get("run_id") != report.get("run_id")
            or not isinstance(runtime.get("visible_cuda_devices"), list)
            or not runtime["visible_cuda_devices"]
            or runtime["visible_cuda_devices"][0] != "Tesla T4"
            or runtime.get("visible_cuda_device_count", len(runtime["visible_cuda_devices"]))
               != len(runtime["visible_cuda_devices"])
            or runtime.get("training_device") != "Tesla T4"
            or runtime.get("training_device_index", 0) != 0
            or runtime.get("training_cuda_device_count", 1) != 1
            or not all(isinstance(runtime.get(key), str) and runtime[key]
                       for key in ("python_version", "torch_version", "cuda_version"))
            or not isinstance(evaluation_config, dict)
            or evaluation_config.get("run_uuid") != record.get("run_uuid")
            or evaluation_config.get("roster_sha256") != record.get("roster_sha256")
            or evaluation_config.get("opponent_code_sha256") != record.get("opponent_code_sha256")
            or evaluation_config.get("registration_kind") != kind
            or report.get("run_config_sha256") != digest_json(evaluation_config, "__no_digest_field__")
            or report.get("run_config_file_sha256") != file_sha256(run_config_path)):
        raise KaggleError("Campaign evaluation report identity, roster, seed plan, or frozen checkpoint does not match registration.")
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 512:
        raise KaggleError("Campaign evaluation must contain exactly 512 episode results.")
    if any(not isinstance(item, dict) or type(item.get("root_seed")) is not int
           or type(item.get("target_seat")) is not int for item in episodes):
        raise KaggleError("Campaign evaluation contains invalid root or seat values.")
    slots = {(item["root_seed"], item["target_seat"]) for item in episodes}
    lower, upper = expected["root_range"]
    if slots != {(root, seat) for root in range(lower, upper + 1) for seat in (0, 1)}:
        raise KaggleError("Campaign evaluation episode roots/seats do not match the frozen roster.")
    if any(item.get("result") not in {"win", "draw", "loss"}
           or item.get("target_seat") not in (0, 1) for item in episodes):
        raise KaggleError("Campaign evaluation contains invalid result or seat values.")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise KaggleError("Campaign evaluation report is missing aggregate W/D/L metrics.")
    wins = sum(row.get("result") == "win" for row in episodes)
    draws = sum(row.get("result") == "draw" for row in episodes)
    losses = 512 - wins - draws
    if (summary.get("game_count"), summary.get("wins"), summary.get("draws"), summary.get("losses")) != (512, wins, draws, losses):
        raise KaggleError("Campaign evaluation aggregate W/D/L does not match the episode rows.")
    from collections import Counter
    failure_diagnostics = summary.get("failure_diagnostics")
    classified = [row for row in episodes if row.get("result") == "loss"
                  and isinstance(row.get("failure_diagnosis"), dict)]
    if (not isinstance(failure_diagnostics, dict)
            or failure_diagnostics.get("losses_classified") != losses
            or len(classified) != losses
            or failure_diagnostics.get("categories") != dict(sorted(
                Counter(row["failure_diagnosis"].get("category") for row in classified).items()))):
        raise KaggleError("Campaign evaluation lacks complete, reconciled failure classifications.")
    from training.eval.registry import (competitive_final_statistics,
                                        competitive_noninferiority_statistics)
    v2_validation = registration.get("cycle_id") == _CAMPAIGN_V2 and kind == "validation"
    expected_stats = (competitive_final_statistics(wins, draws, losses) if v2_validation or kind == "final"
                      else competitive_noninferiority_statistics(wins, draws, losses))
    if kind == "validation" and not v2_validation:
        if report.get("promotion_eligible") is not expected_stats["passed"]:
            raise KaggleError("Campaign validation report decision does not match its frozen non-inferiority rule.")
    elif kind == "final" and (report.get("promotion_eligible") is not expected_stats["passed"]
          or summary.get("competitive_target_passed") is not expected_stats["passed"]):
        raise KaggleError("Final report decision does not match the frozen 60% observed-win criterion.")

    sample_rows = report.get("representative_replays")
    if not isinstance(sample_rows, list) or len(sample_rows) > 8:
        raise KaggleError("Campaign evaluation replay sample exceeds its frozen limit.")
    expected_files = {"g008-evaluation.json": file_sha256(report_path),
                      "kaggle-execution-evidence.json": file_sha256(runtime_path)}
    seen_replays = set()
    for sample in sample_rows:
        if not isinstance(sample, dict):
            raise KaggleError("Campaign replay sample entry is malformed.")
        relative = safe_relative_path(sample.get("replay_file", ""))
        if not relative.startswith("replays/") or relative in seen_replays:
            raise KaggleError("Campaign replay sample paths must be unique files under replays/.")
        replay, replay_path = _campaign_json(output_root, relative)
        replay_sha = file_sha256(replay_path)
        _schema_validate(replay, "lc-replay-1.schema.json")
        if (replay.get("format") != "lc-replay/1"
                or replay.get("replay_sha256") != digest_json(replay, "replay_sha256")
                or sample.get("replay_sha256") != replay.get("replay_sha256")
                or replay.get("run_id") != report.get("run_id")):
            raise KaggleError("Campaign sampled replay failed its embedded or manifest-bound checksum.")
        expected_files[relative] = replay_sha
        seen_replays.add(relative)
    listed = manifest.get("files")
    if (manifest.get("format") != "g008-evaluation-artifacts/1"
            or manifest.get("run_uuid") != record.get("run_uuid")
            or manifest.get("run_config_sha256") != report.get("run_config_sha256")
            or manifest.get("run_config_file_sha256") != report.get("run_config_file_sha256")
            or manifest.get("manifest_sha256") != digest_json(manifest, "manifest_sha256")
            or not isinstance(listed, list)):
        raise KaggleError("Campaign evaluation manifest identity or digest is invalid.")
    normalized = {}
    for item in listed:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise KaggleError("Campaign evaluation manifest contains a malformed file record.")
        path = safe_relative_path(item["path"])
        if path in normalized or not re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])):
            raise KaggleError("Campaign evaluation manifest contains duplicate paths or invalid hashes.")
        normalized[path] = item["sha256"]
    if normalized != expected_files:
        raise KaggleError("Campaign evaluation manifest does not exactly bind the report and sampled replay files.")
    return {"report": report, "report_sha256": file_sha256(report_path),
            "manifest": manifest, "manifest_sha256": file_sha256(manifest_path),
            "wins": wins, "draws": draws, "losses": losses,
            "attempt_event_index": attempt_event["event_index"],
            "attempt_event_hash": attempt_event["event_hash"]}


def _validate_campaign_training_output(registration_event: dict, output_root: Path,
                                      source_root: Path, expected_package_sha256: str | None = None) -> dict:
    """Validate frozen G008 training telemetry and checkpoint metadata before freeze."""
    from training.ppo.protocol import digest_json

    reg_data = registration_event["event_data"]
    record = reg_data["record"]
    if expected_package_sha256 is not None:
        observed_package_sha256 = package_tree_sha256(source_root)
        if (observed_package_sha256 != expected_package_sha256
                or (reg_data.get("cycle_id") != _CAMPAIGN_V2
                    and expected_package_sha256 != record.get("package_sha256"))):
            raise KaggleError("G008 training artifact package does not match the exact terminal attempt hash.")
    metrics, metrics_path = _campaign_json(output_root, "training-metrics.json")
    runtime, runtime_path = _campaign_json(output_root, "kaggle-execution-evidence.json")
    run_config, _ = _campaign_json(source_root, "run-config.json")
    v2_campaign = reg_data.get("cycle_id") == _CAMPAIGN_V2
    schedule = (run_config.get("g008_campaign_train") if v2_campaign
                else run_config.get("g008_train_schedule"))
    if (metrics.get("run_id") != record.get("run_uuid")
            or not isinstance(schedule, dict)
            or schedule.get("run_uuid") != record.get("run_uuid")
            or schedule.get("root_seed") != record.get("root_seed")
            or (v2_campaign and (schedule.get("campaign_id") != record.get("campaign_id")
                                 or schedule.get("registration_id") != registration_event["event_data"].get("registration_id")
                                 or schedule.get("pair_id") != record.get("pair_id")
                                 or schedule.get("strategy_role") != record.get("strategy_role")
                                 or schedule.get("ppo_seed") != record.get("ppo_seed")
                                 or schedule.get("schedule_descriptor_sha256") != record.get("schedule_descriptor_sha256")))
            or schedule.get("seed_plan_sha256") != record.get("seed_plan_sha256")
            or not isinstance(metrics.get("device"), dict)
            or metrics["device"].get("device") != "cuda"
            or metrics["device"].get("cuda_available") is not True
            or metrics["device"].get("cuda_name") != "Tesla T4"
            or runtime.get("format") != "circuit-ai-kaggle-runtime-evidence/1"
            or runtime.get("run_id") != record.get("run_uuid")
            or not isinstance(runtime.get("visible_cuda_devices"), list)
            or not runtime["visible_cuda_devices"]
            or runtime["visible_cuda_devices"][0] != "Tesla T4"
            or runtime.get("visible_cuda_device_count", len(runtime["visible_cuda_devices"]))
               != len(runtime["visible_cuda_devices"])
            or runtime.get("training_device") != "Tesla T4"
            or runtime.get("training_device_index", 0) != 0
            or runtime.get("training_cuda_device_count", 1) != 1
            or not all(isinstance(runtime.get(key), str) and runtime[key]
                       for key in ("python_version", "torch_version", "cuda_version"))
            or metrics.get("software", {}).get("python") != runtime.get("python_version", "").split()[0]
            or metrics.get("software", {}).get("torch") != runtime.get("torch_version")
            or metrics.get("software", {}).get("cuda") != runtime.get("cuda_version")
            or metrics["device"].get("cuda_optimizer_updates", 0) <= 0):
        raise KaggleError("G008 training metrics do not prove CUDA execution for the registered run.")
    training = metrics.get("training")
    updates = training.get("updates") if isinstance(training, dict) else None
    update_budget = record.get("update_budget")
    actual_config = training.get("config") if isinstance(training, dict) else None
    expected_config = dict(run_config.get("train_config", {}))
    expected_config["device"] = "cuda"
    if (type(update_budget) is not int or not isinstance(updates, list) or len(updates) != update_budget
            or training.get("start_update") != 0 or not isinstance(actual_config, dict)
            or actual_config.get("total_updates") != update_budget
            or actual_config != expected_config):
        raise KaggleError("G008 training metrics do not match the exact registered update budget.")
    shared_config = {key: value for key, value in actual_config.items()
                     if key not in {"device", "snapshot_opponent_probability", "heuristic_opponent_probability"}}
    if (record.get("ppo_config") is not None
            and record.get("ppo_config") != _campaign_train_config_wire(shared_config)):
        raise KaggleError("G008 PPO config differs from the shared paired configuration frozen in the registry.")
    if v2_campaign:
        from training.ppo.protocol import digest_json as _digest_json

        auto_device_config = dict(actual_config)
        auto_device_config["device"] = "auto"
        registered_train_config = record.get("train_config")
        if (not isinstance(registered_train_config, dict)
                or _digest_json(auto_device_config, "") != record.get("config_sha256")
                or _campaign_train_config_wire(auto_device_config) != registered_train_config
                or _digest_json(_campaign_train_config_native(registered_train_config), "")
                   != record.get("config_sha256")):
            raise KaggleError("G008 v2 full TrainConfig differs from its registered config identity.")
    if v2_campaign and (actual_config.get("seed") != record.get("ppo_seed")
                        or actual_config.get("total_updates") != 1_000
                        or actual_config.get("checkpoint_every") != 50
                        or training.get("schedule_descriptor_sha256") != record.get("schedule_descriptor_sha256")
                        or schedule.get("schedule_descriptor_sha256") != record.get("schedule_descriptor_sha256")
                        or training.get("campaign_id") != record.get("campaign_id")
                        or training.get("pair_id") != record.get("pair_id")
                        or training.get("strategy_role") != record.get("strategy_role")):
        raise KaggleError("G008 v2 runtime schedule, seed, or training depth differs from its exact registration.")
    snapshot_probability = actual_config.get("snapshot_opponent_probability")
    heuristic_probability = actual_config.get("heuristic_opponent_probability")
    if (type(snapshot_probability) not in (int, float) or isinstance(snapshot_probability, bool)
            or type(heuristic_probability) not in (int, float) or isinstance(heuristic_probability, bool)
            or not math.isfinite(snapshot_probability) or not math.isfinite(heuristic_probability)
            or not 0 <= snapshot_probability <= 1 or not 0 <= heuristic_probability <= 1):
        raise KaggleError("G008 opponent sampling probabilities are invalid.")
    latest = (1 - snapshot_probability) * (1 - heuristic_probability)
    expected_probabilities = {
        "latest": round(latest * 10_000),
        "step0": round(snapshot_probability * 10_000),
        "cruise": round((1 - snapshot_probability) * heuristic_probability * 10_000 / 3),
        "local_mobility": round((1 - snapshot_probability) * heuristic_probability * 10_000 / 3),
        "interceptor": round((1 - snapshot_probability) * heuristic_probability * 10_000 / 3),
    }
    if record.get("opponent_probabilities") != expected_probabilities:
        raise KaggleError("G008 runtime opponent sampling differs from the frozen roster probability mix.")
    required_metrics = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                        "mean_step_reward", "rollout_seconds", "optimizer_seconds", "update_latency_seconds")
    style_metric_keys = {"cruise": "cruise", "local_mobility": "local-mobility",
                         "interceptor": "interceptor"}
    opponent_totals = {name: 0 for name in expected_probabilities}
    for expected_update, update in enumerate(updates, 1):
        if (not isinstance(update, dict) or update.get("update") != expected_update
                or update.get("device") != "cuda" or update.get("cuda_updates", 0) <= 0
                or update.get("cuda_updates") != update.get("optimizer_updates")
                or not isinstance(update.get("episodes_by_opponent"), dict)
                or type(update.get("partial_episodes_at_rollout_end")) is not int
                or any(type(update.get(key)) not in (int, float) or not math.isfinite(update[key])
                       for key in required_metrics)):
            raise KaggleError("G008 training PPO telemetry is incomplete or contains non-finite metrics.")
        if v2_campaign:
            counts = [update.get("snapshot_opponent_arenas"), update.get("latest_opponent_arenas"),
                      update.get("heuristic_opponent_arenas")]
            styles = update.get("heuristic_opponent_style_counts")
            if (any(type(value) is not int or value < 0 for value in counts)
                    or sum(counts) != actual_config.get("arena_count")
                    or not isinstance(styles, dict)
                    or set(styles) != set(style_metric_keys.values())
                    or any(type(value) is not int or value < 0 for value in styles.values())
                    or sum(styles.values()) != counts[2]
                    or (record["strategy_role"] == "baseline" and counts != [0, actual_config.get("arena_count"), 0])):
                raise KaggleError("G008 v2 per-update opponent samples do not reconcile to the frozen arm and arena count.")
            opponent_totals["step0"] += counts[0]
            opponent_totals["latest"] += counts[1]
            for name, metric_key in style_metric_keys.items():
                opponent_totals[name] += styles[metric_key]
    if v2_campaign and record["strategy_role"] == "mixed":
        total_samples = update_budget * actual_config["arena_count"]
        for name, basis_points in expected_probabilities.items():
            observed = opponent_totals[name] / total_samples
            expected = basis_points / 10_000
            if abs(observed - expected) > 0.01:
                raise KaggleError("G008 v2 opponent arena samples drift beyond the frozen 1pp mix tolerance.")
    if (metrics["device"].get("optimizer_updates") != metrics["device"].get("cuda_optimizer_updates")
            or sum(item.get("cuda_updates", 0) for item in updates) != metrics["device"].get("cuda_optimizer_updates")):
        raise KaggleError("G008 training optimizer update totals do not reconcile to CUDA telemetry.")

    step = update_budget
    checkpoint_sha = None
    files = [{"path": "training-metrics.json", "sha256": file_sha256(metrics_path)},
             {"path": "kaggle-execution-evidence.json", "sha256": file_sha256(runtime_path)}]
    for checkpoint_step in sorted({0, step}):
        checkpoint_id = f"step-{checkpoint_step}"
        checkpoint_rel = f"checkpoints/{checkpoint_id}.pt"
        onnx_rel = f"checkpoints/{checkpoint_id}.onnx"
        metadata_rel = f"checkpoints/{checkpoint_id}.metadata.json"
        checkpoint_path, onnx_path = output_root / checkpoint_rel, output_root / onnx_rel
        metadata, _ = _campaign_json(output_root, metadata_rel)
        if (checkpoint_path.is_symlink() or not checkpoint_path.is_file()
                or onnx_path.is_symlink() or not onnx_path.is_file()):
            raise KaggleError(f"Registered G008 run is missing checkpoint artifacts for {checkpoint_id}.")
        pt_sha, onnx_sha = file_sha256(checkpoint_path), file_sha256(onnx_path)
        if (metadata.get("format") != "lc-checkpoint/1" or metadata.get("checkpoint_id") != checkpoint_id
                or metadata.get("step") != checkpoint_step
                or metadata.get("pytorch", {}).get("file") != checkpoint_rel
                or metadata.get("pytorch", {}).get("sha256") != pt_sha
                or metadata.get("onnx", {}).get("file") != onnx_rel
                or metadata.get("onnx", {}).get("sha256") != onnx_sha
                or metadata.get("metadata_sha256") != digest_json(metadata, "metadata_sha256")):
            raise KaggleError(f"Registered G008 checkpoint metadata or file hashes do not reconcile for {checkpoint_id}.")
        files.extend({"path": rel, "sha256": file_sha256(output_root / rel)}
                     for rel in (checkpoint_rel, onnx_rel, metadata_rel))
        if checkpoint_step == step:
            checkpoint_sha = pt_sha
    manifest = {"format": "g008-training-artifacts/1", "campaign_id": record["campaign_id"],
                "registration_event_index": registration_event["event_index"],
                "registration_event_hash": registration_event["event_hash"],
                "run_uuid": record["run_uuid"], "update_budget": step,
                "training_metrics_sha256": file_sha256(metrics_path),
                "checkpoint_sha256": checkpoint_sha, "files": files}
    manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
    manifest_path = output_root / "g008-training-artifact-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"metrics": metrics, "metrics_sha256": file_sha256(metrics_path),
            "checkpoint_sha256": checkpoint_sha, "manifest": manifest,
            "manifest_sha256": file_sha256(manifest_path),
            "schedule_descriptor_sha256": record.get("schedule_descriptor_sha256")}


def _campaign_training_output_root(output_root: Path, run_uuid: str) -> Path:
    """Resolve Kaggle's run-scoped output directory without guessing between runs."""
    output_root = Path(output_root)
    if output_root.is_symlink() or not output_root.is_dir():
        raise KaggleError("Campaign training output root is missing or unsafe.")
    if (output_root / "training-metrics.json").is_file():
        return output_root
    arena_root = output_root / "arena-output"
    run_root = arena_root / run_uuid
    if (arena_root.is_symlink() or run_root.is_symlink() or not arena_root.is_dir()
            or not run_root.is_dir() or not (run_root / "training-metrics.json").is_file()):
        raise KaggleError("Campaign training output is missing the exact registered run UUID directory.")
    return run_root


def _campaign_evaluation_output_root(output_root: Path, run_uuid: str) -> Path:
    """Resolve Kaggle's run-scoped evaluation directory for the exact registered UUID."""
    output_root = Path(output_root)
    if output_root.is_symlink() or not output_root.is_dir():
        raise KaggleError("Campaign evaluation output root is missing or unsafe.")
    if (output_root / "g008-evaluation.json").is_file():
        return output_root
    arena_root = output_root / "arena-output"
    run_root = arena_root / run_uuid
    if (arena_root.is_symlink() or run_root.is_symlink() or not arena_root.is_dir()
            or not run_root.is_dir() or not (run_root / "g008-evaluation.json").is_file()):
        raise KaggleError("Campaign evaluation output is missing the exact registered run UUID directory.")
    return run_root


def _expected_recovery_current_version(events: list[dict], registration_event: dict) -> int | None:
    """Return the exact reconciled cursor for a cycle-specific workload."""
    data = registration_event.get("event_data", {})
    cycle_id = data.get("cycle_id")
    if _is_competitive_campaign(cycle_id):
        amendment = _campaign_amendment(events, registration_event)
        latest_assigned = next((event["event_data"].get("assigned_version") for event in reversed(events)
                                if event.get("event_type") == "workload_attempt"
                                and event.get("event_data", {}).get("cycle_id") == cycle_id
                                and event["event_data"].get("assigned_version") is not None), None)
        if latest_assigned is not None:
            return latest_assigned
        return amendment.get("current_version_number")
    if cycle_id == "g008-recovery-cycle-3":
        amendment = _cycle3_amendment(events, registration_event)
    elif cycle_id == "g008-recovery-cycle-2":
        cycle_event = next((event for event in events if event.get("event_type") == "recovery_cycle_amendment"
                            and event.get("event_data", {}).get("cycle_id") == cycle_id), None)
        if cycle_event is None:
            raise KaggleError("Cycle-2 registration has no matching cycle amendment.")
        amendment = cycle_event["event_data"]
    else:
        amendment = None
    if cycle_id in {"g008-recovery-cycle-2", "g008-recovery-cycle-3"}:
        latest_attempt = next((event for event in reversed(events)
                               if event.get("event_type") == "workload_attempt"
                               and event.get("event_data", {}).get("cycle_id") == cycle_id), None)
        if cycle_id == "g008-recovery-cycle-3":
            latest_observation = next((event for event in reversed(events)
                                       if event.get("event_type") == "exploratory_kernel_observation_v1"
                                       and event.get("event_data", {}).get("kernel_id")
                                       == _kernel_id(amendment["kernel_id"])), None)
            if (latest_observation is not None
                    and (latest_attempt is None
                         or latest_observation.get("event_index", -1) > latest_attempt.get("event_index", -1))):
                return latest_observation["event_data"].get("version_number")
        if latest_attempt:
            latest_data = latest_attempt["event_data"]
            return (latest_data.get("assigned_version") if latest_data.get("outcome") == "terminal"
                    else latest_data.get("no_assignment_metadata", {}).get("post_push_current_version"))
        return amendment.get("current_version_number")
    previous = [event for event in events if event.get("event_type") == "workload_attempt"
                and event.get("event_data", {}).get("registration_event_index") == registration_event.get("event_index")]
    if previous and previous[-1]["event_data"].get("assigned_version") is not None:
        return previous[-1]["event_data"].get("assigned_version")
    historical = next((event["event_data"]["historical_overrun_evidence"].get("current_version_number")
                       for event in events if event.get("event_type") == "recovery_amendment"), None)
    return historical


def _expected_recovery_terminal_status(events: list[dict], registration_event: dict) -> str | None:
    """Return the status that must still be live at the reconciled C3 version cursor."""
    data = registration_event.get("event_data", {})
    if _is_competitive_campaign(data.get("cycle_id")):
        _campaign_amendment(events, registration_event)
        latest = next((event["event_data"].get("terminal_status") for event in reversed(events)
                    if event.get("event_type") == "workload_attempt"
                    and event.get("event_data", {}).get("cycle_id") == data["cycle_id"]
                    and event["event_data"].get("assigned_version") is not None), None)
        amendment = _campaign_amendment(events, registration_event)
        if latest is not None:
            return latest
        if data.get("cycle_id") == _CAMPAIGN_V2:
            return events[amendment["selected_training_attempt_event_index"]]["event_data"].get("terminal_status")
        return amendment.get("current_terminal_status")
    if data.get("cycle_id") != "g008-recovery-cycle-3":
        return None
    amendment = _cycle3_amendment(events, registration_event)
    latest_attempt = next((event for event in reversed(events)
                           if event.get("event_type") == "workload_attempt"
                           and event.get("event_data", {}).get("cycle_id") == data["cycle_id"]
                           and event.get("event_data", {}).get("outcome") == "terminal"
                           and event.get("event_data", {}).get("assigned_version") is not None), None)
    latest_observation = next((event for event in reversed(events)
                               if event.get("event_type") == "exploratory_kernel_observation_v1"
                               and event.get("event_data", {}).get("kernel_id")
                               == _kernel_id(amendment["kernel_id"])), None)
    if (latest_observation is not None
            and (latest_attempt is None
                 or latest_observation.get("event_index", -1) > latest_attempt.get("event_index", -1))):
        return latest_observation["event_data"].get("terminal_status")
    if latest_attempt is not None:
        return latest_attempt["event_data"].get("terminal_status")
    return amendment.get("current_terminal_status")


def _validate_recovery_budget(registry, registration_event: dict, timeout: int,
                              quota_remaining: int, *, resuming_attempt_id: str | None = None,
                              attempt_id: str | None = None) -> dict:
    events = registry.events()
    registration_data = registration_event.get("event_data", {})
    if _is_competitive_campaign(registration_data.get("cycle_id")):
        amendment = _campaign_amendment(events, registration_event)
        record = registration_data.get("record", {})
        kind = registration_data.get("registration_kind")
        if (kind not in ({"train", "validation", "final"} if registration_data.get("cycle_id") == _CAMPAIGN_V2
                         else {"benchmark", "train", "validation", "final"})
                or type(timeout) is not int or timeout != record.get("session_timeout_seconds")):
            raise KaggleError("Campaign budget check requires the exact registered workload timeout and kind.")
        category = "benchmark" if kind == "benchmark" else "training" if kind == "train" else "evaluation"
        retry_budget = next((event for event in reversed(events)
                             if event.get("event_type") in {
                                 "g008_competitive_campaign_v2_retry_budget_v1",
                                 "g008_competitive_campaign_v2_retry_budget_v2",
                                 "g008_competitive_campaign_v2_retry_budget_v3"}), None)
        if registration_data.get("cycle_id") == _CAMPAIGN_V2 and retry_budget is not None:
            authorization = retry_budget.get("event_data", {})
            target_index = authorization.get("failed_registration_event_index")
            if (registration_event.get("event_index") == target_index
                    and (attempt_id or resuming_attempt_id) != authorization.get("retry_attempt_id")):
                raise KaggleError("Campaign v2 retry budget is bound to one exact retry attempt ID.")
        caps = {"benchmark": amendment.get("benchmark_gpu_seconds_cap", 0),
                "training": amendment["training_gpu_seconds_cap"],
                "evaluation": amendment["evaluation_gpu_seconds_cap"]}
        total_cap = amendment["total_gpu_seconds_cap"]
        if registration_data.get("cycle_id") == _CAMPAIGN_V2 and retry_budget is not None:
            authorization = retry_budget["event_data"]
            caps["training"] = authorization["effective_training_gpu_seconds_cap"]
            caps["evaluation"] = authorization["effective_evaluation_gpu_seconds_cap"]
            total_cap = authorization["effective_total_gpu_seconds_cap"]
        states = {}
        actual_by_category = {key: 0 for key in caps}
        output_bytes_by_category = {key: 0 for key in caps}
        for event in events:
            item = event.get("event_data", {})
            if item.get("cycle_id") != registration_data.get("cycle_id"):
                continue
            aid = item.get("attempt_id")
            typ = event.get("event_type")
            if typ == "workload_preflight":
                states[aid] = {"phase": "preflight", "data": item}
            elif typ == "workload_push_dispatched":
                states.setdefault(aid, {"phase": "dispatched", "data": item})["phase"] = "dispatched"
            elif typ == "workload_push_accepted":
                states.setdefault(aid, {"phase": "accepted", "data": item})["phase"] = "accepted"
            elif typ == "workload_attempt":
                states[aid] = {"phase": "terminal", "data": item}
                prior = events[item["registration_event_index"]]["event_data"]
                prior_kind = prior["registration_kind"]
                prior_category = ("benchmark" if prior_kind == "benchmark" else
                                  "training" if prior_kind == "train" else "evaluation")
                actual_by_category[prior_category] += item["actual_gpu_seconds"]
                output_bytes_by_category[prior_category] += item["output_bytes"]
        pending = {aid: state for aid, state in states.items() if state["phase"] != "terminal"}
        if resuming_attempt_id is not None and pending.get(resuming_attempt_id, {}).get("phase") not in {"preflight", "accepted"}:
            raise KaggleError("Only the exact pending campaign attempt may resume.")
        if set(pending) - ({resuming_attempt_id} if resuming_attempt_id else set()):
            raise KaggleError("An unresolved campaign attempt blocks every new workload push.")
        completed_registrations = {state["data"]["registration_event_index"] for state in states.values()
                                   if state["phase"] == "terminal"}
        unfinished_by_category = {key: 0 for key in caps}
        registered_timeouts = 0
        pointer = "competitive_campaign_v2_event" if registration_data.get("cycle_id") == _CAMPAIGN_V2 else "competitive_campaign_event"
        campaign_index = registration_data["record"][f"{pointer}_index"]
        registration_count = 13 if registration_data.get("cycle_id") == _CAMPAIGN_V2 else 9
        for index in range(campaign_index + 1, campaign_index + registration_count + 1):
            item = events[index]
            if item.get("event_type") != "registration":
                raise KaggleError("Campaign workload registration block is incomplete or changed.")
            if index in completed_registrations and index != registration_event.get("event_index"):
                continue
            row_data = item["event_data"]
            row_kind = row_data["registration_kind"]
            row_category = "benchmark" if row_kind == "benchmark" else "training" if row_kind == "train" else "evaluation"
            row_timeout = timeout if index == registration_event.get("event_index") else row_data["record"]["session_timeout_seconds"]
            # Previously completed current workloads may be retried only within the remaining category cap.
            if index == registration_event.get("event_index") or index not in completed_registrations:
                unfinished_by_category[row_category] += row_timeout
                registered_timeouts += row_timeout
        if (any(actual_by_category[key] + unfinished_by_category[key] > caps[key] for key in caps)
                or sum(actual_by_category.values()) + sum(unfinished_by_category.values())
                > total_cap):
            raise KaggleError("Full timeout reservation exceeds a campaign category or total compute cap.")
        reserve = amendment["protected_final_reserve_gpu_seconds"]
        if quota_remaining < registered_timeouts + reserve:
            raise KaggleError("Kaggle quota cannot cover every unfinished campaign workload and protected final reserve.")
        if registration_data.get("cycle_id") == _CAMPAIGN_V2:
            observation = next((event["event_data"] for event in events
                                if event.get("event_type") == "exploratory_kernel_observation_v1"
                                and event.get("event_data", {}).get("kernel_id")
                                == record.get("kernel_id")), None)
            if observation is None:
                raise KaggleError("Campaign v2 budget requires the registered exploratory v15 observation.")
            # v15 is a prior exploratory workload, not one of the registered v2
            # training reservations. Charge it once to the project-wide budget
            # below, while retaining its bytes in the aggregate output cap.
            output_bytes_by_category["training"] += observation["output_bytes"]
            output_caps = amendment["output_caps_bytes"]
            if (record.get("workload_output_bytes_cap") > output_caps["per_run"]
                    or sum(data.get("output_bytes", 0) for event in events
                           if event.get("event_type") == "workload_attempt"
                           for data in [event.get("event_data", {})])
                       + sum(event.get("event_data", {}).get("output_bytes", 0) for event in events
                             if event.get("event_type") == "exploratory_kernel_observation_v1")
                       > output_caps["aggregate"]):
                raise KaggleError("Exploratory v15 output exceeds the campaign aggregate artifact-output cap.")
            prior_gpu_seconds = sum(event.get("event_data", {}).get("actual_gpu_seconds", 0) for event in events
                                    if event.get("event_type") == "workload_attempt")
            prior_gpu_seconds += sum(event.get("event_data", {}).get("actual_gpu_seconds", 0) for event in events
                                     if event.get("event_type") == "exploratory_kernel_observation_v1")
            project_experiment = prior_gpu_seconds + sum(unfinished_by_category.values())
            project_total = project_experiment + reserve
            if project_experiment > 21_600 or project_total > 28_800:
                raise KaggleError("Campaign v2 reservations exceed the frozen project experiment or total GPU budget.")
        return {"actual": actual_by_category, "output_bytes": output_bytes_by_category,
                "campaign_gpu_seconds_projected": sum(actual_by_category.values()) + sum(unfinished_by_category.values()),
                "campaign_gpu_seconds_with_final_reserve": sum(actual_by_category.values()) + sum(unfinished_by_category.values()) + reserve,
                **({"exploratory_gpu_seconds": observation["actual_gpu_seconds"],
                   "project_gpu_seconds_projected": project_experiment,
                   "project_gpu_seconds_with_final_reserve": project_total}
                   if registration_data.get("cycle_id") == _CAMPAIGN_V2 else {}),
                "quota_gpu_seconds_remaining": quota_remaining,
                "protected_final_reserve_gpu_seconds": reserve, "cycle_id": registration_data["cycle_id"]}
    if registration_data.get("cycle_id") == "g008-recovery-cycle-3":
        amendment = _cycle3_amendment(events, registration_event)
        kind = registration_data.get("registration_kind")
        category = "benchmark" if kind == "benchmark" else "training" if kind == "train" else None
        record = registration_data.get("record", {})
        if category is None or type(timeout) is not int or timeout != record.get("session_timeout_seconds"):
            raise KaggleError("Cycle-3 budget check requires the exact registered workload timeout and kind.")

        cycle_id = registration_data["cycle_id"]
        cycle_actual_gpu = {"benchmark": 0, "train": 0}
        cycle_actual_output = {"benchmark": 0, "training": 0}
        states = {}
        for event in events:
            data = event.get("event_data", {})
            if event.get("event_type") == "exploratory_kernel_observation_v1":
                continuation_index = data.get("continuation_event_index")
                if (type(continuation_index) is int and 0 <= continuation_index < len(events)
                        and events[continuation_index].get("event_type") == "cycle3_retry_policy_amendment_v1"
                        and events[continuation_index].get("event_data", {}).get("registration_event_index")
                        == registration_event.get("event_index")):
                    cycle_actual_gpu["benchmark"] += data["actual_gpu_seconds"]
                    cycle_actual_output["benchmark"] += data["output_bytes"]
                continue
            if data.get("cycle_id") != cycle_id:
                continue
            typ, attempt_id = event.get("event_type"), data.get("attempt_id")
            if typ == "workload_preflight":
                states[attempt_id] = {"phase": "preflight", "data": data}
            elif typ == "workload_push_dispatched":
                states.setdefault(attempt_id, {"phase": "dispatched", "data": data})["phase"] = "dispatched"
            elif typ == "workload_push_accepted":
                states.setdefault(attempt_id, {"phase": "accepted", "data": data})["phase"] = "accepted"
            elif typ == "workload_attempt":
                states[attempt_id] = {"phase": "terminal", "data": data}
                prior_registration = events[data["registration_event_index"]]["event_data"]
                prior_kind = prior_registration["registration_kind"]
                prior_category = "benchmark" if prior_kind == "benchmark" else "training"
                cycle_actual_gpu[prior_kind] += data["actual_gpu_seconds"]
                cycle_actual_output[prior_category] += data["output_bytes"]

        pending = {aid: state for aid, state in states.items() if state["phase"] != "terminal"}
        if resuming_attempt_id is not None and pending.get(resuming_attempt_id, {}).get("phase") not in {
                "preflight", "accepted"}:
            raise KaggleError("Only the exact current-cycle pending attempt may be resumed.")
        other_pending = set(pending) - ({resuming_attempt_id} if resuming_attempt_id else set())
        if other_pending:
            raise KaggleError("An unresolved cycle-3 attempt blocks all new workload events and pushes.")

        caps = amendment["compute_caps_gpu_seconds"]
        workload_cap = record.get("workload_gpu_seconds_cap")
        if workload_cap != caps[kind] or cycle_actual_gpu[kind] + timeout > workload_cap:
            raise KaggleError("Full timeout reservation exceeds the current-cycle workload GPU budget.")
        pending_gpu = sum(state["data"]["session_timeout_seconds"] for aid, state in pending.items()
                          if aid != resuming_attempt_id)
        current_projected = (amendment["project_gpu_seconds_before"] + sum(cycle_actual_gpu.values())
                             + pending_gpu + timeout)
        future_train = caps["train"] if kind == "benchmark" else 0
        experiment_projected = current_projected + future_train
        final_reserve = amendment["protected_final_reserve_gpu_seconds"]
        projected_total = experiment_projected + final_reserve
        if experiment_projected > caps["project_experiment"]:
            raise KaggleError("Full timeout reservation exceeds the project experiment allocation.")
        if projected_total > caps["project_total"]:
            raise KaggleError("Full timeout reservation exceeds the project total or protected reserve.")
        if quota_remaining < timeout + future_train + final_reserve:
            raise KaggleError("Kaggle quota cannot cover this attempt, remaining training, and final reserve.")

        output_caps = amendment["output_category_reservations_bytes"]
        run_output_cap = record.get("workload_output_bytes_cap")
        pending_output = {"benchmark": 0, "training": 0}
        for aid, state in pending.items():
            if aid == resuming_attempt_id:
                continue
            pending_data = state["data"]
            pending_reg = events[pending_data["registration_event_index"]]["event_data"]
            pending_category = "benchmark" if pending_reg["registration_kind"] == "benchmark" else "training"
            pending_output[pending_category] += pending_reg["record"]["workload_output_bytes_cap"]
        output_caps_bytes = amendment["output_caps_bytes"]
        if (cycle_actual_output[category] + pending_output[category] + run_output_cap > output_caps[category]
                or run_output_cap > output_caps_bytes["per_run"]):
            raise KaggleError("Full registered output reservation exceeds its cycle-3 category cap.")
        output_reserved = sum(output_caps.values())
        output_projected = amendment["project_output_bytes_before"] + output_reserved
        if output_projected > output_caps_bytes["aggregate"]:
            raise KaggleError("Cycle-3 output categories exceed the aggregate project cap.")
        return {"actual": cycle_actual_gpu, "output_bytes": cycle_actual_output,
                "campaign_gpu_seconds_current_attempt_projected": current_projected,
                "future_train_reserve_gpu_seconds": future_train,
                "campaign_gpu_seconds_projected": experiment_projected,
                "campaign_gpu_seconds_with_final_reserve": projected_total,
                "quota_gpu_seconds_remaining": quota_remaining,
                "protected_final_reserve_gpu_seconds": final_reserve,
                "cycle_id": cycle_id,
                "project_output_bytes_projected_with_reserves": output_projected}

    cycle2_event = next((event for event in events if event.get("event_type") == "recovery_cycle_amendment"), None)
    if cycle2_event is not None:
        cycle2 = cycle2_event["event_data"]
        registration_data = registration_event["event_data"]
        cycle_id = "g008-recovery-cycle-2"
        if (registration_data.get("cycle_id") != cycle_id
                or registration_event.get("event_type") != "registration"):
            raise KaggleError("Cycle-2 budget check requires an exact current-cycle registration.")
        kind = registration_data["registration_kind"]
        category = "benchmark" if kind == "benchmark" else "training" if kind == "train" else None
        if category is None or registration_data["record"].get("cycle_id") != cycle_id:
            raise KaggleError("Cycle-2 registration kind or identity is invalid.")
        if type(timeout) is not int or timeout != registration_data["record"].get("session_timeout_seconds"):
            raise KaggleError("Cycle-2 budget check must reserve the exact timeout frozen in its registration.")

        cycle_actual_gpu = {"benchmark": 0, "train": 0}
        cycle_actual_output = {"benchmark": 0, "training": 0}
        states = {}
        for event in events:
            data = event.get("event_data", {})
            if data.get("cycle_id") != cycle_id:
                continue
            typ = event.get("event_type")
            attempt_id = data.get("attempt_id")
            if typ == "workload_preflight":
                states[attempt_id] = {"phase": "preflight", "data": data}
            elif typ == "workload_push_dispatched":
                states.setdefault(attempt_id, {"phase": "dispatched", "data": data})["phase"] = "dispatched"
            elif typ == "workload_push_accepted":
                states.setdefault(attempt_id, {"phase": "accepted", "data": data})["phase"] = "accepted"
            elif typ == "workload_attempt":
                states[attempt_id] = {"phase": "terminal", "data": data}
                reg = events[data["registration_event_index"]]["event_data"]
                attempt_kind = events[data["registration_event_index"]]["event_data"]["registration_kind"]
                attempt_category = "benchmark" if attempt_kind == "benchmark" else "training"
                cycle_actual_gpu[attempt_kind] += data["actual_gpu_seconds"]
                cycle_actual_output[attempt_category] += data["output_bytes"]

        pending = {attempt_id: state for attempt_id, state in states.items() if state["phase"] != "terminal"}
        if resuming_attempt_id is not None:
            if pending.get(resuming_attempt_id, {}).get("phase") not in {"preflight", "accepted"}:
                raise KaggleError("Only the exact current-cycle pending attempt may be resumed.")
        other_pending = set(pending) - ({resuming_attempt_id} if resuming_attempt_id else set())
        if other_pending:
            raise KaggleError("An unresolved cycle-2 attempt blocks all new workload events and pushes.")

        caps = cycle2["compute_caps_gpu_seconds"]
        workload_cap = registration_data["record"]["workload_gpu_seconds_cap"]
        if workload_cap != caps[kind] or cycle_actual_gpu[kind] + timeout > workload_cap:
            raise KaggleError("Full timeout reservation exceeds the current-cycle workload GPU budget.")
        pending_gpu = sum(state["data"]["session_timeout_seconds"] for aid, state in pending.items()
                          if aid != resuming_attempt_id)
        current_projected = (cycle2["project_gpu_seconds_before"] + sum(cycle_actual_gpu.values())
                             + pending_gpu + timeout)
        future_train = caps["train"] if kind == "benchmark" else 0
        experiment_projected = current_projected + future_train
        final_reserve = cycle2["protected_final_reserve_gpu_seconds"]
        projected_total = experiment_projected + final_reserve
        if experiment_projected > caps["project_experiment"]:
            raise KaggleError("Full timeout reservation exceeds the project experiment allocation.")
        if projected_total > caps["project_total"]:
            raise KaggleError("Full timeout reservation exceeds the project total or protected reserve.")
        if quota_remaining < timeout + future_train + final_reserve:
            raise KaggleError("Kaggle quota cannot cover this attempt, remaining training, and final reserve.")

        output_caps = cycle2["output_category_reservations_bytes"]
        run_output_cap = registration_data["record"]["workload_output_bytes_cap"]
        pending_output = {"benchmark": 0, "training": 0}
        for aid, state in pending.items():
            if aid == resuming_attempt_id:
                continue
            data = state["data"]
            reg_data = events[data["registration_event_index"]]["event_data"]
            pending_category = "benchmark" if reg_data["registration_kind"] == "benchmark" else "training"
            pending_output[pending_category] += reg_data["record"]["workload_output_bytes_cap"]
        if (cycle_actual_output[category] + pending_output[category] + run_output_cap > output_caps[category]
                or run_output_cap > cycle2["output_caps_bytes"]["per_run"]):
            raise KaggleError("Full registered output reservation exceeds its cycle-2 category cap.")
        output_reserved = sum(output_caps.values())
        output_projected = cycle2["project_output_bytes_before"] + output_reserved
        if output_projected > cycle2["output_caps_bytes"]["aggregate"]:
            raise KaggleError("Cycle-2 output categories exceed the aggregate project cap.")
        return {"actual": cycle_actual_gpu, "output_bytes": cycle_actual_output,
                "campaign_gpu_seconds_current_attempt_projected": current_projected,
                "future_train_reserve_gpu_seconds": future_train,
                "campaign_gpu_seconds_projected": experiment_projected,
                "campaign_gpu_seconds_with_final_reserve": projected_total,
                "quota_gpu_seconds_remaining": quota_remaining,
                "protected_final_reserve_gpu_seconds": final_reserve,
                "cycle_id": cycle_id,
                "project_output_bytes_projected_with_reserves": output_projected}

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
    future_train = 7200 if kind == "benchmark" else 0
    current_projected = reservation["campaign_gpu_seconds_before"] + sum(actual.values()) + timeout
    experiment_projected = current_projected + future_train
    final_gpu = reservation["reserved_gpu_seconds"]["validation"] + reservation["reserved_gpu_seconds"]["final"]
    if experiment_projected > 21600:
        raise KaggleError("Full timeout reservation exceeds the project experiment allocation.")
    projected_total = experiment_projected + final_gpu
    if projected_total > 28800:
        raise KaggleError("Full timeout reservation exceeds the project total compute budget or protected final reserve.")
    if quota_remaining < timeout + future_train + 7200:
        raise KaggleError("Kaggle GPU quota cannot cover this attempt, remaining training, and final reserve.")
    output_reserved = sum(reservation["reserved_output_bytes"].values())
    if reservation["campaign_output_bytes_before"] + output + output_reserved > 5_000_000_000:
        raise KaggleError("Full registered output reservation exceeds the project output cap.")
    return {"actual": actual, "output_bytes": output,
            "campaign_gpu_seconds_current_attempt_projected": current_projected,
            "future_train_reserve_gpu_seconds": future_train,
            "campaign_gpu_seconds_projected": experiment_projected,
            "campaign_gpu_seconds_with_final_reserve": projected_total,
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
    cycle_id = reg_data.get("cycle_id")
    cycle2 = next((event["event_data"] for event in events
                   if event.get("event_type") == "recovery_cycle_amendment"), None)
    if cycle_id == "g008-recovery-cycle-2" and cycle2 is None:
        raise KaggleError("Cycle-2 registration has no matching cycle amendment.")
    campaign_amendment = (_campaign_amendment(events, registration)
                          if _is_competitive_campaign(cycle_id) else None)
    cycle3_amendment = (_cycle3_amendment(events, registration)
                        if cycle_id == "g008-recovery-cycle-3" else None)
    cycle3_retry_amendment = next((event["event_data"] for event in events
                                   if event.get("event_type") == "recovery_cycle_retry_amendment_v1"), None)
    record, hashes = _recovery_record(registration, implementation_variant_id, events)
    kernel_id = _kernel_id(record["kernel_id"])
    timeout = record["session_timeout_seconds"]
    cycle3_rebase_event = next((event for event in reversed(events)
                                if event.get("event_type") == "cycle3_retry_rebased_authorization_v1"
                                and event.get("event_data", {}).get("registration_event_index")
                                == registration_event_index), None)
    cycle3_rebase_authorization = (cycle3_rebase_event.get("event_data")
                                  if cycle3_rebase_event is not None else None)
    if cycle_id == "g008-recovery-cycle-3" and registration_event_index == 35 and cycle3_rebase_authorization is not None:
        authorization = cycle3_rebase_authorization
        observation_index = authorization.get("observation_event_index")
        observation_event = (events[observation_index]
                             if type(observation_index) is int and 0 <= observation_index < len(events) else None)
        observation = observation_event.get("event_data", {}) if observation_event else {}
        if (authorization.get("attempt_id") != attempt_id
                or authorization.get("expected_current_version") != _expected_recovery_current_version(events, registration)
                or authorization.get("kernel_id") != kernel_id
                or authorization.get("registration_event_hash") != registration.get("event_hash")
                or authorization.get("session_timeout_seconds") != timeout
                or authorization.get("reserved_gpu_seconds") != timeout
                or observation_event is None
                or observation_event.get("event_type") != "exploratory_kernel_observation_v1"
                or observation_event.get("event_hash") != authorization.get("observation_event_hash")
                or observation.get("kernel_id") != kernel_id
                or observation.get("version_number") != authorization.get("expected_current_version")
                or observation.get("run_uuid") != authorization.get("observed_run_uuid")
                or observation.get("terminal_status") != authorization.get("observed_terminal_status")
                or authorization.get("observed_terminal_status") != "COMPLETE"
                or any(authorization.get(key) != value for key, value in hashes.items())):
            raise KaggleError("Cycle-3 rebased authorization does not match this exact attempt, version, registration, and package.")
    expected_cap = record["workload_gpu_seconds_cap"]
    if timeout > expected_cap:
        raise KaggleError("Registered timeout exceeds the recovery workload cap.")
    source_dir = Path(source_dir)
    if package_tree_sha256(source_dir) != hashes["package_sha256"]:
        raise KaggleError("Frozen Kaggle package bytes differ from the registered package_sha256.")
    frozen_package_temp = tempfile.TemporaryDirectory(prefix="g008-kaggle-package-")
    frozen_package = Path(frozen_package_temp.name) / "source"
    try:
        shutil.copytree(source_dir, frozen_package)
    except OSError as exc:
        raise KaggleError("Could not create an immutable local snapshot of the registered Kaggle package.") from exc
    if package_tree_sha256(frozen_package) != hashes["package_sha256"]:
        raise KaggleError("Frozen Kaggle package changed while its submission snapshot was copied.")
    # Freeze files and directories so the exact verified snapshot remains the
    # object hashed and submitted throughout this attempt.
    for path in [frozen_package, *frozen_package.rglob("*")]:
        if path.is_dir():
            path.chmod(path.stat().st_mode & ~0o222)
        elif path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    source_dir = frozen_package
    marker = source_dir / ".arena-source-revision"
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
        if cycle3_rebase_authorization is not None:
            auth_index = cycle3_rebase_event["event_index"]
            if (cycle3_rebase_authorization.get("attempt_id") != attempt_id
                    or cycle3_rebase_authorization.get("registration_event_index") != registration_event_index
                    or cycle3_rebase_authorization.get("registration_event_hash") != registration.get("event_hash")
                    or (phase is None and auth_index != len(events) - 1)
                    or (phase == "workload_preflight"
                        and (len(existing_attempt) != 1
                             or auth_index + 1 != existing_attempt[0].get("event_index")))):
                raise KaggleError("Cycle-3 retry rebase is bound to a different attempt or is no longer the exact authorized preflight head.")
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
                            if (reg_data["registration_kind"] == "train"
                                or campaign_amendment is not None and reg_data["registration_kind"] != "benchmark")
                            else None)):
                    raise KaggleError("Existing preflight tail does not match its exact registration, package, timeout, or operator evidence.")
            else:
                expected_version = _expected_recovery_current_version(events, registration)
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
            if campaign_amendment is not None:
                expected_campaign_status = _expected_recovery_terminal_status(events, registration)
                if expected_campaign_status is None:
                    if after["current_version_number"] != campaign_amendment["current_version_number"]:
                        raise KaggleError("Live campaign version differs from the exact frozen initial version.")
                elif prior_status != expected_campaign_status:
                    raise KaggleError("Live status differs from the campaign's exact reconciled terminal attempt.")
                if cycle_id == _CAMPAIGN_V2:
                    _validate_campaign_v2_phase(events, registration)
            if cycle3_amendment is not None:
                prior_cycle3_dispatch = any(
                    event.get("event_type") == "workload_push_dispatched"
                    and event.get("event_data", {}).get("cycle_id") == cycle_id
                    for event in events)
                if not prior_cycle3_dispatch:
                    from training.eval.registry import cycle3_preflight_context_sha256
                    live_context = {
                        "kernel_id": kernel_id,
                        "current_version_number": after["current_version_number"],
                        "current_terminal_status": prior_status,
                        "current_status_evidence": {
                            "status": prior_status,
                            "current_version_number": after["current_version_number"],
                            "kernel_id": kernel_id,
                        },
                        "quota_gpu_seconds_remaining_before": quota_remaining,
                        "quota_evidence": {
                            "remaining_gpu_seconds": quota_remaining,
                            "used_gpu_seconds": quota_used_before,
                            "source": "pinned Kaggle SDK quota_view",
                        },
                    }
                    if cycle3_preflight_context_sha256(live_context) != cycle3_amendment.get(
                            "preflight_context_sha256"):
                        raise KaggleError("Live initial Cycle-3 status/quota context differs from its authorized snapshot.")
                elif prior_status != _expected_recovery_terminal_status(events, registration):
                    raise KaggleError("Live Cycle-3 status differs from its exact reconciled terminal attempt.")
            budget = _validate_recovery_budget(
                registry, registration, timeout, quota_remaining,
                resuming_attempt_id=attempt_id if phase == "workload_preflight" else None,
                attempt_id=attempt_id,
            )
            if (cycle3_retry_amendment is not None and cycle3_rebase_authorization is None
                    and registration_event_index == 35
                    and not any(event.get("event_type") == "workload_push_dispatched"
                                and event.get("event_data", {}).get("registration_event_index") == 35
                                for event in events)):
                # Freeze package bytes, then sample live context immediately before
                # writing preflight/dispatch events and calling Kaggle.
                if package_tree_sha256(Path(source_dir)) != hashes["package_sha256"]:
                    raise KaggleError("Frozen Kaggle package changed before retry dispatch; no attempt was appended.")
                retry_before = _sdk_metadata(api, kernel_id)
                retry_status, _ = _sdk_status(api, kernel_id)
                retry_after = _sdk_metadata(api, kernel_id)
                quota_remaining, quota_used_before = _quota_seconds(api)
                if (retry_before != retry_after or retry_after["current_version_number"] != expected_version
                        or retry_status != _expected_recovery_terminal_status(events, registration)):
                    raise KaggleError(
                        "Live status/version differs from the authorized Cycle-3 retry context; no attempt was dispatched.")
                from training.eval.registry import cycle3_retry_context_sha256
                retry_live_context = {
                    "kernel_id": kernel_id,
                    "current_version_number": retry_after["current_version_number"],
                    "current_terminal_status": retry_status,
                    "current_status_evidence": {
                        "status": retry_status,
                        "current_version_number": retry_after["current_version_number"],
                        "kernel_id": kernel_id,
                    },
                    "quota_gpu_seconds_remaining_before": quota_remaining,
                    "quota_evidence": {
                        "remaining_gpu_seconds": quota_remaining,
                        "used_gpu_seconds": quota_used_before,
                        "source": "pinned Kaggle SDK quota_view",
                    },
                }
                if cycle3_retry_context_sha256(retry_live_context) != cycle3_retry_amendment.get(
                        "preflight_context_sha256"):
                    raise KaggleError(
                        "Live status/quota differs from the authorized Cycle-3 retry context; no push was attempted.")
                budget = _validate_recovery_budget(
                    registry, registration, timeout, quota_remaining,
                    resuming_attempt_id=attempt_id if phase == "workload_preflight" else None,
                    attempt_id=attempt_id,
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
                    "lock_head_hash": (registry.lock_head_hash
                                       if (reg_data["registration_kind"] == "train"
                                           or campaign_amendment is not None and reg_data["registration_kind"] != "benchmark")
                                       else None),
                }
                if cycle_id is not None:
                    preflight_data["cycle_id"] = cycle_id
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
            if cycle_id is not None:
                dispatch_data["cycle_id"] = cycle_id
            dispatch_data.update(preflight_event_index=preflight_event["event_index"],
                                 preflight_event_hash=preflight_event["event_hash"],
                                 dispatch_timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                                 request_identity=f"{kernel_id}:{attempt_id}:{timeout}")
            if package_tree_sha256(source_dir) != hashes["package_sha256"]:
                raise KaggleError("Read-only submission snapshot differs from the registered package; no push was attempted.")
            dispatch_event = registry.dispatch_workload_push(dispatch_data)
            push_started = monotonic()
            try:
                response = api.kernels_push(str(Path(source_dir).resolve()), timeout=str(timeout), acc="NvidiaTeslaT4")
            except Exception as exc:
                # A generic exception cannot prove that Kaggle did not assign a version.
                # Keep the dispatch reserved, but preserve safe transport diagnostics for recovery.
                response = getattr(exc, "response", None)
                status = (getattr(response, "status_code", None) if response is not None else None)
                if status is None:
                    status = getattr(exc, "status", getattr(exc, "status_code", None))
                status_text = f", HTTP {status}" if isinstance(status, int) else ""
                request_id = None
                headers = getattr(response, "headers", None) if response is not None else None
                if headers is not None:
                    request_id = headers.get("X-Request-Id") or headers.get("Request-Id")
                request_text = f", request_id={request_id}" if isinstance(request_id, str) and request_id else ""
                server_message = None
                if response is not None:
                    try:
                        payload = response.json()
                    except (ValueError, AttributeError):
                        payload = None
                    if isinstance(payload, dict):
                        details = []
                        for key in ("error", "message", "detail", "code", "reason"):
                            value = payload.get(key)
                            if isinstance(value, str) and value.strip():
                                details.append(value)
                            elif isinstance(value, dict):
                                details.extend(item for item in value.values()
                                               if isinstance(item, str) and item.strip())
                        if details:
                            server_message = re.sub(r"[\r\n\t]+", " ", "; ".join(details))[:240]
                            server_message = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [redacted]", server_message)
                            server_message = re.sub(
                                r"(?i)(access[_ -]?token|api[_ -]?token)\s*[:=]\s*[^\s,;]+",
                                r"\1=[redacted]", server_message)
                    elif response is not None:
                        body = getattr(response, "text", None)
                        if isinstance(body, str) and body.strip():
                            server_message = re.sub(r"[\r\n\t]+", " ", body)[:240]
                            server_message = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [redacted]", server_message)
                            server_message = re.sub(
                                r"(?i)(access[_ -]?token|api[_ -]?token)\s*[:=]\s*[^\s,;]+",
                                r"\1=[redacted]", server_message)
                message_text = f", server={server_message}" if server_message else ""
                raise KaggleError(
                    f"Kaggle push response is ambiguous ({type(exc).__name__}{status_text}{request_text}{message_text}); "
                    "dispatch remains pending until exact kernel metadata is reconciled."
                ) from exc
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
                        if cycle_id is not None:
                            rejection_data["cycle_id"] = cycle_id
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
            if cycle_id is not None:
                accepted_data["cycle_id"] = cycle_id
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
                attempt_id=attempt_id,
            )
            push_started = None

        # A Kaggle version can wait in QUEUED for many minutes before any GPU
        # work begins. Start the bounded session/reconciliation window only
        # when the exact assigned version first reports an active worker.
        deadline = float("inf")
        push_started = None
        status_evidence = []
        terminal_metadata = None
        terminal_status = None
        while True:
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
            if not _status_is_waiting_for_worker(status_item):
                if push_started is None:
                    push_started = monotonic()
                    deadline = push_started + timeout + reconciliation_slack_seconds
                elif monotonic() >= deadline:
                    raise KaggleError("Kaggle attempt remains nonterminal at the registered monotonic timeout deadline.")
            now = monotonic()
            sleep(min(float(poll_interval_seconds), max(0.0, deadline - now)))
        elapsed_seconds = (0 if push_started is None
                           else max(0, math.ceil(monotonic() - push_started)))
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
        if cycle_id is not None:
            terminal_data["cycle_id"] = cycle_id
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
            or not isinstance(visible_cuda_devices, list) or not visible_cuda_devices
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
    if (visible_cuda_devices[0] != device["cuda_name"]
            or evidence.get("visible_cuda_device_count", len(visible_cuda_devices)) != len(visible_cuda_devices)
            or evidence.get("training_device_index", 0) != 0
            or evidence.get("training_cuda_device_count", 1) != 1):
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
            or not isinstance(visible, list) or not visible
            or visible[0] != "Tesla T4"
            or evidence.get("visible_cuda_device_count", len(visible)) != len(visible)
            or evidence.get("training_device_index", 0) != 0
            or evidence.get("training_cuda_device_count", 1) != 1 or not isinstance(device, dict)
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
    legacy_summary_fields = {
        "candidate_update_only_joint_transitions_per_second",
        "incumbent_update_only_joint_transitions_per_second",
        "candidate_speedup_ratio",
        "wall_inclusive_joint_transitions_per_second",
        "update_only_seconds",
        "wall_seconds",
        "optimizer_seconds",
        "rollout_seconds",
    }
    legacy_summaries = {
        "candidate_update_only_joint_transitions_per_second": expected_summaries[
            "candidate_update_only_joint_transitions_per_second"],
        "incumbent_update_only_joint_transitions_per_second": expected_summaries[
            "incumbent_update_only_joint_transitions_per_second"],
        "candidate_speedup_ratio": expected_summaries["candidate_speedup_ratio"],
        "wall_inclusive_joint_transitions_per_second": pct(
            derived_by_role["candidate"]["wall_inclusive_joint_transitions_per_second"]),
        "update_only_seconds": pct(derived_by_role["candidate"]["update_only_seconds"]),
        "wall_seconds": pct(derived_by_role["candidate"]["wall_seconds"]),
        "optimizer_seconds": pct(derived_by_role["candidate"]["optimizer_seconds"]),
        "rollout_seconds": pct(derived_by_role["candidate"]["rollout_seconds"]),
    }

    def summary_matches(actual: object, expected: dict[str, dict[str, float]]) -> bool:
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return False
        for field, expected_percentiles in expected.items():
            actual_percentiles = actual.get(field)
            if not isinstance(actual_percentiles, dict) or set(actual_percentiles) != set(expected_percentiles):
                return False
            for percentile, value in expected_percentiles.items():
                observed = actual_percentiles.get(percentile)
                if (isinstance(observed, bool) or not isinstance(observed, (int, float))
                        or not math.isfinite(observed) or observed != value):
                    return False
        return True

    if not isinstance(summary, dict):
        raise KaggleError("G008 /3 summary or selected implementation is invalid.")
    if set(summary) == legacy_summary_fields:
        summaries_valid = summary_matches(summary, legacy_summaries)
    else:
        summaries_valid = summary_matches(summary, expected_summaries)
    if (not summaries_valid
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
    recovery.add_argument("--variant", choices=("incumbent", "candidate", "baseline", "mixed", "paired", "frozen-candidate"))
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
                    artifact = register_recovery_benchmark_artifact(
                        registry, terminal_event, args.output_dir,
                        args.output_dir / "g008-recovery-benchmark-artifact.json",
                    )
                    if _is_competitive_campaign(registration.get("cycle_id")) and registration.get("cycle_id") != _CAMPAIGN_V2:
                        lock = registry.close_registrations()
                        if lock.get("event_type") != "registration_closed":
                            raise KaggleError("Campaign registration lock did not close after its benchmark artifact.")
                        registry.attest_benchmark(artifact["event_index"])

            def bind_complete_campaign_artifacts(terminal_event: dict) -> None:
                terminal_data = terminal_event.get("event_data", {})
                if (terminal_data.get("terminal_status") != "COMPLETE"
                        or not _is_competitive_campaign(registration.get("cycle_id"))
                        or registration.get("registration_kind") == "benchmark"):
                    return
                live_events = registry.events()
                exact_registration = live_events[args.registration_event_index]
                if (exact_registration.get("event_hash") != events[args.registration_event_index].get("event_hash")
                        or terminal_event.get("event_index") != len(live_events) - 1):
                    raise KaggleError("Campaign output reconciliation requires the exact latest registered terminal event.")
                kind = registration["registration_kind"]
                v2 = registration.get("cycle_id") == _CAMPAIGN_V2
                if kind == "train":
                    attempt_package_sha256 = terminal_data.get("package_sha256")
                    _record, effective_hashes = _recovery_record(
                        exact_registration, registration["record"].get("strategy_role"), live_events)
                    if any(terminal_data.get(key) != value for key, value in effective_hashes.items()):
                        raise KaggleError("Campaign training terminal hashes differ from the exact package revision chain.")
                    artifact_root = _campaign_training_output_root(
                        args.output_dir, registration["record"]["run_uuid"])
                    artifact = _validate_campaign_training_output(
                        exact_registration, artifact_root, args.source_dir, attempt_package_sha256)
                    record = registration["record"]
                    if v2:
                        if attempt_package_sha256 != terminal_data.get("package_sha256"):
                            raise KaggleError("Campaign training attestation is missing its exact attempt package hash.")
                        payload = {
                            "schema_id": "g008-training-artifact-attestation", "schema_version": 2,
                            "campaign_id": record["campaign_id"],
                            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                            "training_registration_event_index": args.registration_event_index,
                            "training_registration_event_hash": exact_registration["event_hash"],
                            "training_attempt_event_index": terminal_event["event_index"],
                            "training_attempt_event_hash": terminal_event["event_hash"],
                            "step": record["update_budget"],
                            "checkpoint_sha256": artifact["checkpoint_sha256"],
                            "output_manifest_sha256": artifact["manifest_sha256"],
                            "metrics_sha256": artifact["metrics_sha256"],
                            "package_sha256": attempt_package_sha256,
                            "source_sha256": terminal_data["source_sha256"],
                            "config_sha256": terminal_data["config_sha256"],
                            "schedule_descriptor_sha256": record["schedule_descriptor_sha256"],
                        }
                        registry.record_competitive_training_artifact_v2(payload)
                        return
                    if record.get("strategy_role") != "mixed" or record.get("pair_id") != "seed-pair-3":
                        return
                    completed_train = []
                    for event in live_events:
                        data = event.get("event_data", {})
                        if (event.get("event_type") == "workload_attempt"
                                and data.get("cycle_id") == registration.get("cycle_id")
                                and data.get("terminal_status") == "COMPLETE"):
                            target_registration = live_events[data["registration_event_index"]]
                            if target_registration["event_data"].get("registration_kind") == "train":
                                completed_train.append(event)
                    if len(completed_train) != 6:
                        return
                    freeze_type = "competitive_candidate_freeze_v2" if v2 else "competitive_candidate_freeze"
                    if any(event.get("event_type") == freeze_type for event in live_events):
                        raise KaggleError("Campaign candidate freeze already exists; refusing a duplicate freeze.")
                    freeze_payload = {
                        "schema_id": "g008-candidate-freeze", "schema_version": 2 if v2 else 1,
                        "campaign_id": record["campaign_id"],
                        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "candidate_registration_event_index": args.registration_event_index,
                        "candidate_registration_event_hash": exact_registration["event_hash"],
                        "train_attempt_event_index": terminal_event["event_index"],
                        "train_attempt_event_hash": terminal_event["event_hash"],
                        "checkpoint_sha256": artifact["checkpoint_sha256"],
                        "checkpoint_artifact_sha256": artifact["checkpoint_sha256"],
                        "output_manifest_sha256": artifact["manifest_sha256"],
                    }
                    if v2:
                        registry.freeze_competitive_candidate_v2(freeze_payload)
                    else:
                        registry.freeze_competitive_candidate(freeze_payload)
                    return

                freeze_type = "competitive_candidate_freeze_v2" if v2 else "competitive_candidate_freeze"
                if v2 and kind == "validation":
                    record = exact_registration["event_data"]["record"]
                    training_registration = next((event for event in live_events
                        if event.get("event_type") == "registration"
                        and event.get("event_data", {}).get("cycle_id") == _CAMPAIGN_V2
                        and event.get("event_data", {}).get("registration_id")
                        == record.get("training_registration_id")), None)
                    if training_registration is None:
                        raise KaggleError("Campaign v2 validation does not resolve its exact training registration.")
                    training_index = training_registration["event_index"]
                    training_attempt = next((event for event in live_events
                        if event.get("event_type") == "workload_attempt"
                        and event.get("event_data", {}).get("cycle_id") == _CAMPAIGN_V2
                        and event.get("event_data", {}).get("registration_event_index") == training_index
                        and event.get("event_data", {}).get("terminal_status") == "COMPLETE"), None)
                    training_artifact = next((event for event in live_events
                        if event.get("event_type") == "competitive_training_artifact_v2"
                        and event.get("event_data", {}).get("training_registration_event_index") == training_index), None)
                    if training_attempt is None or training_artifact is None:
                        raise KaggleError("Campaign v2 validation requires the exact COMPLETE attested training checkpoint.")
                    candidate_sha = training_artifact["event_data"]["checkpoint_sha256"]
                    evaluation_root = _campaign_evaluation_output_root(
                        args.output_dir, record["run_uuid"])
                    artifact = _validate_campaign_evaluation_output(
                        exact_registration, terminal_event, evaluation_root, candidate_sha,
                        args.source_dir)
                    from training.eval.registry import competitive_final_statistics
                    stats = competitive_final_statistics(artifact["wins"], artifact["draws"], artifact["losses"])
                    validation_payload = {
                        "schema_id": "g008-competitive-validation-result", "schema_version": 2,
                        "campaign_id": record["campaign_id"],
                        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "training_registration_event_index": training_index,
                        "training_registration_event_hash": training_registration["event_hash"],
                        "training_attempt_event_index": training_attempt["event_index"],
                        "training_attempt_event_hash": training_attempt["event_hash"],
                        "training_artifact_event_index": training_artifact["event_index"],
                        "training_artifact_event_hash": training_artifact["event_hash"],
                        "validation_registration_event_index": args.registration_event_index,
                        "validation_registration_event_hash": exact_registration["event_hash"],
                        "validation_attempt_event_index": artifact["attempt_event_index"],
                        "validation_attempt_event_hash": artifact["attempt_event_hash"],
                        "checkpoint_sha256": candidate_sha,
                        "report_sha256": artifact["report_sha256"],
                        "output_manifest_sha256": artifact["manifest_sha256"],
                        "game_count": 512, "wins": artifact["wins"], "draws": artifact["draws"],
                        "losses": artifact["losses"], "win_rate": stats["win_rate"],
                        "wilson_95_lower": stats["wilson_95_lower"],
                        "wilson_95_upper": stats["wilson_95_upper"],
                        "opponent_id": "g007-step-300",
                    }
                    registry.record_competitive_validation_result_v2(validation_payload)
                    updated_events = registry.events()
                    validations = [event for event in updated_events
                                   if event.get("event_type") == "competitive_validation_result_v2"
                                   and event.get("event_data", {}).get("campaign_id") == record["campaign_id"]]
                    if len(validations) == 6:
                        comparison = registry.record_competitive_strategy_comparison_v2(
                            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
                        selected = comparison["event_data"]["selected_candidate"]
                        freeze_payload = {
                            "schema_id": "g008-candidate-freeze", "schema_version": 2,
                            "campaign_id": record["campaign_id"],
                            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                            "strategy_comparison_event_index": comparison["event_index"],
                            "strategy_comparison_event_hash": comparison["event_hash"],
                            "candidate_registration_event_index": selected["training_registration_event_index"],
                            "candidate_registration_event_hash": selected["training_registration_event_hash"],
                            "train_attempt_event_index": selected["training_attempt_event_index"],
                            "train_attempt_event_hash": selected["training_attempt_event_hash"],
                            "training_artifact_event_index": selected["training_artifact_event_index"],
                            "training_artifact_event_hash": selected["training_artifact_event_hash"],
                            "validation_registration_event_index": selected["validation_registration_event_index"],
                            "validation_registration_event_hash": selected["validation_registration_event_hash"],
                            "validation_attempt_event_index": selected["validation_attempt_event_index"],
                            "validation_attempt_event_hash": selected["validation_attempt_event_hash"],
                            "validation_result_event_index": selected["validation_result_event_index"],
                            "validation_result_event_hash": selected["validation_result_event_hash"],
                            "checkpoint_sha256": selected["checkpoint_sha256"],
                            "checkpoint_artifact_sha256": selected["checkpoint_sha256"],
                            "output_manifest_sha256": selected["output_manifest_sha256"],
                        }
                        registry.freeze_competitive_candidate_v2(freeze_payload)
                    return
                freeze = next((event for event in live_events
                               if event.get("event_type") == freeze_type
                               and event.get("event_data", {}).get("campaign_id") == registration["record"]["campaign_id"]), None)
                if freeze is None:
                    raise KaggleError("Campaign evaluation cannot reconcile before the candidate freeze is registered.")
                candidate_sha = freeze["event_data"].get("checkpoint_artifact_sha256") or freeze["event_data"].get("checkpoint_sha256")
                artifact = _validate_campaign_evaluation_output(
                    exact_registration, terminal_event,
                    _campaign_evaluation_output_root(args.output_dir, registration["record"]["run_uuid"]),
                    candidate_sha,
                    args.source_dir)
                report, manifest = artifact["report"], artifact["manifest"]
                result_common = {
                    "candidate_freeze_event_index": freeze["event_index"],
                    "candidate_freeze_event_hash": freeze["event_hash"],
                    "candidate_checkpoint_sha256": candidate_sha,
                    "report_sha256": artifact["report_sha256"],
                    "output_manifest_sha256": artifact["manifest_sha256"],
                    "game_count": 512, "wins": artifact["wins"], "draws": artifact["draws"],
                    "losses": artifact["losses"], "opponent_id": report["opponent_id"],
                }
                if kind == "validation":
                    validation_type = "competitive_validation_result_v2" if v2 else "competitive_validation_result"
                    if any(event.get("event_type") == validation_type for event in live_events):
                        raise KaggleError("Campaign validation result already exists; refusing duplicate reconciliation.")
                    from training.eval.registry import (competitive_noninferiority_statistics,
                                                        _G008_VALIDATION_REFERENCE_SHA256)
                    stats = competitive_noninferiority_statistics(artifact["wins"], artifact["draws"], artifact["losses"])
                    validation_payload = {
                        "schema_id": "g008-candidate-validation-result", "schema_version": 2 if v2 else 1,
                        "campaign_id": report["campaign_id"],
                        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        **result_common,
                        "validation_registration_event_index": args.registration_event_index,
                        "validation_registration_event_hash": exact_registration["event_hash"],
                        "validation_attempt_event_index": artifact["attempt_event_index"],
                        "validation_attempt_event_hash": artifact["attempt_event_hash"],
                        "reference_report_sha256": _G008_VALIDATION_REFERENCE_SHA256,
                        "reference_game_count": 512, "reference_wins": 217,
                        "reference_draws": 51, "reference_losses": 244,
                        "noninferiority_margin": "0.065000",
                        "win_rate": stats["win_rate"], "wilson_95_lower": stats["wilson_95_lower"],
                        "wilson_95_upper": stats["wilson_95_upper"],
                        "win_rate_difference_95_lower": stats["win_rate_difference_95_lower"],
                        "win_rate_difference_95_upper": stats["win_rate_difference_95_upper"],
                        "passed": stats["passed"],
                    }
                    if v2:
                        registry.record_competitive_validation_result_v2(validation_payload)
                    else:
                        registry.record_competitive_validation_result(validation_payload)
                else:
                    final_type = "competitive_final_result_v2" if v2 else "competitive_final_result"
                    if any(event.get("event_type") == final_type for event in live_events):
                        raise KaggleError("Campaign final result already exists; refusing duplicate reconciliation.")
                    from training.eval.registry import competitive_final_statistics
                    stats = competitive_final_statistics(artifact["wins"], artifact["draws"], artifact["losses"])
                    final_payload = {
                        "schema_id": "g008-final-evaluation-result", "schema_version": 2 if v2 else 1,
                        "campaign_id": report["campaign_id"],
                        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        **result_common,
                        "final_registration_event_index": args.registration_event_index,
                        "final_registration_event_hash": exact_registration["event_hash"],
                        "final_attempt_event_index": artifact["attempt_event_index"],
                        "final_attempt_event_hash": artifact["attempt_event_hash"],
                        "roster_sha256": report["roster_sha256"],
                        "opponent_code_sha256": report["opponent_code_sha256"],
                        "win_rate": stats["win_rate"], "wilson_95_lower": stats["wilson_95_lower"],
                        "wilson_95_upper": stats["wilson_95_upper"], "passed": stats["passed"],
                    }
                    if v2:
                        registry.record_competitive_final_result_v2(final_payload)
                    else:
                        registry.record_competitive_final_result(final_payload)

            requested_variant = args.variant
            if _is_competitive_campaign(registration.get("cycle_id")):
                expected_campaign_variant = ("paired" if registration.get("registration_kind") == "benchmark"
                                             else record.get("strategy_role") if registration.get("registration_kind") == "train"
                                             else "frozen-candidate")
                if requested_variant is not None and requested_variant != expected_campaign_variant:
                    raise KaggleError(f"Campaign registration requires --variant {expected_campaign_variant}.")
                requested_variant = expected_campaign_variant
            elif requested_variant is None:
                requested_variant = "incumbent"

            terminal = run_recovery_workload(
                registry, args.registration_event_index, args.source_dir,
                attempt_id=args.attempt_id,
                implementation_variant_id=requested_variant,
                no_external_writer_confirmed=args.no_external_writer_confirmed,
                api=api, output_bytes=download_outputs,
                terminal_callback=lambda event: (bind_complete_benchmark(event),
                                                 bind_complete_campaign_artifacts(event)),
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
