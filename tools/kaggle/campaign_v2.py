"""Local bootstrap for the preregistered G008 paired campaign v2.

Preparing builds the exact packages/configuration for all thirteen frozen rows.
Registration is a separate explicit operation and is the only function here
that contacts Kaggle (quota only); it never submits a kernel.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import uuid
from typing import Any

from tools.kaggle import cli
from training.eval import registry as registry_module
from training.eval.g008_validation import _EXPECTED_ROSTERS as _VALIDATION_ROSTERS
from training.eval.provenance import seed_plan_sha256
from training.ppo.g008_benchmark import load_principal_config
from training.ppo.protocol import digest_json, file_sha256
from training.ppo.train import TrainConfig, g008_campaign_schedule_descriptor


CAMPAIGN_ID = "g008-competitive-20261021-v2"
CYCLE_ID = "g008-competitive-campaign-v2"
KERNEL_ID = "pascalcoimbra/circuit-ai"
REGISTRY_PATH = Path(".omx/evidence/g008-experiment-registry.jsonl")
MIN_QUOTA_SECONDS = 18_600
_ROLES = ("baseline", "mixed")
_PAIR_SEEDS = {"seed-pair-1": (3301, 2_100_001),
               "seed-pair-2": (3302, 2_200_001),
               "seed-pair-3": (3303, 2_300_001)}
_POOL_BPS = {
    "baseline": {"latest": 10_000, "step0": 0, "cruise": 0,
                  "local_mobility": 0, "interceptor": 0},
    "mixed": {"latest": 6_000, "step0": 1_000, "cruise": 1_000,
              "local_mobility": 1_000, "interceptor": 1_000},
}
_SELECTION_RULE = (
    "paired-score-gain-t95-lower-positive-no-pair-regresses-over-5pp-then-"
    "median-arm-checkpoint-v1"
)
_CREATED_AT_UTC = "2026-10-21T02:00:00Z"
_UUID_NAMESPACE = uuid.UUID("1c086c87-e8e1-4eaf-aa96-266790d0ef2c")


class CampaignBootstrapError(RuntimeError):
    """The frozen campaign cannot be safely prepared or registered."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _events(registry) -> list[dict[str, Any]]:
    try:
        events = registry.events()
    except Exception as exc:
        raise CampaignBootstrapError(f"Cannot read the verified experiment registry: {exc}") from exc
    if (len(events) != 55 or events[-1].get("event_index") != 54
            or events[-1].get("event_type") != "workload_attempt"):
        raise CampaignBootstrapError("G008 v2 requires exact current registry head event 54.")
    expected_types = {29: "registration", 48: "benchmark_artifact",
                      49: "registration_closed", 50: "benchmark_attestation"}
    for index, event_type in expected_types.items():
        if events[index].get("event_index") != index or events[index].get("event_type") != event_type:
            raise CampaignBootstrapError(f"Registry provenance anchor event {index} is not {event_type}.")
    if events[54].get("event_data", {}).get("terminal_status") != "COMPLETE":
        raise CampaignBootstrapError("Selected v17 training attempt is not reconciled COMPLETE.")
    if any(e.get("event_type") in {"g008_competitive_campaign_amendment_v1",
                                    "g008_competitive_campaign_amendment_v2"} for e in events):
        raise CampaignBootstrapError("A competitive campaign amendment already exists.")
    return events


def _source_inventory_sha256(package: Path) -> str:
    """Hash staged implementation/data sources, excluding workload-specific files."""
    excluded = {"training/eval/assets/g007-step-300.pt"}
    entries = []
    roots = (package / "training", package / "schemas")
    for root in roots:
        if not root.is_dir():
            raise CampaignBootstrapError(f"Prepared package is missing {root.relative_to(package)}.")
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(package).as_posix()):
            relative = path.relative_to(package).as_posix()
            if path.is_symlink():
                raise CampaignBootstrapError("Prepared package source inventory contains a symlink.")
            if path.is_file() and relative not in excluded and "__pycache__" not in path.parts:
                entries.append({"path": relative, "sha256": file_sha256(path)})
    entries.sort(key=lambda row: row["path"])
    return _sha(entries)


def _run_uuid(registration_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, registration_id))


def _training_settings(registration_id: str, run_uuid: str, pair_id: str,
                       role: str, root_seed: int, ppo_seed: int,
                       config: TrainConfig, seed_sha: str) -> dict[str, Any]:
    descriptor = g008_campaign_schedule_descriptor(
        run_uuid=run_uuid, root_seed=root_seed, arena_count=config.arena_count,
        total_updates=config.total_updates, rollout_steps=config.rollout_steps,
        seed_plan_sha=seed_sha)
    campaign = {
        "campaign_id": CAMPAIGN_ID, "registration_id": registration_id,
        "run_uuid": run_uuid, "pair_id": pair_id, "strategy_role": role,
        "root_seed": root_seed, "ppo_seed": ppo_seed, "update_budget": 1_000,
        "seed_plan_sha256": seed_sha, "opponent_probabilities": _POOL_BPS[role],
        "config_sha256": digest_json(config.__dict__, ""),
        "shared_config_sha256": digest_json({key: value for key, value in config.__dict__.items()
            if key not in {"device", "seed", "snapshot_opponent_probability",
                           "heuristic_opponent_probability"}}, ""),
        "schedule_descriptor_sha256": descriptor["schedule_descriptor_sha256"],
    }
    return {"run_id": run_uuid, "train_config": config.__dict__,
            "g008_campaign_train": campaign}


def _evaluation_settings(registration_id: str, run_uuid: str, mode: str,
                         roster: dict[str, Any], seed_sha: str) -> dict[str, Any]:
    evaluation = {
        "campaign_id": CAMPAIGN_ID, "registration_kind": mode, "mode": mode,
        "run_uuid": run_uuid,
        "candidate_dataset_source": "pascalcoimbra/g008-frozen-candidate",
        "candidate_checkpoint_path": "/kaggle/input/g008-frozen-candidate/candidate.pt",
        "roster_sha256": roster["roster_sha256"],
        "opponent_code_sha256": roster["opponent_code_sha256"],
        "game_count": 512, "seat_order": [0, 1], "seed_plan_sha256": seed_sha,
        "opponent_id": roster["opponent_id"],
    }
    if mode == "validation":
        checkpoint = (Path(__file__).resolve().parents[2]
                      / "artifacts/circuit-ai/runs/lc-circuit-ai-selfplay-300-viewer-ready"
                        "/checkpoints/step-300.pt")
        if not checkpoint.is_file():
            raise CampaignBootstrapError("Frozen G007 step-300 checkpoint is unavailable.")
        evaluation.update({"opponent_asset_path": "training/eval/assets/g007-step-300.pt",
                           "opponent_checkpoint_sha256": file_sha256(checkpoint)})
    return {"run_id": run_uuid, "g008_evaluation": evaluation}


def _wire_train_config(config: TrainConfig) -> dict[str, Any]:
    wire = dict(config.__dict__)
    for key in cli._CAMPAIGN_TRAIN_FLOAT_FIELDS:
        wire[key] = repr(float(wire[key]))
    return wire


def prepare_campaign_v2(output_dir: Path, *, registry=None,
                        created_at_utc: str | None = None) -> dict[str, Any]:
    """Prepare all 13 exact workload packages and return their amendment.

    No Kaggle API is called. Existing destinations are preserved and rejected.
    """
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise CampaignBootstrapError("Campaign output directory already exists; refusing overwrite.")
    if registry is None:
        from training.eval.registry import Registry
        registry = Registry(REGISTRY_PATH)
    events = _events(registry)
    registry_module._validate_recovery_cycle3_retry_suffix(events)
    event29_record = events[29]["event_data"]["record"]
    if event29_record.get("implementation_variants") or event29_record.get("allowed_implementation_variants"):
        variants = event29_record.get("allowed_implementation_variants", [])
        incumbent = next((v for v in variants if v.get("variant_id") == "incumbent"), None)
        backend_sha = (incumbent or {}).get("backend_sha256", event29_record.get("backend_sha256"))
    else:
        backend_sha = event29_record.get("backend_sha256")
    if not isinstance(backend_sha, str) or len(backend_sha) != 64:
        raise CampaignBootstrapError("Selected benchmark-attested backend identity is missing from event 29.")
    selected_training = event29_record
    selected_attempt = events[54]["event_data"]
    if (selected_attempt.get("registration_event_index") != 29
            or selected_attempt.get("registration_event_hash") != events[29]["event_hash"]
            or selected_attempt.get("implementation_variant_id") != "incumbent"
            or selected_attempt.get("terminal_status") != "COMPLETE"
            or selected_attempt.get("artifact_sha256") is not None):
        raise CampaignBootstrapError("Event 54 is not the exact artifact-free v17 selected-incumbent smoke run.")
    seed_sha = seed_plan_sha256()
    if seed_sha != registry_module._RECOVERY_SEED_PLAN_SHA256:
        raise CampaignBootstrapError("Local frozen seed plan does not match registry commitment.")
    frozen, _ = load_principal_config()
    train_module = Path(__import__("training.ppo.train", fromlist=["TrainConfig"]).__file__).resolve()
    engine_module = Path(__import__("training.arena.engine", fromlist=["step"]).__file__).resolve()
    trainer_sha, engine_sha = file_sha256(train_module), file_sha256(engine_module)
    package_root = output_dir / "packages"
    package_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    run_configs: dict[str, dict[str, Any]] = {}
    train_rows: dict[tuple[str, str], dict[str, Any]] = {}

    def build_row(registration_id: str, kind: str, run_uuid: str,
                  settings: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        config_path = output_dir / f"{registration_id}.run-config.json"
        config_path.write_text(json.dumps(settings, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        package = package_root / registration_id
        try:
            cli.prepare(KERNEL_ID, "g008-campaign-v2", package, config_path)
            package_sha = cli.package_tree_sha256(package)
            source_sha = _source_inventory_sha256(package)
            if (file_sha256(package / "training/ppo/train.py") != trainer_sha
                    or file_sha256(package / "training/arena/engine.py") != engine_sha):
                raise CampaignBootstrapError("Prepared package changed the frozen trainer or arena engine bytes.")
        except Exception as exc:
            raise CampaignBootstrapError(f"Could not prepare exact package {registration_id}: {exc}") from exc
        record.update({"package_sha256": package_sha, "source_sha256": source_sha,
                       "backend_sha256": backend_sha, "trainer_sha256": trainer_sha,
                       "engine_sha256": engine_sha, "seed_plan_sha256": seed_sha,
                       "kernel_id": KERNEL_ID, "accelerator_count": 1,
                       "machine_shape": "NvidiaTeslaT4", "open_space_allowed": False,
                       "session_timeout_seconds": 1_200 if kind == "train" else 600,
                       "workload_gpu_seconds_cap": 1_200 if kind == "train" else 600,
                       "workload_output_bytes_cap": 100_000_000})
        record_sha = _sha(record)
        run_configs[registration_id] = settings
        return {"registration_id": registration_id, "registration_kind": kind,
                "record_sha256": record_sha, "record": record}

    for pair_id, (ppo_seed, root_seed) in _PAIR_SEEDS.items():
        for role in _ROLES:
            registration_id = f"g008-v2-train-{pair_id}-{role}"
            run_uuid = _run_uuid(registration_id)
            config = replace(frozen, device="auto", seed=ppo_seed, total_updates=1_000,
                             checkpoint_every=50,
                             snapshot_opponent_probability=0.1 if role == "mixed" else 0.0,
                             heuristic_opponent_probability=(1 / 3) if role == "mixed" else 0.0)
            config.validate()
            settings = _training_settings(registration_id, run_uuid, pair_id, role,
                                          root_seed, ppo_seed, config, seed_sha)
            train_config = _wire_train_config(config)
            record = {
                "campaign_id": CAMPAIGN_ID, "run_uuid": run_uuid,
                "registration_kind": "train", "strategy_role": role,
                "pair_id": pair_id, "ppo_seed": ppo_seed, "root_seed": root_seed,
                "update_budget": 1_000, "shared_config_sha256": settings["g008_campaign_train"]["shared_config_sha256"],
                "train_config": train_config,
                "architecture": {"width": config.width, "height": config.height,
                                 "hidden_size": config.hidden_size},
                "opponent_probabilities": _POOL_BPS[role],
                "config_sha256": digest_json(config.__dict__, ""),
                "schedule_descriptor_sha256": settings["g008_campaign_train"]["schedule_descriptor_sha256"],
            }
            row = build_row(registration_id, "train", run_uuid, settings, record)
            rows.append(row)
            train_rows[pair_id, role] = row

    paired_seeds_sha = _sha([{"pair_id": pair_id, "ppo_seed": ppo,
                              "root_seed": root, "seat_order": [0, 1]}
                             for pair_id, (ppo, root) in _PAIR_SEEDS.items()])
    validation_roster = {
        "roster_sha256": registry_module._VALIDATION_OPPONENT_ROSTER_SHA256,
        "opponent_code_sha256": registry_module._VALIDATION_OPPONENT_SHA256,
        "game_count": 512, "seat_order": [0, 1],
        "paired_seeds_sha256": paired_seeds_sha,
    }
    for pair_id in _PAIR_SEEDS:
        for role in _ROLES:
            training = train_rows[pair_id, role]
            registration_id = f"g008-v2-validation-{pair_id}-{role}"
            run_uuid = _run_uuid(registration_id)
            settings = _evaluation_settings(registration_id, run_uuid, "validation",
                                            _VALIDATION_ROSTERS["validation"], seed_sha)
            evaluation_config = settings["g008_evaluation"]
            record = {
                "campaign_id": CAMPAIGN_ID, "run_uuid": run_uuid,
                "registration_kind": "validation", "strategy_role": role,
                "pair_id": pair_id, "training_registration_id": training["registration_id"],
                "training_run_uuid": training["record"]["run_uuid"],
                "training_record_sha256": training["record_sha256"],
                "roster_sha256": validation_roster["roster_sha256"],
                "opponent_code_sha256": validation_roster["opponent_code_sha256"],
                "paired_seeds_sha256": paired_seeds_sha, "game_count": 512,
                "seat_order": [0, 1],
                "config_sha256": digest_json(evaluation_config, "__no_digest_field__"),
            }
            rows.append(build_row(registration_id, "validation", run_uuid, settings, record))

    final_id = "g008-v2-final-suite"
    final_uuid = _run_uuid(final_id)
    from training.eval.g008_final_roster import EXPECTED_FINAL_ROSTER
    final_settings = _evaluation_settings(final_id, final_uuid, "final",
                                          EXPECTED_FINAL_ROSTER, seed_sha)
    final_config = final_settings["g008_evaluation"]
    final_record = {
        "campaign_id": CAMPAIGN_ID, "run_uuid": final_uuid,
        "registration_kind": "final", "strategy_role": "final",
        "roster_sha256": EXPECTED_FINAL_ROSTER["roster_sha256"],
        "opponent_code_sha256": EXPECTED_FINAL_ROSTER["opponent_code_sha256"],
        "game_count": 512, "seat_order": [0, 1],
        "config_sha256": digest_json(final_config, "__no_digest_field__"),
        "pre_final_data_visible": False,
    }
    rows.append(build_row(final_id, "final", final_uuid, final_settings, final_record))

    created = created_at_utc or _CREATED_AT_UTC
    if not created.endswith("Z"):
        raise CampaignBootstrapError("created_at_utc must be UTC and end in Z.")
    validation_roster["paired_seeds_sha256"] = paired_seeds_sha
    amendment = {
        "schema_id": "g008-competitive-campaign", "schema_version": 2,
        "campaign_id": CAMPAIGN_ID, "created_at_utc": created,
        "prior_head_hash": events[54]["event_hash"], "prior_final_event_index": 54,
        "benchmark_artifact_event_index": 48, "benchmark_artifact_event_hash": events[48]["event_hash"],
        "benchmark_lock_event_index": 49, "benchmark_lock_event_hash": events[49]["event_hash"],
        "benchmark_attestation_event_index": 50, "benchmark_attestation_event_hash": events[50]["event_hash"],
        "selected_training_registration_event_index": 29,
        "selected_training_registration_event_hash": events[29]["event_hash"],
        "selected_training_attempt_event_index": 54,
        "selected_training_attempt_event_hash": events[54]["event_hash"],
        "selected_training_run_uuid": selected_training["run_uuid"],
        "kernel_id": KERNEL_ID, "current_version_number": 17,
        "training_gpu_seconds_cap": 7_200, "evaluation_gpu_seconds_cap": 4_200,
        "total_gpu_seconds_cap": 11_400, "protected_final_reserve_gpu_seconds": 7_200,
        "output_caps_bytes": {"aggregate": 5_000_000_000, "per_run": 100_000_000},
        "quota_gpu_seconds_remaining_before": MIN_QUOTA_SECONDS,
        "validation_roster": validation_roster,
        "candidate_selection_rule_id": _SELECTION_RULE,
        "records": rows,
    }
    registry_module._validate_data("g008_competitive_campaign_amendment_v2", amendment)
    (output_dir / "campaign-v2-amendment.json").write_text(
        json.dumps(amendment, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return amendment


def register_campaign_v2(amendment: dict[str, Any], output_dir: Path, *, registry=None,
                         api=None) -> list[dict[str, Any]]:
    """Explicitly append the campaign after live quota and exact-head checks.

    This asks Kaggle only for GPU quota and does not submit or inspect kernels.
    """
    if registry is None:
        from training.eval.registry import Registry
        registry = Registry(REGISTRY_PATH)
    events = _events(registry)
    if (amendment.get("prior_head_hash") != events[54]["event_hash"]
            or amendment.get("benchmark_artifact_event_hash") != events[48]["event_hash"]
            or amendment.get("benchmark_lock_event_hash") != events[49]["event_hash"]
            or amendment.get("benchmark_attestation_event_hash") != events[50]["event_hash"]):
        raise CampaignBootstrapError("Prepared amendment no longer matches exact registry anchors.")
    output_dir = Path(output_dir).resolve()
    package_root = output_dir / "packages"
    for row in amendment.get("records", []):
        registration_id = row.get("registration_id")
        package = package_root / str(registration_id)
        config_path = package / "run-config.json"
        try:
            observed_package_sha = cli.package_tree_sha256(package)
            observed_source_sha = _source_inventory_sha256(package)
            packaged_config = json.loads(config_path.read_text(encoding="utf-8"))
            expected_config = cli._load_run_settings(
                output_dir / f"{registration_id}.run-config.json")
        except Exception as exc:
            raise CampaignBootstrapError(
                f"Prepared package for {registration_id} is unavailable or invalid: {exc}") from exc
        if (observed_package_sha != row["record"].get("package_sha256")
                or observed_source_sha != row["record"].get("source_sha256")
                or packaged_config != expected_config
                or packaged_config.get("run_id") != row["record"].get("run_uuid")):
            raise CampaignBootstrapError(
                f"Prepared package for {registration_id} no longer matches registered hashes/identity.")
    if api is None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        except (ImportError, AttributeError) as exc:
            raise CampaignBootstrapError("Kaggle CLI SDK is unavailable or unauthenticated.") from exc
    try:
        remaining, _ = cli._quota_seconds(api)
    except Exception as exc:
        raise CampaignBootstrapError(f"Unable to verify live Kaggle GPU quota: {exc}") from exc
    if remaining < MIN_QUOTA_SECONDS:
        raise CampaignBootstrapError(
            f"Kaggle quota has {remaining} GPU-s; v2 requires at least {MIN_QUOTA_SECONDS} GPU-s.")
    amendment = dict(amendment)
    amendment["quota_gpu_seconds_remaining_before"] = remaining
    registry_module._validate_data("g008_competitive_campaign_amendment_v2", amendment)
    try:
        return registry.register_g008_competitive_campaign_v2_batch(amendment)
    except Exception as exc:
        raise CampaignBootstrapError(f"Atomic v2 registry append failed: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--register", action="store_true",
                        help="explicitly append after live Kaggle quota verification")
    args = parser.parse_args(argv)
    from training.eval.registry import Registry
    registry = Registry(args.registry)
    if args.register:
        amendment_path = args.output_dir / "campaign-v2-amendment.json"
        if amendment_path.is_file():
            try:
                amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignBootstrapError(
                    f"Prepared campaign amendment is unavailable or invalid: {exc}") from exc
        else:
            amendment = prepare_campaign_v2(args.output_dir, registry=registry)
        register_campaign_v2(amendment, args.output_dir, registry=registry)
        print("Registered exact 13-row G008 campaign v2; no Kaggle submission was made.")
    else:
        amendment = prepare_campaign_v2(args.output_dir, registry=registry)
        print(f"Prepared 13 exact G008 campaign packages under {args.output_dir}; registry unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
