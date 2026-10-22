import hashlib
import json
import importlib.util
import sys
import types
from dataclasses import fields
from functools import lru_cache
from pathlib import Path

import pytest

from tools.kaggle.cli import (KaggleError, _campaign_evaluation_output_root,
                              _load_run_settings, _run, _sdk_metadata, download, poll_status, prepare, submit)
from training.arena.engine import ACTION_ORDER, initialize_episode, step
from training.ppo.protocol import digest_json, file_sha256
from training.ppo.train import TrainConfig


def test_kaggle_command_failure_reports_safe_rate_limit_class(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "tools.kaggle.cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="429 Too Many Requests https://private.example/token"
        ),
    )
    with pytest.raises(KaggleError, match="rate limited") as error:
        _run(["kaggle", "kernels", "push"])
    assert "private.example" not in str(error.value)


def g008_principal_train_config() -> dict:
    from training.ppo.g008_benchmark import load_principal_config

    config, _ = load_principal_config()
    return {field.name: "auto" if field.name == "device" else getattr(config, field.name)
            for field in fields(TrainConfig)}


def make_run(root: Path, run_id="run-42") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir()
    checkpoint_refs, checkpoint_metadata = [], []
    for index, checkpoint_id in enumerate(("step-0", "step-1")):
        artifacts = {}
        for fmt, extension in (("pytorch", "pt"), ("onnx", "onnx")):
            rel = f"checkpoints/{checkpoint_id}.{extension}"
            (root / rel).write_bytes(f"{checkpoint_id}-{fmt}".encode())
            artifacts[fmt] = {"file": rel, "sha256": file_sha256(root / rel)}
            checkpoint_refs.append({"checkpoint_id": checkpoint_id, "step": index,
                                    "file": rel, "sha256": artifacts[fmt]["sha256"],
                                    "format": fmt, "observation_schema": "lc-observation/1",
                                    "action_schema": "lc-action/1"})
        metadata = {
            "format": "lc-checkpoint/1", "checkpoint_id": checkpoint_id, "step": index,
            "model_architecture": {"name": "test-mlp", "version": "1", "config": {}},
            "observation_schema": "lc-observation/1", "action_schema": "lc-action/1",
            "onnx": {**artifacts["onnx"], "opset": 17, "runtime": "onnxruntime-web"},
            "pytorch": {**artifacts["pytorch"], "format": "state_dict+optimizer"},
            "artifacts": [{"format": fmt, **artifacts[fmt]} for fmt in ("onnx", "pytorch")],
        }
        metadata["metadata_sha256"] = digest_json(metadata, "metadata_sha256")
        metadata_rel = f"checkpoints/{checkpoint_id}.metadata.json"
        (root / metadata_rel).write_text(json.dumps(metadata), encoding="utf-8")
        checkpoint_metadata.append(metadata_rel)

    initial, streams = initialize_episode(4, 4, 0, 0)
    replay = {
        "format": "lc-replay/1", "run_id": run_id, "match_id": "match-0",
        "engine_version": "lc-engine/1", "observation_schema": "lc-observation/1",
        "action_schema": "lc-action/1", "master_seed": "0", "arena_index": 0,
        "derived_streams": [{"label": stream.label, "initstate": str(stream.initstate),
                             "initseq": str(stream.initseq)} for stream in streams],
        "agents": [{"agent_id": "agent-0", "checkpoint_id": "step-1"},
                   {"agent_id": "agent-1", "checkpoint_id": "step-0"}],
        "initial_state": {"width": initial.width, "height": initial.height,
                          "positions": [{"x": a.position[0], "y": a.position[1]} for a in initial.agents],
                          "headings": [ACTION_ORDER[a.heading] for a in initial.agents],
                          "trails": [[{"x": x, "y": y} for x, y in a.trail] for a in initial.agents]},
        "ticks": [], "result": "draw", "tick_count": 0,
    }
    current = initial
    while current.result is None:
        actions = [agent.heading for agent in current.agents]
        transition = step(current, actions)
        terminals = [f"agent_{i}_dead" for i, agent in enumerate(transition.state.agents) if not agent.alive]
        replay["ticks"].append({"tick": current.tick, "joint_actions": actions,
                                "collision_causes": [list(causes) for causes in transition.causes],
                                "terminal_causes": terminals})
        current = transition.state
    replay["result"] = current.result
    replay["tick_count"] = len(replay["ticks"])
    replay["replay_sha256"] = digest_json(replay, "replay_sha256")
    (root / "replays/replay-0.json").parent.mkdir()
    (root / "replays/replay-0.json").write_text(json.dumps(replay), encoding="utf-8")
    seeds = [str(i) for i in range(32)]
    games = [
        {"seed": str(i // 2), "target_seat": i % 2, "result": "draw", "ticks": 0,
         "collision_causes": [], "replay_file": "replays/replay-0.json"}
        for i in range(64)
    ]
    summary = {"wins": 0, "draws": 64, "losses": 0, "win_rate": 0,
               "wilson_95": [0, 0.1], "mean_survival_ticks": 0,
               "median_survival_ticks": 0, "collision_causes": {}}
    report = {"format": "lc-eval/1", "run_id": run_id, "checkpoint_id": "step-1",
              "protocol": "lc-eval/1", "held_out_seeds": seeds,
              "matchups": [{"opponent_id": opponent, "games": games, "summary": summary}
                           for opponent in ("open-space", "step-0")]}
    report["report_sha256"] = digest_json(report, "report_sha256")
    (root / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    files = []
    all_paths = [ref["file"] for ref in checkpoint_refs] + checkpoint_metadata + ["replays/replay-0.json", "evaluation.json"]
    for rel in all_paths:
        files.append({"path": rel, "sha256": file_sha256(root / rel)})
    manifest = {
        "format": "lc-run/1", "run_id": run_id, "created_utc": "2026-10-19T00:00:00Z",
        "engine_version": "lc-engine/1", "observation_schema": "lc-observation/1",
        "action_schema": "lc-action/1", "reward_schema": "lc-reward/1",
        "replay_schema": "lc-replay/1", "master_seed": "0", "training_config": {},
        "training_steps": 1,
        "checkpoints": checkpoint_refs,
        "evaluation": {"protocol": "lc-eval/1", "held_out_seeds": seeds,
                       "games_per_matchup": 64, "opponent_ids": ["open-space", "step-0"],
                       "metrics_file": "evaluation.json", "replay_files": ["replays/replay-0.json"]},
        "software": {}, "files": files,
    }
    manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_campaign_evaluation_output_root_resolves_exact_kaggle_run(tmp_path):
    run_uuid = "c19ca811-b968-562f-a813-6f249547109d"
    output = tmp_path / "download"
    run_root = output / "arena-output" / run_uuid
    run_root.mkdir(parents=True)
    (run_root / "g008-evaluation.json").write_text("{}", encoding="utf-8")

    assert _campaign_evaluation_output_root(output, run_uuid) == run_root


def test_campaign_evaluation_output_root_accepts_direct_output(tmp_path):
    (tmp_path / "g008-evaluation.json").write_text("{}", encoding="utf-8")

    assert _campaign_evaluation_output_root(tmp_path, "unused") == tmp_path


def make_g008_benchmark(root: Path, run_id="g008-run") -> None:
    import uuid
    from training.eval.provenance import (benchmark_run_keys, schedule_spec_digest,
                                          seed_plan_sha256)
    from training.ppo.g008_benchmark import CONFIG_PATH, load_principal_config

    root.mkdir(parents=True, exist_ok=True)
    run_keys = benchmark_run_keys()
    run_ids = {key: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{key}")) for key in run_keys}
    schedule_digests = dict(_cached_benchmark_schedule_digests(
        seed_plan_sha256(), schedule_spec_digest(), tuple(run_ids.items())))
    principal_config, config_document = load_principal_config()
    roster = [{"run_key": key, "run_id": run_ids[key],
               "expanded_schedule_digest": schedule_digests[key]}
              for key in run_keys]
    optimizer_updates_per_training_update = 4

    def run_metrics(update_count, repeat_index=0):
        total = update_count * optimizer_updates_per_training_update
        update_logs = [{"device": "cuda", "cuda_updates": optimizer_updates_per_training_update,
                        "optimizer_updates": optimizer_updates_per_training_update,
                        "rollout_seconds": 1.0 + repeat_index,
                        "optimizer_seconds": 2.0 + repeat_index,
                        "checkpoint_seconds": 0.1,
                        "update_latency_seconds": 3.0 + repeat_index}
                       for _ in range(update_count)]
        return {"format": "circuit-ai-g008-benchmark-run-metrics/1",
                "device": {"device": "cuda", "cuda_available": True, "cuda_name": "Tesla T4",
                           "optimizer_updates": total, "cuda_optimizer_updates": total},
                "training": {"updates": update_logs,
                             "mean_rollout_seconds": 1.0 + repeat_index,
                             "update_elapsed_seconds": update_count * (3.0 + repeat_index),
                             "elapsed_seconds": update_count * (4.0 + repeat_index)}}

    device = {"device": "cuda", "cuda_available": True, "cuda_name": "Tesla T4",
              "optimizer_updates": optimizer_updates_per_training_update,
              "cuda_optimizer_updates": optimizer_updates_per_training_update}
    repeats = []
    for index in range(5):
        run_key = f"pair-{index:02d}"
        metrics = run_metrics(3, index)
        path = root / "measured" / f"repeat-{index:02d}" / "training-metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics))
        transitions = principal_config.arena_count * principal_config.rollout_steps * principal_config.total_updates
        rollout = (1.0 + index) * 3
        optimizer = (2.0 + index) * 3
        checkpoint = sum(0.1 for _ in range(3))
        update_only = sum((3.0 + index) - 0.1 for _ in range(3))
        wall = (4.0 + index) * 3
        repeats.append({"repeat": index, "run_key": run_key, "run_id": run_ids[run_key],
                        "completed_joint_arena_transitions": transitions,
                        "agent_decisions": transitions * 2,
                        "update_only_joint_transitions_per_second": transitions / update_only,
                        "wall_inclusive_joint_transitions_per_second": transitions / wall,
                        "update_only_agent_decisions_per_second": transitions * 2 / update_only,
                        "wall_inclusive_agent_decisions_per_second": transitions * 2 / wall,
                        "rollout_seconds": rollout, "optimizer_seconds": optimizer,
                        "update_only_seconds": update_only,
                        "checkpoint_inclusive_update_elapsed_seconds": 3 * (3.0 + index),
                        "checkpoint_seconds": checkpoint, "wall_seconds": wall,
                        "expanded_schedule_digest": schedule_digests[run_key],
                        "optimizer_updates": 3 * optimizer_updates_per_training_update,
                        "cuda_optimizer_updates": 3 * optimizer_updates_per_training_update,
                        "training_metrics_sha256": file_sha256(path)})
    warmup_hashes = []
    for index in range(2):
        run_key = f"warmup-python-oracle-incumbent-{index:02d}"
        path = root / "warmups" / f"warmup-{index:02d}" / "training-metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_metrics(1)))
        warmup_hashes.append(file_sha256(path))
    warmups = [{"run_key": f"warmup-python-oracle-incumbent-{index:02d}",
                "run_id": run_ids[f"warmup-python-oracle-incumbent-{index:02d}"],
                "expanded_schedule_digest": schedule_digests[f"warmup-python-oracle-incumbent-{index:02d}"],
                "training_metrics_sha256": warmup_hashes[index]}
               for index in range(2)]
    summary_fields = ("update_only_joint_transitions_per_second", "wall_inclusive_joint_transitions_per_second",
                      "update_only_agent_decisions_per_second", "wall_inclusive_agent_decisions_per_second",
                      "rollout_seconds", "optimizer_seconds", "update_only_seconds",
                      "checkpoint_inclusive_update_elapsed_seconds", "checkpoint_seconds", "wall_seconds",
                      "completed_joint_arena_transitions", "agent_decisions")
    from statistics import median
    def pct(values):
        values = sorted(values)
        q1 = median(values[:len(values) // 2])
        q3 = median(values[(len(values) + 1) // 2:])
        return {"median": median(values), "min": min(values), "max": max(values), "iqr": q3 - q1}
    report = {"format": "circuit-ai-g008-principal-benchmark/2", "measured_repeats": 5,
              "updates_per_repeat": 3,
              "source": config_document["source"],
              "config": principal_config.__dict__,
              "config_sha256": config_document["config_sha256"],
              "config_artifact_sha256": file_sha256(CONFIG_PATH),
              "seed_plan_sha256": seed_plan_sha256(), "schedule_spec_digest": schedule_spec_digest(),
              "schedule_roster": roster, "warmups": {"count": 2, "updates_each": 1, "runs": warmups},
              "device": device,
              "summary": {field: pct([repeat[field] for repeat in repeats]) for field in summary_fields},
              "optimizer_updates_total": sum(repeat["optimizer_updates"] for repeat in repeats),
              "cuda_optimizer_updates_total": sum(repeat["cuda_optimizer_updates"] for repeat in repeats),
              "repeats": repeats}
    report_path = root / "g008-principal-benchmark.json"
    report_path.write_text(json.dumps(report))
    evidence = {"format": "circuit-ai-kaggle-g008-benchmark-evidence/1", "run_id": run_id,
                "python_version": "3.11.11", "torch_version": "2.5.1+cu124", "cuda_version": "12.4",
                "visible_cuda_devices": ["Tesla T4", "Tesla T4"], "visible_cuda_device_count": 2,
                "training_device_index": 0, "training_cuda_device_count": 1,
                "benchmark_report_sha256": file_sha256(report_path), "benchmark_report": report}
    (root / "kaggle-execution-evidence.json").write_text(json.dumps(evidence))


@lru_cache(maxsize=8)
def _cached_benchmark_schedule_digests(seed_plan_hash, spec_digest, run_id_pairs):
    """Reuse only immutable test digests for the same frozen 191k-slot roster."""
    from training.ppo.g008_benchmark import prepare_benchmark_schedules

    schedules = prepare_benchmark_schedules(dict(run_id_pairs))
    return tuple((key, schedules[key]["expanded_schedule_digest"])
                 for key, _ in run_id_pairs)


def _embedded_package_files(package: Path) -> dict[str, bytes]:
    import base64
    import re
    import zipfile

    runner = (package / "run_kaggle.py").read_text(encoding="utf-8")
    match = re.search(r'SOURCE_ARCHIVE_B64 = "([^"]+)"', runner)
    assert match is not None
    with zipfile.ZipFile(__import__("io").BytesIO(base64.b64decode(match.group(1)))) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


def _assert_nonfinal_campaign_archive_has_no_final_holdout(package: Path) -> None:
    files = _embedded_package_files(package)
    forbidden_paths = {
        "training/configs/g008-split-seed-plan.json",
        "training/eval/provenance.py", "training/eval/registry.py",
        "training/eval/lc_eval.py", "training/eval/g008_final_roster.py",
        "training/eval/g008_final_opponent.py",
        "training/ppo/g008_benchmark.py",
        "training/ppo/g008_incumbent/train.py",
        "schemas/g008-split-seed-plan-3.schema.json",
    }
    assert forbidden_paths.isdisjoint(files)
    assert "training/eval/g008_campaign_seed.py" in files
    source = b"\n".join(files.values()).lower()
    for token in (
            b"open-space", b"open_space", b"4000000", b"4_000_000",
            b"final-suite-v1", b"0fa20d714149054782dd5a83ab9ae99ebca6582b6c5702bdcc206bf7e3b61c27",
            b"9e5103176218ca092206ba50899a35c43ad488ce825a95af487e76798ce60123",
            b"4043dda35bb8cc2adc43112cb5b00eff50ac721eb062b15d0d83b7ae41510b7f"):
        assert token not in source


def test_submit_requires_explicit_confirmation_and_validates_source(tmp_path):
    source = tmp_path / "kernel"
    source.mkdir()
    (source / "kernel-metadata.json").write_text(json.dumps({"id": "me/light-cycles"}))
    (source / ".arena-source-revision").write_text("local-r7")
    calls = []
    run = lambda args: calls.append(list(args)) or "Complete"
    with pytest.raises(KaggleError, match="requires --confirm"):
        submit("me/light-cycles", "local-r7", source, confirm=False, run=run)
    assert calls == []
    output = submit("me/light-cycles", "local-r7", source, confirm=False, dry_run=True, run=run)
    assert output.startswith("DRY RUN:") and calls == []
    with pytest.raises(KaggleError, match="does not match"):
        submit("me/other", "local-r7", source, confirm=True, run=run)
    assert calls == []


def test_confirmed_submit_checks_running_then_pushes_selected_source(tmp_path):
    source = tmp_path / "kernel"
    source.mkdir()
    (source / "kernel-metadata.json").write_text(json.dumps({"id": "me/light-cycles"}))
    (source / ".arena-source-revision").write_text("rev-abc")
    calls = []
    def run(args):
        calls.append(list(args))
        if args[2] == "list":
            return "ref title\nme/light-cycles Light Cycles"
        return "Kernel status: Complete" if args[2] == "status" else "Kernel version 4 successfully pushed"
    assert "version 4" in submit("me/light-cycles", "rev-abc", source, confirm=True, run=run)
    assert calls[0] == ["kaggle", "kernels", "list", "--user", "me", "--search", "light-cycles", "--page-size", "100"]
    assert calls[1] == ["kaggle", "kernels", "status", "me/light-cycles"]
    assert calls[2][:4] == ["kaggle", "kernels", "push", "-p"]
    calls.clear()
    with pytest.raises(KaggleError, match="already running"):
        submit("me/light-cycles", "rev-abc", source, confirm=True,
               run=lambda args: calls.append(list(args)) or
               ("ref title\nme/light-cycles Light Cycles" if args[2] == "list" else "Kernel status: Running"))
    assert [call[2] for call in calls] == ["list", "status"]


def test_confirmed_submit_pushes_new_kernel_without_status_lookup(tmp_path):
    source = tmp_path / "kernel"
    source.mkdir()
    (source / "kernel-metadata.json").write_text(json.dumps({"id": "me/new-kernel"}))
    (source / ".arena-source-revision").write_text("rev-new")
    calls = []
    def run(args):
        calls.append(list(args))
        if args[2] == "list":
            return "Not found"
        return "Kernel version 1 successfully pushed"
    assert "version 1" in submit("me/new-kernel", "rev-new", source, confirm=True, run=run)
    assert [call[2] for call in calls] == ["list", "push"]


def test_submit_fails_closed_on_malformed_kernel_list(tmp_path):
    source = tmp_path / "kernel"
    source.mkdir()
    (source / "kernel-metadata.json").write_text(json.dumps({"id": "me/light-cycles"}))
    (source / ".arena-source-revision").write_text("rev-abc")
    calls = []
    with pytest.raises(KaggleError, match="unrecognized response"):
        submit("me/light-cycles", "rev-abc", source, confirm=True,
               run=lambda args: calls.append(list(args)) or "unexpected listing response")
    assert [call[2] for call in calls] == ["list"]


def test_polling_is_bounded_and_distinguishes_terminal_statuses():
    sleeps = []
    replies = iter(("Kernel status: Running", "Kernel status: Complete"))
    state, _ = poll_status("me/k", 3, max_attempts=2, interval_seconds=0.25,
                           run=lambda args: next(replies), sleep=sleeps.append)
    assert state == "success" and sleeps == [0.25]
    state, _ = poll_status("me/k", 3, max_attempts=1,
                           run=lambda args: "Kernel status: Running")
    assert state == "timeout"
    state, _ = poll_status("me/k", 3, max_attempts=1,
                           run=lambda args: "Kernel status: Error")
    assert state == "error"
    state, _ = poll_status("me/k", 3, max_attempts=1,
                           run=lambda args: 'KernelWorkerStatus.CANCEL_ACKNOWLEDGED')
    assert state == "error"


def test_download_validates_stages_and_atomically_promotes(tmp_path):
    destination = tmp_path / "artifacts/run-42"
    calls = []
    def run(args):
        calls.append(list(args))
        if args[2] == "status":
            return "Kernel status: Complete"
        make_run(Path(args[args.index("-p") + 1]))
        return "downloaded"
    result = download("me/light-cycles", 5, "run-42", destination, run=run)
    assert result == destination.resolve()
    assert (destination / "manifest.json").is_file()
    assert calls[-1][:4] == ["kaggle", "kernels", "output", "me/light-cycles/5"]
    with pytest.raises(KaggleError, match="already exists"):
        download("me/light-cycles", 5, "run-42", destination, run=run)


def test_download_promotes_exact_nested_kaggle_working_run(tmp_path):
    destination = tmp_path / "artifacts/run-42"
    def run(args):
        if args[2] == "status":
            return "Kernel status: Complete"
        stage = Path(args[args.index("-p") + 1])
        (stage / "__results__.html").write_text("Kaggle wrapper output")
        nested = stage / "arena-output" / "run-42"
        make_run(nested)
        make_run(stage / "arena-output" / "unrelated-run", "unrelated-run")
        return "downloaded"
    result = download("me/light-cycles", 5, "run-42", destination, run=run)
    assert (result / "manifest.json").is_file()
    assert not (result / "__results__.html").exists()
    assert not (result.parent / "unrelated-run").exists()


def test_download_validates_and_promotes_g008_principal_benchmark(tmp_path):
    destination = tmp_path / "artifacts/g008-run"
    def run(args):
        if args[2] == "status":
            return "Kernel status: Complete"
        stage = Path(args[args.index("-p") + 1])
        (stage / "__results__.html").write_text("wrapper")
        make_g008_benchmark(stage / "arena-output" / "g008-run")
        return "downloaded"

    result = download("me/circuit-ai", 9, "g008-run", destination, run=run)
    assert result == destination.resolve()
    assert (destination / "g008-principal-benchmark.json").is_file()
    assert len(list((destination / "measured").glob("repeat-*/training-metrics.json"))) == 5
    assert not (destination / "__results__.html").exists()


@pytest.mark.parametrize("mutation", [
    "old-format", "missing-plan-hash", "wrong-plan-hash", "missing-spec-digest",
    "stale-spec-digest", "missing-roster", "altered-roster", "stale-repeat-binding",
    "stale-warmup-binding", "stale-warmup-metrics-hash", "missing-config",
    "malformed-config", "altered-config", "missing-config-sha", "wrong-config-sha",
    "missing-config-artifact-sha", "wrong-config-artifact-sha",
    "missing-source", "wrong-source", "missing-optimizer-updates-total",
    "wrong-optimizer-updates-total", "missing-cuda-optimizer-updates-total",
    "wrong-cuda-optimizer-updates-total", "boolean-optimizer-updates-total",
])
def test_g008_report_import_rejects_stale_or_incomplete_lineage(tmp_path, mutation):
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    root = tmp_path / "g008-run"
    make_g008_benchmark(root)
    report_path = root / "g008-principal-benchmark.json"
    report = json.loads(report_path.read_text())
    if mutation == "old-format":
        report["format"] = "circuit-ai-g008-principal-benchmark/1"
    elif mutation == "missing-plan-hash":
        report.pop("seed_plan_sha256")
    elif mutation == "wrong-plan-hash":
        report["seed_plan_sha256"] = "0" * 64
    elif mutation == "missing-spec-digest":
        report.pop("schedule_spec_digest")
    elif mutation == "stale-spec-digest":
        report["schedule_spec_digest"] = "0" * 64
    elif mutation == "missing-roster":
        report.pop("schedule_roster")
    elif mutation == "altered-roster":
        report["schedule_roster"].pop()
    elif mutation == "stale-repeat-binding":
        report["repeats"][0]["expanded_schedule_digest"] = "0" * 64
    elif mutation == "stale-warmup-binding":
        report["warmups"]["runs"][0]["run_id"] = "00000000-0000-4000-8000-000000000000"
    elif mutation == "stale-warmup-metrics-hash":
        report["warmups"]["runs"][0]["training_metrics_sha256"] = "0" * 64
    elif mutation == "missing-config":
        report.pop("config")
    elif mutation == "malformed-config":
        report["config"] = "not-an-object"
    elif mutation == "altered-config":
        report["config"]["arena_count"] += 1
    elif mutation == "missing-config-sha":
        report.pop("config_sha256")
    elif mutation == "wrong-config-sha":
        report["config_sha256"] = "0" * 64
    elif mutation == "missing-config-artifact-sha":
        report.pop("config_artifact_sha256")
    elif mutation == "wrong-config-artifact-sha":
        report["config_artifact_sha256"] = "0" * 64
    elif mutation == "missing-source":
        report.pop("source")
    elif mutation == "wrong-source":
        report["source"] = "different-frozen-source"
    elif mutation == "missing-optimizer-updates-total":
        report.pop("optimizer_updates_total")
    elif mutation == "wrong-optimizer-updates-total":
        report["optimizer_updates_total"] -= 1
    elif mutation == "missing-cuda-optimizer-updates-total":
        report.pop("cuda_optimizer_updates_total")
    elif mutation == "wrong-cuda-optimizer-updates-total":
        report["cuda_optimizer_updates_total"] -= 1
    elif mutation == "boolean-optimizer-updates-total":
        report["optimizer_updates_total"] = True
    report_path.write_text(json.dumps(report))
    evidence_path = root / "kaggle-execution-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["benchmark_report"] = report
    evidence["benchmark_report_sha256"] = file_sha256(report_path)
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(KaggleError):
        validate_principal_benchmark_artifacts(root, "g008-run")


@pytest.mark.parametrize("run_kind", ["measured", "warmup"])
def test_g008_report_import_rejects_non_object_device_evidence(tmp_path, run_kind):
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    root = tmp_path / "g008-run"
    make_g008_benchmark(root)
    report_path = root / "g008-principal-benchmark.json"
    report = json.loads(report_path.read_text())
    index = 0
    if run_kind == "measured":
        metrics_path = root / "measured" / "repeat-00" / "training-metrics.json"
        report["repeats"][index]["training_metrics_sha256"] = file_sha256(metrics_path)
    else:
        metrics_path = root / "warmups" / "warmup-00" / "training-metrics.json"
        report["warmups"]["runs"][index]["training_metrics_sha256"] = file_sha256(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    metrics["device"] = "malformed"
    metrics_path.write_text(json.dumps(metrics))
    if run_kind == "measured":
        report["repeats"][index]["training_metrics_sha256"] = file_sha256(metrics_path)
    else:
        report["warmups"]["runs"][index]["training_metrics_sha256"] = file_sha256(metrics_path)
    report_path.write_text(json.dumps(report))
    evidence_path = root / "kaggle-execution-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["benchmark_report"] = report
    evidence["benchmark_report_sha256"] = file_sha256(report_path)
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(KaggleError, match="device evidence is invalid"):
        validate_principal_benchmark_artifacts(root, "g008-run")


@pytest.mark.parametrize("run_kind", ["measured", "warmup"])
@pytest.mark.parametrize("mutation", ["missing", "mismatched"])
def test_g008_report_import_rejects_run_cuda_name_not_matching_report(tmp_path, run_kind, mutation):
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    root = tmp_path / "g008-run"
    make_g008_benchmark(root)
    report_path = root / "g008-principal-benchmark.json"
    report = json.loads(report_path.read_text())
    if run_kind == "measured":
        metrics_path = root / "measured" / "repeat-00" / "training-metrics.json"
    else:
        metrics_path = root / "warmups" / "warmup-00" / "training-metrics.json"

    metrics = json.loads(metrics_path.read_text())
    if mutation == "missing":
        metrics["device"].pop("cuda_name")
    else:
        metrics["device"]["cuda_name"] = "NVIDIA A100"
    metrics_path.write_text(json.dumps(metrics))

    report_entry = report["repeats"][0] if run_kind == "measured" else report["warmups"]["runs"][0]
    report_entry["training_metrics_sha256"] = file_sha256(metrics_path)
    report_path.write_text(json.dumps(report))
    evidence_path = root / "kaggle-execution-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["benchmark_report"] = report
    evidence["benchmark_report_sha256"] = file_sha256(report_path)
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(KaggleError, match="device evidence is invalid"):
        validate_principal_benchmark_artifacts(root, "g008-run")


@pytest.mark.parametrize("mutation", [
    "missing-matching-totals", "zero-total", "boolean-total", "mismatched-report-total",
    "missing-training-updates", "wrong-training-update-count", "zero-update-count",
    "missing-update-cuda-count", "non-cuda-update", "lower-per-update-count",
])
@pytest.mark.parametrize("run_kind", ["measured", "warmup"])
def test_g008_report_import_rejects_invalid_optimizer_update_evidence(tmp_path, run_kind, mutation):
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    root = tmp_path / "g008-run"
    make_g008_benchmark(root)
    report_path = root / "g008-principal-benchmark.json"
    report = json.loads(report_path.read_text())
    if run_kind == "measured":
        metrics_path = root / "measured" / "repeat-00" / "training-metrics.json"
        entry = report["repeats"][0]
    else:
        metrics_path = root / "warmups" / "warmup-00" / "training-metrics.json"
        entry = report["warmups"]["runs"][0]
    metrics = json.loads(metrics_path.read_text())
    run_device = metrics["device"]
    training_updates = metrics["training"]["updates"]

    if mutation == "missing-matching-totals":
        run_device.pop("optimizer_updates")
        run_device.pop("cuda_optimizer_updates")
        entry.pop("optimizer_updates", None)
        entry.pop("cuda_optimizer_updates", None)
    elif mutation == "zero-total":
        run_device["optimizer_updates"] = run_device["cuda_optimizer_updates"] = 0
        if run_kind == "measured":
            entry["optimizer_updates"] = entry["cuda_optimizer_updates"] = 0
    elif mutation == "boolean-total":
        run_device["optimizer_updates"] = run_device["cuda_optimizer_updates"] = True
        if run_kind == "measured":
            entry["optimizer_updates"] = entry["cuda_optimizer_updates"] = True
    elif mutation == "mismatched-report-total":
        if run_kind == "measured":
            entry["optimizer_updates"] -= 1
            entry["cuda_optimizer_updates"] -= 1
        else:
            run_device["optimizer_updates"] -= 1
            run_device["cuda_optimizer_updates"] -= 1
    elif mutation == "missing-training-updates":
        metrics["training"].pop("updates")
    elif mutation == "wrong-training-update-count":
        training_updates.pop()
    elif mutation == "zero-update-count":
        training_updates[0]["optimizer_updates"] = 0
        training_updates[0]["cuda_updates"] = 0
    elif mutation == "missing-update-cuda-count":
        training_updates[0].pop("cuda_updates")
    elif mutation == "non-cuda-update":
        training_updates[0]["device"] = "cpu"
    elif mutation == "lower-per-update-count":
        training_updates[0]["optimizer_updates"] -= 1
        training_updates[0]["cuda_updates"] -= 1
        run_device["optimizer_updates"] -= 1
        run_device["cuda_optimizer_updates"] -= 1
        if run_kind == "measured":
            entry["optimizer_updates"] -= 1
            entry["cuda_optimizer_updates"] -= 1

    metrics_path.write_text(json.dumps(metrics))
    entry["training_metrics_sha256"] = file_sha256(metrics_path)
    report_path.write_text(json.dumps(report))
    evidence_path = root / "kaggle-execution-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["benchmark_report"] = report
    evidence["benchmark_report_sha256"] = file_sha256(report_path)
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(KaggleError):
        validate_principal_benchmark_artifacts(root, "g008-run")


@pytest.mark.parametrize("field", [
    "update_only_joint_transitions_per_second", "wall_seconds", "completed_joint_arena_transitions",
])
def test_g008_report_import_rejects_repeat_summary_values_not_derived_from_metrics(tmp_path, field):
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    root = tmp_path / "g008-run"
    make_g008_benchmark(root)
    report_path = root / "g008-principal-benchmark.json"
    report = json.loads(report_path.read_text())
    report["repeats"][0][field] += 1
    report_path.write_text(json.dumps(report))
    evidence_path = root / "kaggle-execution-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["benchmark_report"] = report
    evidence["benchmark_report_sha256"] = file_sha256(report_path)
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(KaggleError, match="disagrees with hashed training metrics"):
        validate_principal_benchmark_artifacts(root, "g008-run")


@pytest.mark.parametrize("mutation", ["missing-version", "empty-version", "missing-gpus",
                                     "multiple-training-gpus", "gpu-mismatch"])
def test_g008_report_import_rejects_incomplete_or_contradictory_kaggle_environment(tmp_path, mutation):
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    root = tmp_path / "g008-run"
    make_g008_benchmark(root)
    evidence_path = root / "kaggle-execution-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    if mutation == "missing-version":
        evidence.pop("torch_version")
    elif mutation == "empty-version":
        evidence["cuda_version"] = "  "
    elif mutation == "missing-gpus":
        evidence["visible_cuda_devices"] = []
    elif mutation == "multiple-training-gpus":
        evidence["training_cuda_device_count"] = 2
    else:
        evidence["visible_cuda_devices"] = ["NVIDIA A100"]
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(KaggleError, match="execution evidence|visible training GPU"):
        validate_principal_benchmark_artifacts(root, "g008-run")


def test_g008_report_import_rejects_consistent_non_t4_device(tmp_path):
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    root = tmp_path / "g008-run"
    make_g008_benchmark(root)
    report_path = root / "g008-principal-benchmark.json"
    report = json.loads(report_path.read_text())
    report["device"]["cuda_name"] = "NVIDIA A100"
    report_path.write_text(json.dumps(report))
    evidence_path = root / "kaggle-execution-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["visible_cuda_devices"] = ["NVIDIA A100"]
    evidence["benchmark_report"] = report
    evidence["benchmark_report_sha256"] = file_sha256(report_path)
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(KaggleError, match="approved Tesla T4"):
        validate_principal_benchmark_artifacts(root, "g008-run")


@pytest.mark.parametrize("field", ["optimizer_updates", "cuda_optimizer_updates"])
def test_g008_report_import_rejects_top_level_device_counts_outside_frozen_warmup(tmp_path, field):
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    root = tmp_path / "g008-run"
    make_g008_benchmark(root)
    report_path = root / "g008-principal-benchmark.json"
    report = json.loads(report_path.read_text())
    report["device"][field] += 1
    report_path.write_text(json.dumps(report))
    evidence_path = root / "kaggle-execution-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["benchmark_report"] = report
    evidence["benchmark_report_sha256"] = file_sha256(report_path)
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(KaggleError, match="top-level device optimizer counts"):
        validate_principal_benchmark_artifacts(root, "g008-run")


@pytest.mark.parametrize("field", ["median", "iqr"])
def test_g008_report_import_rejects_tampered_summary_percentiles(tmp_path, field):
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    root = tmp_path / "g008-run"
    make_g008_benchmark(root)
    report_path = root / "g008-principal-benchmark.json"
    report = json.loads(report_path.read_text())
    report["summary"]["wall_seconds"][field] += 1
    report_path.write_text(json.dumps(report))
    evidence_path = root / "kaggle-execution-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["benchmark_report"] = report
    evidence["benchmark_report_sha256"] = file_sha256(report_path)
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(KaggleError, match="summary wall_seconds percentiles"):
        validate_principal_benchmark_artifacts(root, "g008-run")


class FakeRecoveryRegistry:
    def __init__(self, path):
        self.path = path
        self._events = []
        self.lock_head_hash = None
        self.events_calls = 0

    def events(self):
        self.events_calls += 1
        return self._events.copy()

    def append(self, event_type, event_data):
        index = len(self._events)
        event = {"event_index": index, "event_type": event_type, "event_data": event_data,
                 "event_hash": format(index + 1, "064x")}
        self._events.append(event)
        return event

    def preflight_workload(self, data):
        return self.append("workload_preflight", data)

    def dispatch_workload_push(self, data):
        return self.append("workload_push_dispatched", data)

    def accept_workload_push(self, data):
        return self.append("workload_push_accepted", data)

    def resolve_workload_attempt(self, data):
        return self.append("workload_attempt", data)


def make_recovery_registry(tmp_path, *, timeout=30, kind="benchmark", campaign_before=8424):
    from tools.kaggle.cli import package_tree_sha256

    registry = FakeRecoveryRegistry(tmp_path / "recovery.jsonl")
    frozen_package_sha256 = package_tree_sha256(make_recovery_source(tmp_path))
    hashes = {name: f"{index:064x}" for index, name in enumerate((
        "package_sha256", "source_sha256", "config_sha256", "seed_plan_sha256",
        "backend_sha256", "trainer_sha256", "engine_sha256",
    ), 10)}
    hashes["package_sha256"] = frozen_package_sha256
    variant = {"variant_id": "paired", **hashes}
    if kind == "train":
        variants = [{**variant, "variant_id": value} for value in ("incumbent", "candidate")]
        variant_id = "incumbent"
    else:
        variants = [variant, {**variant, "variant_id": "candidate"}]
        variant_id = "paired"
    amendment = {
        "recovery_registrations": [{"registration_id": "recovery-bench", "registration_kind": "benchmark"},
                                   {"registration_id": "recovery-train", "registration_kind": "train"}],
    }
    record = {"session_timeout_seconds": timeout, "workload_gpu_seconds_cap": 3600 if kind == "benchmark" else 7200,
              "workload_output_bytes_cap": 1_000_000, "accelerator_count": 1,
              "machine_shape": "NvidiaTeslaT4", "kernel_id": "owner/kernel", **hashes}
    record["implementation_variants" if kind == "benchmark" else "allowed_implementation_variants"] = variants
    rid = "recovery-bench" if kind == "benchmark" else "recovery-train"
    other_id = "recovery-train" if kind == "benchmark" else "recovery-bench"
    workloads = [{"registration_id": rid, "registration_kind": kind, "record": record},
                 {"registration_id": other_id, "registration_kind": "train" if kind == "benchmark" else "benchmark",
                  "record": {**record, "session_timeout_seconds": 7200 if kind == "benchmark" else 3600,
                             "workload_gpu_seconds_cap": 7200 if kind == "benchmark" else 3600}}]
    amendment_event = registry.append("recovery_amendment", {
        "historical_overrun_evidence": {"current_version_number": 10},
        "reservation_metadata": {"campaign_gpu_seconds_before": campaign_before, "campaign_output_bytes_before": 0,
                                 "reserved_gpu_seconds": {"validation": 0, "final": 7200},
                                 "reserved_output_bytes": {"benchmark": 1000000, "train": 1000000}},
    })
    reg_events = []
    for workload in workloads:
        item = dict(workload)
        item["record"] = {**item["record"], "recovery_amendment_event_index": amendment_event["event_index"],
                          "recovery_amendment_event_hash": amendment_event["event_hash"]}
        event = registry.append("registration", {"registration_id": item["registration_id"],
                                                  "registration_kind": item["registration_kind"],
                                                  "record": item["record"]})
        reg_events.append(event)
    selected = next(event for event in reg_events if event["event_data"]["registration_kind"] == kind)
    return registry, selected, variant_id


def _recovery_registry_with_completed_benchmark(tmp_path, *, campaign_before=8424):
    registry, training, _ = make_recovery_registry(
        tmp_path, timeout=7200, kind="train", campaign_before=campaign_before,
    )
    benchmark = next(event for event in registry.events()
                     if event.get("event_type") == "registration"
                     and event["event_data"]["registration_kind"] == "benchmark")
    registry.append("workload_attempt", {
        "registration_event_index": benchmark["event_index"],
        "actual_gpu_seconds": 3600,
        "output_bytes": 0,
    })
    return registry, training


def test_recovery_budget_keeps_final_reserve_outside_experiment_allocation(tmp_path):
    from tools.kaggle.cli import _validate_recovery_budget

    registry, training = _recovery_registry_with_completed_benchmark(tmp_path)

    budget = _validate_recovery_budget(registry, training, 7200, quota_remaining=50000)

    assert budget["campaign_gpu_seconds_projected"] == 8424 + 3600 + 7200
    assert budget["campaign_gpu_seconds_with_final_reserve"] == 8424 + 3600 + 7200 + 7200
    assert budget["campaign_gpu_seconds_projected"] <= 21600
    assert budget["campaign_gpu_seconds_with_final_reserve"] <= 28800


def test_recovery_budget_rejects_experiment_allocation_overrun(tmp_path):
    from tools.kaggle.cli import _validate_recovery_budget

    registry, training = _recovery_registry_with_completed_benchmark(tmp_path, campaign_before=12000)

    with pytest.raises(KaggleError, match="project experiment allocation"):
        _validate_recovery_budget(registry, training, 7200, quota_remaining=50000)


def test_benchmark_budget_includes_future_training_and_final_reserves(tmp_path):
    from tools.kaggle.cli import _validate_recovery_budget

    registry, benchmark, _ = make_recovery_registry(
        tmp_path, timeout=3600, kind="benchmark", campaign_before=8424,
    )

    budget = _validate_recovery_budget(registry, benchmark, 3600, quota_remaining=50000)

    assert budget["campaign_gpu_seconds_current_attempt_projected"] == 8424 + 3600
    assert budget["future_train_reserve_gpu_seconds"] == 7200
    assert budget["campaign_gpu_seconds_projected"] == 8424 + 3600 + 7200
    assert budget["campaign_gpu_seconds_with_final_reserve"] == 8424 + 3600 + 7200 + 7200


def test_benchmark_budget_rejects_future_training_over_experiment_cap(tmp_path):
    from tools.kaggle.cli import _validate_recovery_budget

    registry, benchmark, _ = make_recovery_registry(
        tmp_path, timeout=3600, kind="benchmark", campaign_before=12000,
    )

    with pytest.raises(KaggleError, match="project experiment allocation"):
        _validate_recovery_budget(registry, benchmark, 3600, quota_remaining=50000)


class _Cycle2BudgetRegistry:
    def __init__(self, events):
        self._events = events

    def events(self):
        return self._events


def _cycle2_budget_fixture(*, benchmark_attempt=None):
    events = [{"event_index": index, "event_type": "legacy", "event_data": {}}
              for index in range(13)]
    events.append({"event_index": 13, "event_type": "recovery_amendment", "event_data": {
        "reservation_metadata": {"campaign_gpu_seconds_before": 8424,
                                  "campaign_output_bytes_before": 26403}}})
    events += [
        {"event_index": 14, "event_type": "registration", "event_data": {"registration_id": "old-bench"}},
        {"event_index": 15, "event_type": "registration", "event_data": {"registration_id": "old-train"}},
        {"event_index": 16, "event_type": "workload_preflight", "event_data": {}},
        {"event_index": 17, "event_type": "workload_push_dispatched", "event_data": {}},
        {"event_index": 18, "event_type": "workload_push_accepted", "event_data": {}},
        {"event_index": 19, "event_type": "workload_attempt", "event_data": {
            "actual_gpu_seconds": 125, "output_bytes": 4519}},
    ]
    categories = {name: 1_000_000_000 for name in ("benchmark", "training", "validation", "final")}
    events.append({"event_index": 20, "event_type": "recovery_cycle_amendment", "event_data": {
        "cycle_id": "g008-recovery-cycle-2", "project_gpu_seconds_before": 8549,
        "project_output_bytes_before": 30922, "compute_caps_gpu_seconds": {
            "benchmark": 5600, "train": 7200, "project_experiment": 21600, "project_total": 28800},
        "protected_final_reserve_gpu_seconds": 7200,
        "output_caps_bytes": {"aggregate": 5_000_000_000, "per_run": 1_000_000_000},
        "output_category_reservations_bytes": categories,
        "quota_evidence": {"remaining_gpu_seconds": 99468}}})
    records = {}
    for index, kind, timeout, output_cap in (
            (21, "benchmark", 2800, 100_000_000), (22, "train", 7200, 1_000_000_000)):
        record = {"cycle_id": "g008-recovery-cycle-2", "session_timeout_seconds": timeout,
                  "workload_gpu_seconds_cap": 5600 if kind == "benchmark" else 7200,
                  "workload_output_bytes_cap": output_cap}
        event = {"event_index": index, "event_type": "registration", "event_data": {
            "registration_id": f"cycle2-{kind}", "registration_kind": kind,
            "cycle_id": "g008-recovery-cycle-2", "record": record}}
        events.append(event)
        records[kind] = event
    if benchmark_attempt is not None:
        attempt = {"attempt_id": "cycle2-bench-01", "cycle_id": "g008-recovery-cycle-2",
                   "registration_event_index": 21, "actual_gpu_seconds": 125, "output_bytes": 4519}
        attempt.update(benchmark_attempt)
        events.append({"event_index": 23, "event_type": "workload_attempt", "event_data": attempt})
    return _Cycle2BudgetRegistry(events), records


def test_cycle2_budget_counts_terminal_event19_once_and_reserves_cycle2_categories():
    from tools.kaggle.cli import _validate_recovery_budget

    registry, records = _cycle2_budget_fixture()
    budget = _validate_recovery_budget(registry, records["benchmark"], 2800, quota_remaining=99468)

    assert budget["campaign_gpu_seconds_current_attempt_projected"] == 8549 + 2800
    assert budget["campaign_gpu_seconds_projected"] == 8549 + 2800 + 7200
    assert budget["campaign_gpu_seconds_with_final_reserve"] == 8549 + 2800 + 7200 + 7200
    assert budget["project_output_bytes_projected_with_reserves"] == 30922 + 4_000_000_000


def test_cycle2_retry_counts_only_cycle2_usage_after_event19_boundary():
    from tools.kaggle.cli import _validate_recovery_budget

    registry, records = _cycle2_budget_fixture(benchmark_attempt={})
    budget = _validate_recovery_budget(registry, records["benchmark"], 2800, quota_remaining=99468)

    assert budget["actual"] == {"benchmark": 125, "train": 0}
    assert budget["campaign_gpu_seconds_current_attempt_projected"] == 8549 + 125 + 2800
    assert budget["output_bytes"] == {"benchmark": 4519, "training": 0}


def _cycle3_budget_fixture(*, include_terminal_attempt=False):
    events = [{"event_index": index, "event_type": "legacy", "event_data": {}}
              for index in range(26)]
    events.append({"event_index": 26, "event_type": "workload_attempt", "event_data": {
        "cycle_id": "g008-recovery-cycle-2", "actual_gpu_seconds": 115, "output_bytes": 3763}})
    amendment_data = {
        "schema_id": "g008-kaggle-recovery-cycle-amendment", "schema_version": 2,
        "cycle_id": "g008-recovery-cycle-3", "prior_final_event_index": 26,
        "prior_head_hash": "a" * 64, "kernel_id": "owner/kernel",
        "current_version_number": 12, "current_terminal_status": "ERROR",
        "project_gpu_seconds_before": 8664, "project_output_bytes_before": 34685,
        "compute_caps_gpu_seconds": {"benchmark": 5600, "train": 7200,
                                     "project_experiment": 21600, "project_total": 28800},
        "protected_final_reserve_gpu_seconds": 7200,
        "output_caps_bytes": {"aggregate": 5_000_000_000, "per_run": 1_000_000_000},
        "output_category_reservations_bytes": {name: 1_000_000_000 for name in
                                                ("benchmark", "training", "validation", "final")},
        "quota_evidence": {"remaining_gpu_seconds": 99468},
    }
    amendment = {"event_index": 27, "event_type": "recovery_cycle_amendment_v2",
                 "event_data": amendment_data, "event_hash": "b" * 64}
    events.append(amendment)
    records = {}
    for index, kind, timeout, output_cap in (
            (28, "benchmark", 2800, 100_000_000), (29, "train", 7200, 1_000_000_000)):
        record = {"cycle_id": "g008-recovery-cycle-3", "session_timeout_seconds": timeout,
                  "workload_gpu_seconds_cap": 5600 if kind == "benchmark" else 7200,
                  "workload_output_bytes_cap": output_cap,
                  "recovery_cycle_amendment_event_index": 27,
                  "recovery_cycle_amendment_event_hash": "b" * 64}
        event = {"event_index": index, "event_type": "registration", "event_data": {
            "registration_id": f"cycle3-{kind}", "registration_kind": kind,
            "cycle_id": "g008-recovery-cycle-3", "record": record}}
        events.append(event)
        records[kind] = event
    if include_terminal_attempt:
        events.append({"event_index": 30, "event_type": "workload_attempt", "event_data": {
            "attempt_id": "cycle3-bench-01", "cycle_id": "g008-recovery-cycle-3",
            "registration_event_index": 28, "actual_gpu_seconds": 125, "output_bytes": 4519}})
    return _Cycle2BudgetRegistry(events), records


def test_cycle3_budget_uses_its_amendment_baseline_and_counts_cycle3_actuals_once():
    from tools.kaggle.cli import _validate_recovery_budget

    registry, records = _cycle3_budget_fixture(include_terminal_attempt=True)
    budget = _validate_recovery_budget(registry, records["benchmark"], 2800, quota_remaining=99468)

    assert budget["cycle_id"] == "g008-recovery-cycle-3"
    assert budget["actual"] == {"benchmark": 125, "train": 0}
    assert budget["output_bytes"] == {"benchmark": 4519, "training": 0}
    assert budget["campaign_gpu_seconds_current_attempt_projected"] == 8664 + 125 + 2800
    assert budget["campaign_gpu_seconds_projected"] == 8664 + 125 + 2800 + 7200
    assert budget["campaign_gpu_seconds_with_final_reserve"] == 8664 + 125 + 2800 + 7200 + 7200
    assert budget["project_output_bytes_projected_with_reserves"] == 34685 + 4_000_000_000


def test_cycle3_registration_must_pin_the_exact_v2_amendment_event():
    from tools.kaggle.cli import _cycle3_amendment, _expected_recovery_current_version

    registry, records = _cycle3_budget_fixture()
    assert _cycle3_amendment(registry.events(), records["benchmark"])["current_version_number"] == 12
    assert _expected_recovery_current_version(registry.events(), records["benchmark"]) == 12
    events = registry.events()
    events.append({"event_index": 30, "event_type": "workload_attempt", "event_data": {
        "cycle_id": "g008-recovery-cycle-3", "outcome": "terminal", "assigned_version": 13}})
    assert _expected_recovery_current_version(events, records["benchmark"]) == 13
    record = records["benchmark"]["event_data"]["record"]
    record["recovery_cycle_amendment_event_hash"] = "c" * 64
    with pytest.raises(KaggleError, match="amendment identity"):
        _cycle3_amendment(registry.events(), records["benchmark"])


def test_cycle3_expected_cursor_and_status_advance_to_exploratory_observation():
    from tools.kaggle.cli import (
        _expected_recovery_current_version,
        _expected_recovery_terminal_status,
    )

    registry, records = _cycle3_budget_fixture(include_terminal_attempt=True)
    events = registry.events()
    registration = records["benchmark"]
    events.extend([
        {"event_index": len(events), "event_type": "cycle3_retry_policy_amendment_v1",
         "event_data": {"registration_event_index": registration["event_index"]}},
        {"event_index": len(events) + 1, "event_type": "exploratory_kernel_observation_v1",
         "event_data": {"kernel_id": "owner/kernel", "version_number": 15,
                        "terminal_status": "COMPLETE", "continuation_event_index": len(events)}},
    ])

    assert _expected_recovery_current_version(events, registration) == 15
    assert _expected_recovery_terminal_status(events, registration) == "COMPLETE"


def test_cycle3_budget_counts_exploratory_observation_gpu_and_output_once():
    from tools.kaggle.cli import _validate_recovery_budget

    registry, records = _cycle3_budget_fixture(include_terminal_attempt=True)
    registration = records["benchmark"]
    events = registry.events()
    policy_index = len(events)
    events.extend([
        {"event_index": policy_index, "event_type": "cycle3_retry_policy_amendment_v1",
         "event_data": {"registration_event_index": registration["event_index"]}},
        {"event_index": policy_index + 1, "event_type": "exploratory_kernel_observation_v1",
         "event_data": {"continuation_event_index": policy_index,
                        "actual_gpu_seconds": 10, "output_bytes": 5}},
    ])

    class Registry:
        def events(self):
            return events

    budget = _validate_recovery_budget(Registry(), registration, 2800, quota_remaining=99468)
    assert budget["actual"] == {"benchmark": 135, "train": 0}
    assert budget["output_bytes"] == {"benchmark": 4524, "training": 0}


def _cycle3_recovery_setup(tmp_path):
    registry, original, variant = make_recovery_registry(tmp_path)
    amendment_data = {
        "schema_id": "g008-kaggle-recovery-cycle-amendment", "schema_version": 2,
        "cycle_id": "g008-recovery-cycle-3", "kernel_id": "owner/kernel",
        "current_version_number": 12,
        "current_terminal_status": "ERROR", "project_gpu_seconds_before": 8664,
        "current_status_evidence": {"status": "ERROR", "current_version_number": 12,
                                    "kernel_id": "owner/kernel"},
        "quota_gpu_seconds_remaining_before": 99468,
        "quota_evidence": {"remaining_gpu_seconds": 99468, "used_gpu_seconds": 8532,
                           "source": "pinned Kaggle SDK quota_view"},
        "project_output_bytes_before": 34685,
        "compute_caps_gpu_seconds": {"benchmark": 5600, "train": 7200,
                                     "project_experiment": 21600, "project_total": 28800},
        "protected_final_reserve_gpu_seconds": 7200,
        "output_caps_bytes": {"aggregate": 5_000_000_000, "per_run": 1_000_000_000},
        "output_category_reservations_bytes": {name: 1_000_000_000 for name in
                                                ("benchmark", "training", "validation", "final")},
    }
    from training.eval.registry import cycle3_preflight_context_sha256
    amendment_data["preflight_context_sha256"] = cycle3_preflight_context_sha256(amendment_data)
    amendment = registry.append("recovery_cycle_amendment_v2", amendment_data)
    record = {**original["event_data"]["record"], "cycle_id": "g008-recovery-cycle-3",
              "session_timeout_seconds": 30, "workload_gpu_seconds_cap": 5600,
              "workload_output_bytes_cap": 1_000_000,
              "recovery_cycle_amendment_event_index": amendment["event_index"],
              "recovery_cycle_amendment_event_hash": amendment["event_hash"]}
    registration = registry.append("registration", {**original["event_data"],
        "cycle_id": "g008-recovery-cycle-3", "record": record})
    return registry, registration, variant


def test_cycle3_recovery_push_starts_at_exact_v12_and_accepts_v13(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = _cycle3_recovery_setup(tmp_path)
    api = FakeRecoveryApi(statuses=("ERROR", "COMPLETE", "COMPLETE"), version=12, assigned=13,
                          quota_remaining=99468, quota_used=8532)

    terminal = run_recovery_workload(
        registry, registration["event_index"], make_recovery_source(tmp_path),
        attempt_id="cycle3-bench-01", implementation_variant_id=variant,
        no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 123,
        monotonic=lambda: 10.0, sleep=lambda _: None, poll_interval_seconds=1,
    )

    assert api.push_calls[0]["timeout"] == "30"
    assert api.current_version == 13
    preflight = next(event for event in registry.events() if event["event_type"] == "workload_preflight")
    assert preflight["event_data"]["expected_current_version"] == 12
    assert terminal["event_data"]["assigned_version"] == 13


@pytest.mark.parametrize("drift", ["status", "quota"])
def test_cycle3_initial_preflight_rejects_status_or_quota_context_drift(tmp_path, drift):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = _cycle3_recovery_setup(tmp_path)
    status = "COMPLETE" if drift == "status" else "ERROR"
    quota = 99467 if drift == "quota" else 99468
    api = FakeRecoveryApi(statuses=(status,), version=12, assigned=13,
                          quota_remaining=quota, quota_used=8532)

    with pytest.raises(KaggleError, match="authorized snapshot"):
        run_recovery_workload(
            registry, registration["event_index"], make_recovery_source(tmp_path),
            attempt_id="cycle3-preflight-drift", implementation_variant_id=variant,
            no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
            monotonic=lambda: 10.0, sleep=lambda _: None, poll_interval_seconds=1,
        )

    assert api.push_calls == []
    assert not any(event["event_type"] == "workload_preflight" for event in registry.events())


def test_cycle3_training_preflight_matches_latest_reconciled_benchmark_status(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, benchmark_registration, _ = _cycle3_recovery_setup(tmp_path)
    registry.append("workload_push_dispatched", {
        "cycle_id": "g008-recovery-cycle-3", "attempt_id": "cycle3-benchmark-terminal",
    })
    registry.append("workload_attempt", {
        "cycle_id": "g008-recovery-cycle-3", "attempt_id": "cycle3-benchmark-terminal",
        "outcome": "terminal", "registration_event_index": benchmark_registration["event_index"],
        "assigned_version": 13, "terminal_status": "COMPLETE",
        "actual_gpu_seconds": 20, "output_bytes": 100,
    })
    amendment = registry.events()[3]
    benchmark_record = benchmark_registration["event_data"]["record"]
    variants = benchmark_record["implementation_variants"]
    train_record = {**benchmark_record,
                    "session_timeout_seconds": 30, "workload_gpu_seconds_cap": 7200,
                    "workload_output_bytes_cap": 1_000_000,
                    "allowed_implementation_variants": [
                        {**item, "variant_id": name} for item, name in zip(variants, ("incumbent", "candidate"))
                    ]}
    train_record.pop("implementation_variants")
    train_registration = registry.append("registration", {
        "cycle_id": "g008-recovery-cycle-3", "registration_id": "cycle3-train",
        "registration_kind": "train", "record": train_record,
    })
    api = FakeRecoveryApi(statuses=("ERROR",), version=13, assigned=14)

    with pytest.raises(KaggleError, match="reconciled terminal attempt"):
        run_recovery_workload(
            registry, train_registration["event_index"], make_recovery_source(tmp_path),
            attempt_id="cycle3-train-status-drift", implementation_variant_id="incumbent",
            no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
            monotonic=lambda: 10.0, sleep=lambda _: None, poll_interval_seconds=1,
        )

    assert api.push_calls == []
    assert not any(event["event_type"] == "workload_preflight" for event in registry.events())


def test_cycle3_training_accepts_updated_sufficient_quota_after_benchmark(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, benchmark_registration, _ = _cycle3_recovery_setup(tmp_path)
    registry.append("workload_push_dispatched", {
        "cycle_id": "g008-recovery-cycle-3", "attempt_id": "cycle3-benchmark-terminal",
    })
    registry.append("workload_attempt", {
        "cycle_id": "g008-recovery-cycle-3", "attempt_id": "cycle3-benchmark-terminal",
        "outcome": "terminal", "registration_event_index": benchmark_registration["event_index"],
        "assigned_version": 13, "terminal_status": "COMPLETE",
        "actual_gpu_seconds": 20, "output_bytes": 100,
    })
    benchmark_record = benchmark_registration["event_data"]["record"]
    variants = benchmark_record["implementation_variants"]
    train_record = {**benchmark_record,
                    "session_timeout_seconds": 30, "workload_gpu_seconds_cap": 7200,
                    "workload_output_bytes_cap": 1_000_000,
                    "allowed_implementation_variants": [
                        {**item, "variant_id": name} for item, name in zip(variants, ("incumbent", "candidate"))
                    ]}
    train_record.pop("implementation_variants")
    train_registration = registry.append("registration", {
        "cycle_id": "g008-recovery-cycle-3", "registration_id": "cycle3-train",
        "registration_kind": "train", "record": train_record,
    })
    api = FakeRecoveryApi(statuses=("COMPLETE", "COMPLETE"), version=13, assigned=14,
                          quota_remaining=20_000, quota_used=80_000)

    terminal = run_recovery_workload(
        registry, train_registration["event_index"], make_recovery_source(tmp_path),
        attempt_id="cycle3-train-after-benchmark", implementation_variant_id="incumbent",
        no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
        monotonic=lambda: 10.0, sleep=lambda _: None, poll_interval_seconds=1,
    )

    assert api.push_calls[0]["timeout"] == "30"
    assert api.current_version == 14
    assert terminal["event_data"]["terminal_status"] == "COMPLETE"


def _append_cycle3_terminal_benchmark_error(registry, benchmark_registration):
    registry.append("workload_push_dispatched", {
        "cycle_id": "g008-recovery-cycle-3", "attempt_id": "cycle3-benchmark-error",
    })
    registry.append("workload_attempt", {
        "cycle_id": "g008-recovery-cycle-3", "attempt_id": "cycle3-benchmark-error",
        "outcome": "terminal", "registration_event_index": benchmark_registration["event_index"],
        "assigned_version": 13, "terminal_status": "ERROR",
        "actual_gpu_seconds": 20, "output_bytes": 100,
    })


def test_cycle3_retry_after_reconciled_error_uses_exact_version_status_and_fresh_quota(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = _cycle3_recovery_setup(tmp_path)
    _append_cycle3_terminal_benchmark_error(registry, registration)
    api = FakeRecoveryApi(statuses=("ERROR", "COMPLETE"), version=13, assigned=14,
                          quota_remaining=20_000, quota_used=80_000)

    terminal = run_recovery_workload(
        registry, registration["event_index"], make_recovery_source(tmp_path),
        attempt_id="cycle3-benchmark-retry-01", implementation_variant_id=variant,
        no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 123,
        monotonic=lambda: 10.0, sleep=lambda _: None, poll_interval_seconds=1,
    )

    preflight = next(event for event in registry.events()
                     if event["event_type"] == "workload_preflight"
                     and event["event_data"]["attempt_id"] == "cycle3-benchmark-retry-01")
    assert preflight["event_data"]["expected_current_version"] == 13
    assert preflight["event_data"]["policy_evidence"]["preflight_status"] == "ERROR"
    assert preflight["event_data"]["quota_evidence"]["remaining_gpu_seconds"] == 20_000
    assert api.push_calls[0]["timeout"] == "30"
    assert api.current_version == 14
    assert terminal["event_data"]["terminal_status"] == "COMPLETE"


def test_cycle3_retry_with_insufficient_fresh_quota_fails_before_preflight_or_dispatch(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = _cycle3_recovery_setup(tmp_path)
    _append_cycle3_terminal_benchmark_error(registry, registration)
    api = FakeRecoveryApi(statuses=("ERROR",), version=13, assigned=14,
                          quota_remaining=10_000, quota_used=90_000)
    prior_events = registry.events()

    with pytest.raises(KaggleError, match="quota cannot cover"):
        run_recovery_workload(
            registry, registration["event_index"], make_recovery_source(tmp_path),
            attempt_id="cycle3-benchmark-retry-insufficient-quota", implementation_variant_id=variant,
            no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
            monotonic=lambda: 10.0, sleep=lambda _: None, poll_interval_seconds=1,
        )

    assert registry.events() == prior_events
    assert api.push_calls == []


def _cycle3_registration_payloads():
    from training.eval.registry import (
        cycle3_authorization_statement,
        cycle3_preflight_context_sha256,
        cycle3_registration_bindings_sha256,
        cycle3_registration_record_sha256,
    )

    proposal_hash = "a" * 64
    workloads = []
    rows = []
    for kind, run_uuid in (("benchmark", "benchmark-run"), ("train", "train-run")):
        record = {"cycle_id": "g008-recovery-cycle-3", "run_uuid": run_uuid,
                  "session_timeout_seconds": 2800 if kind == "benchmark" else 7200,
                  "package_sha256": ("1" if kind == "benchmark" else "2") * 64}
        registration_id = f"cycle3-{kind}"
        row = {"registration_id": registration_id, "registration_kind": kind,
               "run_uuid": run_uuid, "record_sha256": cycle3_registration_record_sha256(record)}
        rows.append(row)
        workloads.append({"registration_id": registration_id, "registration_kind": kind, "record": record})
    bindings_hash = cycle3_registration_bindings_sha256(rows)
    amendment = {
        "cycle_id": "g008-recovery-cycle-3", "proposal_sha256": proposal_hash,
        "registration_bindings_sha256": bindings_hash, "recovery_registrations": rows,
        "kernel_id": "owner/kernel", "current_version_number": 12,
        "current_terminal_status": "ERROR",
        "current_status_evidence": {"status": "ERROR", "current_version_number": 12,
                                    "kernel_id": "owner/kernel"},
        "quota_gpu_seconds_remaining_before": 99468,
        "quota_evidence": {"remaining_gpu_seconds": 99468, "used_gpu_seconds": 8532,
                           "source": "pinned Kaggle SDK quota_view"},
        "recovery_registrations": rows,
    }
    context_hash = cycle3_preflight_context_sha256(amendment)
    amendment["preflight_context_sha256"] = context_hash
    amendment["user_authorization"] = {
        "proposal_sha256": proposal_hash, "registration_bindings_sha256": bindings_hash,
        "preflight_context_sha256": context_hash,
        "statement": cycle3_authorization_statement(proposal_hash, bindings_hash, context_hash),
        "confirmed_at_utc": "2026-10-20T22:00:00Z"}
    return amendment, workloads


def test_cycle3_registration_helper_binds_authorization_and_adds_derived_pointers(tmp_path):
    from tools.kaggle.cli import register_recovery_cycle3_workloads

    class Cycle3Registry(FakeRecoveryRegistry):
        def __init__(self, path):
            super().__init__(path)
            self.batch_calls = 0

        def register_recovery_cycle3_batch(self, amendment, workloads):
            self.batch_calls += 1
            amendment_event = self.append("recovery_cycle_amendment_v2", amendment)
            events = [amendment_event]
            for item in workloads:
                record = {**item["record"],
                          "recovery_cycle_amendment_event_index": amendment_event["event_index"],
                          "recovery_cycle_amendment_event_hash": amendment_event["event_hash"]}
                events.append(self.append("registration", {
                    "registration_id": item["registration_id"],
                    "registration_kind": item["registration_kind"],
                    "record": record, "cycle_id": "g008-recovery-cycle-3"}))
            return events

    registry = Cycle3Registry(tmp_path / "registry.jsonl")
    amendment, workloads = _cycle3_registration_payloads()
    api = FakeRecoveryApi(statuses=("ERROR",), version=12, quota_remaining=99468, quota_used=8532)
    events = register_recovery_cycle3_workloads(registry, amendment, workloads, api=api)

    assert registry.batch_calls == 1
    assert [event["event_type"] for event in events] == [
        "recovery_cycle_amendment_v2", "registration", "registration"]
    assert [event["event_data"]["registration_kind"] for event in events[1:]] == ["benchmark", "train"]
    for event, workload in zip(events[1:], workloads):
        record = event["event_data"]["record"]
        assert record["recovery_cycle_amendment_event_index"] == events[0]["event_index"]
        assert record["recovery_cycle_amendment_event_hash"] == events[0]["event_hash"]
        assert record == {**workload["record"],
                          "recovery_cycle_amendment_event_index": events[0]["event_index"],
                          "recovery_cycle_amendment_event_hash": events[0]["event_hash"]}


def test_cycle3_registration_helper_rejects_record_drift_before_registry_mutation(tmp_path):
    from tools.kaggle.cli import register_recovery_cycle3_workloads

    registry = FakeRecoveryRegistry(tmp_path / "registry.jsonl")
    amendment, workloads = _cycle3_registration_payloads()
    workloads[0]["record"]["package_sha256"] = "f" * 64

    with pytest.raises(KaggleError, match="record hash"):
        register_recovery_cycle3_workloads(registry, amendment, workloads)
    assert registry.events() == []


def test_cycle3_registration_helper_rejects_preflight_snapshot_drift_before_mutation(tmp_path):
    from tools.kaggle.cli import register_recovery_cycle3_workloads

    registry = FakeRecoveryRegistry(tmp_path / "registry.jsonl")
    amendment, workloads = _cycle3_registration_payloads()
    amendment["quota_gpu_seconds_remaining_before"] -= 1

    with pytest.raises(KaggleError, match="authorization"):
        register_recovery_cycle3_workloads(registry, amendment, workloads)
    assert registry.events() == []


@pytest.mark.parametrize("drift", ["kernel", "version", "status", "quota"])
def test_cycle3_registration_helper_rejects_live_context_drift_without_append(tmp_path, drift):
    from tools.kaggle.cli import register_recovery_cycle3_workloads

    registry = FakeRecoveryRegistry(tmp_path / "registry.jsonl")
    amendment, workloads = _cycle3_registration_payloads()
    api = FakeRecoveryApi(statuses=("ERROR",), version=12, quota_remaining=99468, quota_used=8532)
    if drift == "kernel":
        api.get_kernel = lambda kernel_id: {"kernel_id": "other/kernel", "current_version_number": 12}
    elif drift == "version":
        api.current_version = 13
    elif drift == "status":
        api.statuses = ["COMPLETE"]
    elif drift == "quota":
        api.quota_remaining -= 1

    with pytest.raises(KaggleError, match="Live Cycle-3 kernel|authorized Cycle-3 snapshot"):
        register_recovery_cycle3_workloads(registry, amendment, workloads, api=api)

    assert registry.events() == []
    assert api.push_calls == []


def _cycle3_retry_payloads():
    from training.eval import registry as registry_module
    from training.eval.registry import (
        cycle3_retry_authorization_statement,
        cycle3_retry_binding_sha256,
        cycle3_retry_context_sha256,
        cycle3_retry_registration_record_sha256,
    )

    proposal_hash = registry_module._CYCLE3_RETRY_PROPOSAL_SHA256
    record = {
        "cycle_id": "g008-recovery-cycle-3",
        "run_uuid": "cycle3-benchmark-retry-run",
        "kernel_id": "owner/kernel",
        "session_timeout_seconds": 2800,
        "workload_gpu_seconds_cap": 5600,
        "workload_output_bytes_cap": 100_000_000,
        "accelerator_count": 1,
        "machine_shape": "NvidiaTeslaT4",
        "implementation_variants": [{"variant_id": "paired"}],
        "package_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "seed_plan_sha256": "4" * 64,
        "backend_sha256": "5" * 64,
        "trainer_sha256": "6" * 64,
        "engine_sha256": "7" * 64,
    }
    registration_id = "cycle3-benchmark-retry"
    record_hash = cycle3_retry_registration_record_sha256(record)
    context = {
        "kernel_id": "owner/kernel",
        "current_version_number": 13,
        "current_terminal_status": "ERROR",
        "current_status_evidence": {
            "status": "ERROR", "current_version_number": 13, "kernel_id": "owner/kernel",
        },
        "quota_gpu_seconds_remaining_before": 80_000,
        "quota_evidence": {
            "remaining_gpu_seconds": 80_000, "used_gpu_seconds": 20_000,
            "source": "pinned Kaggle SDK quota_view",
        },
    }
    context_hash = cycle3_retry_context_sha256(context)
    binding_hash = cycle3_retry_binding_sha256(
        registration_id, "benchmark", record["run_uuid"], record_hash,
    )
    amendment = {
        "schema_id": "g008-kaggle-recovery-cycle-retry-amendment",
        "schema_version": 1,
        "cycle_id": "g008-recovery-cycle-3-retry",
        "created_at_utc": "2026-10-20T22:00:00Z",
        "prior_head_hash": "4f7c8d909349b3a95c91758a2fe6c7490f241ed5db9df9925ce4f600163db966",
        "prior_final_event_index": 33,
        "cycle3_amendment_event_index": 27,
        "cycle3_amendment_event_hash": "7c6bf3510e1cc3ccb20da8198e1915e28b5c363c0c65845f910819ba3eabf7c6",
        "proposal_sha256": proposal_hash,
        "retry_registration_id": registration_id,
        "registration_kind": "benchmark",
        "retry_run_uuid": record["run_uuid"],
        "retry_record_sha256": record_hash,
        "registration_binding_sha256": binding_hash,
        "preflight_context_sha256": context_hash,
        **context,
    }
    amendment["user_authorization"] = {
        "proposal_sha256": proposal_hash,
        "record_sha256": record_hash,
        "preflight_context_sha256": context_hash,
        "statement": cycle3_retry_authorization_statement(proposal_hash, record_hash, context_hash),
        "confirmed_at_utc": "2026-10-20T22:01:00Z",
    }
    workload = {
        "registration_id": registration_id,
        "registration_kind": "benchmark",
        "record": record,
    }
    return amendment, workload


class FakeCycle3RetryRegistry:
    """Capture the retry batch call without touching the project registry."""

    def __init__(self):
        self.calls = []
        self.prior_events = [
            {"event_index": index, "event_hash": "a" * 64, "event_data": {}}
            for index in range(34)
        ]
        self.prior_events[27]["event_hash"] = "7c6bf3510e1cc3ccb20da8198e1915e28b5c363c0c65845f910819ba3eabf7c6"
        self.prior_events[33].update({
            "event_hash": "4f7c8d909349b3a95c91758a2fe6c7490f241ed5db9df9925ce4f600163db966",
            "event_data": {
                "outcome": "terminal", "terminal_status": "ERROR", "assigned_version": 13,
                "terminal_metadata": {"kernel_id": "owner/kernel"},
            },
        })

    def events(self):
        return list(self.prior_events)

    def register_recovery_cycle3_retry_batch(self, amendment, workload):
        self.calls.append((amendment, workload))
        amendment_event = {
            "event_index": 34,
            "event_type": "recovery_cycle_retry_amendment_v1",
            "event_hash": "e" * 64,
            "event_data": amendment,
        }
        record = {
            **workload["record"],
            "cycle3_retry_amendment_event_index": 34,
            "cycle3_retry_amendment_event_hash": amendment_event["event_hash"],
            "recovery_cycle_amendment_event_index": 27,
            "recovery_cycle_amendment_event_hash": amendment["cycle3_amendment_event_hash"],
        }
        registration_event = {
            "event_index": 35,
            "event_type": "registration",
            "event_hash": "f" * 64,
            "event_data": {
                "registration_id": workload["registration_id"],
                "registration_kind": workload["registration_kind"],
                "record": record,
                "cycle_id": "g008-recovery-cycle-3",
            },
        }
        return [amendment_event, registration_event]


def test_cycle3_retry_registration_helper_checks_live_snapshot_and_atomically_appends_34_35():
    from tools.kaggle.cli import register_recovery_cycle3_retry

    registry = FakeCycle3RetryRegistry()
    amendment, workload = _cycle3_retry_payloads()
    api = FakeRecoveryApi(statuses=("ERROR",), version=13,
                          quota_remaining=80_000, quota_used=20_000)

    events = register_recovery_cycle3_retry(registry, amendment, workload, api=api)

    assert [event["event_index"] for event in events] == [34, 35]
    assert [event["event_type"] for event in events] == [
        "recovery_cycle_retry_amendment_v1", "registration",
    ]
    assert len(registry.calls) == 1
    # The live context is sampled around the status call, then quota is captured.
    assert api.metadata_calls == ["owner/kernel", "owner/kernel"]
    assert api.status_calls == ["owner/kernel"]
    assert api.quota_calls == 1
    assert api.push_calls == []
    appended_amendment, appended_workload = registry.calls[0]
    assert appended_amendment == amendment
    assert appended_workload == workload
    registered_record = events[1]["event_data"]["record"]
    assert registered_record == {
        **workload["record"],
        "cycle3_retry_amendment_event_index": 34,
        "cycle3_retry_amendment_event_hash": events[0]["event_hash"],
        "recovery_cycle_amendment_event_index": 27,
        "recovery_cycle_amendment_event_hash": amendment["cycle3_amendment_event_hash"],
    }


@pytest.mark.parametrize("drift", ["kernel", "version", "status", "quota", "running"])
def test_cycle3_retry_registration_helper_rejects_live_context_drift_without_append(drift):
    from tools.kaggle.cli import register_recovery_cycle3_retry

    registry = FakeCycle3RetryRegistry()
    amendment, workload = _cycle3_retry_payloads()
    api = FakeRecoveryApi(statuses=("ERROR",), version=13,
                          quota_remaining=80_000, quota_used=20_000)
    if drift == "kernel":
        api.get_kernel = lambda kernel_id: {
            "kernel_id": "other/kernel", "current_version_number": 14,
        }
    elif drift == "version":
        api.drift_after_status = True
    elif drift == "status":
        api.statuses = ["COMPLETE"]
    elif drift == "quota":
        api.quota_remaining -= 1
    elif drift == "running":
        api.statuses = ["RUNNING"]

    with pytest.raises(KaggleError):
        register_recovery_cycle3_retry(registry, amendment, workload, api=api)

    assert registry.calls == []
    assert api.push_calls == []


@pytest.mark.parametrize("drift", ["authorization", "record"])
def test_cycle3_retry_registration_helper_rejects_frozen_authorization_drift_before_append(drift):
    from tools.kaggle.cli import register_recovery_cycle3_retry

    registry = FakeCycle3RetryRegistry()
    amendment, workload = _cycle3_retry_payloads()
    if drift == "authorization":
        amendment["user_authorization"]["statement"] += " changed"
    else:
        workload["record"]["package_sha256"] = "f" * 64
    api = FakeRecoveryApi(statuses=("ERROR",), version=13,
                          quota_remaining=80_000, quota_used=20_000)

    with pytest.raises(KaggleError):
        register_recovery_cycle3_retry(registry, amendment, workload, api=api)

    assert registry.calls == []
    assert api.metadata_calls == []
    assert api.status_calls == []
    assert api.quota_calls == 0
    assert api.push_calls == []


def test_cycle3_retry_preflight_rejects_quota_drift_before_preflight_or_dispatch(tmp_path, monkeypatch):
    from tools.kaggle import cli

    base, records = _cycle3_budget_fixture()
    events = base.events()
    events[27]["event_hash"] = "7c6bf3510e1cc3ccb20da8198e1915e28b5c363c0c65845f910819ba3eabf7c6"
    for event in events[28:30]:
        event["event_data"]["record"]["recovery_cycle_amendment_event_hash"] = events[27]["event_hash"]
    events.extend({"event_index": index, "event_type": "legacy", "event_data": {}}
                  for index in range(30, 32))
    events.append({"event_index": 32, "event_type": "workload_push_dispatched", "event_hash": "d" * 64,
                   "event_data": {"cycle_id": "g008-recovery-cycle-3", "attempt_id": "cycle3-benchmark-error"}})
    events.append({"event_index": 33, "event_type": "workload_attempt", "event_hash": "4f7c8d909349b3a95c91758a2fe6c7490f241ed5db9df9925ce4f600163db966",
                   "event_data": {"cycle_id": "g008-recovery-cycle-3", "attempt_id": "cycle3-benchmark-error",
                                  "outcome": "terminal", "registration_event_index": 28,
                                  "assigned_version": 13, "terminal_status": "ERROR",
                                  "terminal_metadata": {"kernel_id": "owner/kernel"},
                                  "actual_gpu_seconds": 104, "output_bytes": 4497}})

    class AppendingRetryRegistry:
        path = tmp_path / "registry.jsonl"

        def __init__(self):
            self._events = events

        def events(self):
            return list(self._events)

        def register_recovery_cycle3_retry_batch(self, amendment, workload):
            amendment_event = {"event_index": 34, "event_type": "recovery_cycle_retry_amendment_v1",
                               "event_hash": "e" * 64, "event_data": amendment}
            record = {**workload["record"], "cycle3_retry_amendment_event_index": 34,
                      "cycle3_retry_amendment_event_hash": amendment_event["event_hash"],
                      "recovery_cycle_amendment_event_index": 27,
                      "recovery_cycle_amendment_event_hash": amendment["cycle3_amendment_event_hash"]}
            registration = {"event_index": 35, "event_type": "registration", "event_hash": "f" * 64,
                            "event_data": {"registration_id": workload["registration_id"],
                                           "registration_kind": workload["registration_kind"],
                                           "record": record, "cycle_id": "g008-recovery-cycle-3"}}
            self._events.extend((amendment_event, registration))
            return [amendment_event, registration]

    registry = AppendingRetryRegistry()
    amendment, workload = _cycle3_retry_payloads()
    api = FakeRecoveryApi(statuses=("ERROR", "ERROR"), version=13,
                          quota_remaining=80_000, quota_used=20_000)
    appended = cli.register_recovery_cycle3_retry(registry, amendment, workload, api=api)
    assert [event["event_index"] for event in appended] == [34, 35]

    source = make_recovery_source(tmp_path)
    monkeypatch.setattr(cli, "package_tree_sha256", lambda _: workload["record"]["package_sha256"])
    before_run = registry.events()
    api.quota_remaining = 79_999
    api.quota_used = 20_001
    with pytest.raises(KaggleError, match="authorized Cycle-3 retry context"):
        cli.run_recovery_workload(
            registry, 35, source, attempt_id="cycle3-benchmark-retry-context-drift",
            implementation_variant_id="paired", no_external_writer_confirmed=True,
            api=api, output_bytes=lambda kernel, version: 0,
            monotonic=lambda: 10.0, sleep=lambda _: None, poll_interval_seconds=1,
        )

    assert registry.events() == before_run
    assert not any(event["event_type"] == "workload_preflight" for event in registry.events())
    assert api.push_calls == []


class FakeRecoveryApi:
    def __init__(self, statuses=("COMPLETE", "COMPLETE"), *, version=10, assigned=11,
                 quota_remaining=40000, quota_used=1000, drift_after_status=False,
                 lose_push_response=False):
        self.statuses = list(statuses)
        self.current_version = version
        self.assigned_version = assigned
        self.quota_remaining = quota_remaining
        self.quota_used = quota_used
        self.drift_after_status = drift_after_status
        self.lose_push_response = lose_push_response
        self.reject_before_assignment = False
        self.metadata_calls = []
        self.status_calls = []
        self.push_calls = []
        self.quota_calls = 0

    def get_kernel(self, kernel_id):
        self.metadata_calls.append(kernel_id)
        return {"kernel_id": kernel_id, "current_version_number": self.current_version}

    def kernels_status(self, kernel_id):
        self.status_calls.append(kernel_id)
        if self.drift_after_status:
            self.current_version += 1
        status = self.statuses.pop(0) if self.statuses else "RUNNING"
        return {"status": status}

    def kernels_push(self, folder, *, timeout, acc):
        self.push_calls.append({"folder": folder, "timeout": timeout, "acc": acc})
        if self.lose_push_response:
            raise TimeoutError("fake lost response")
        if self.reject_before_assignment:
            return {"version_number": 0, "error": "rejected by server"}
        self.current_version = self.assigned_version
        return {"version_number": self.assigned_version}

    def quota_view(self):
        self.quota_calls += 1
        if self.quota_calls > 1:
            self.quota_used += 20
            self.quota_remaining -= 20
        return {"gpu_quota": {"remaining_gpu_seconds": self.quota_remaining,
                              "used_gpu_seconds": self.quota_used}}


def test_sdk_metadata_uses_generated_client_context_without_version_selector(monkeypatch):
    requests = []

    class GeneratedGetKernelRequest:
        def __init__(self):
            self.user_name = None
            self.kernel_slug = None
            self.version_label = None

    class GeneratedGetKernelResponse:
        def __init__(self):
            self.metadata = types.SimpleNamespace(ref="owner/slug", current_version_number=17)

    class KernelsApiClient:
        def get_kernel(self, request):
            requests.append(request)
            return GeneratedGetKernelResponse()

    generated_module = types.ModuleType("kagglesdk.kernels.types.kernels_api_service")
    generated_module.ApiGetKernelRequest = GeneratedGetKernelRequest
    monkeypatch.setitem(sys.modules, generated_module.__name__, generated_module)

    client = types.SimpleNamespace(
        kernels=types.SimpleNamespace(kernels_api_client=KernelsApiClient()),
    )
    client_context = {"entered": 0, "exited": 0}

    class KaggleApiWithoutConvenienceMetadata:
        def build_kaggle_client(self):
            class ContextManager:
                def __enter__(self):
                    client_context["entered"] += 1
                    return client

                def __exit__(self, exc_type, exc_value, traceback):
                    client_context["exited"] += 1
                    return False

            return ContextManager()

    api = KaggleApiWithoutConvenienceMetadata()
    assert not hasattr(api, "get_kernel")
    assert not hasattr(api, "kernels")

    assert _sdk_metadata(api, "owner/slug") == {
        "kernel_id": "owner/slug", "current_version_number": 17,
    }
    assert client_context == {"entered": 1, "exited": 1}
    assert len(requests) == 1
    request = requests[0]
    assert type(request) is GeneratedGetKernelRequest
    assert request.user_name == "owner"
    assert request.kernel_slug == "slug"
    assert request.version_label is None


def make_recovery_source(tmp_path):
    source = tmp_path / "recovery-package"
    source.mkdir(parents=True, exist_ok=True)
    (source / "kernel-metadata.json").write_text(json.dumps({"id": "owner/kernel"}))
    (source / ".arena-source-revision").write_text("frozen-r1")
    (source / "run_kaggle.py").write_text("# frozen test source\n")
    return source


@pytest.mark.parametrize("kind", ["benchmark", "train"])
def test_recovery_adapter_pushes_exact_timeout_accepts_assignment_and_brackets_latest_polls(tmp_path, kind):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = make_recovery_registry(tmp_path, kind=kind)
    if kind == "train":
        registry.lock_head_hash = "f" * 64
        registry.append("registration_closed", {"pre_lock_head_hash": "e" * 64})
    api = FakeRecoveryApi(statuses=("COMPLETE", "RUNNING", "COMPLETE"))
    clock = [10.0]
    event = run_recovery_workload(
        registry, registration["event_index"], make_recovery_source(tmp_path),
        attempt_id="attempt-1", implementation_variant_id=variant,
        no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 123,
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=1, reconciliation_slack_seconds=5,
    )
    assert api.push_calls[0]["timeout"] == "30"
    assert api.push_calls[0]["acc"] == "NvidiaTeslaT4"
    assert api.status_calls == ["owner/kernel"] * 3
    assert api.metadata_calls.count("owner/kernel") == 7
    assert [item["event_type"] for item in registry.events()][-4:] == [
        "workload_preflight", "workload_push_dispatched", "workload_push_accepted", "workload_attempt",
    ]
    assert event["event_data"]["assigned_version"] == 11
    assert event["event_data"]["output_bytes"] == 123


def test_recovery_preflight_requires_explicit_no_external_writer_confirmation(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = make_recovery_registry(tmp_path)
    api = FakeRecoveryApi()
    with pytest.raises(KaggleError, match="no outside writer"):
        run_recovery_workload(registry, registration["event_index"], make_recovery_source(tmp_path),
                              attempt_id="attempt-1", implementation_variant_id=variant,
                              no_external_writer_confirmed=False, api=api,
                              output_bytes=lambda kernel, version: 0)
    assert api.push_calls == [] and api.metadata_calls == []
    assert not any(event["event_type"] == "workload_preflight" for event in registry.events())


def test_recovery_rejects_package_drift_before_any_event_or_push(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = make_recovery_registry(tmp_path)
    source = make_recovery_source(tmp_path)
    (source / "run_kaggle.py").write_text("# mutated after registration\n")
    api = FakeRecoveryApi()
    with pytest.raises(KaggleError, match="registered package_sha256"):
        run_recovery_workload(
            registry, registration["event_index"], source,
            attempt_id="attempt-mutated-package", implementation_variant_id=variant,
            no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
        )
    assert api.push_calls == []
    assert not any(event["event_type"] in {"workload_preflight", "workload_push_dispatched"}
                   for event in registry.events())


def test_recovery_package_tree_hash_rejects_symlinks(tmp_path):
    from tools.kaggle.cli import package_tree_sha256

    source = make_recovery_source(tmp_path)
    target = tmp_path / "outside-secret"
    target.write_text("must not be followed")
    (source / "linked-input").symlink_to(target)
    with pytest.raises(KaggleError, match="cannot contain symlinks"):
        package_tree_sha256(source)


def test_recovery_resumes_preflight_after_package_drift_without_duplicate_events(tmp_path, monkeypatch):
    import tools.kaggle.cli as cli

    registry, registration, variant = make_recovery_registry(tmp_path)
    source = make_recovery_source(tmp_path)
    api = FakeRecoveryApi(statuses=("COMPLETE", "COMPLETE", "COMPLETE"))
    original_hash = cli.package_tree_sha256
    calls = []

    def change_before_dispatch(path):
        calls.append(path)
        return original_hash(path) if len(calls) < 4 else "0" * 64

    monkeypatch.setattr(cli, "package_tree_sha256", change_before_dispatch)
    with pytest.raises(KaggleError, match="Read-only submission snapshot"):
        cli.run_recovery_workload(
            registry, registration["event_index"], source,
            attempt_id="attempt-package-race", implementation_variant_id=variant,
            no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
        )
    assert len(calls) == 4
    assert api.push_calls == []
    assert registry.events()[-1]["event_type"] == "workload_preflight"
    assert not any(event["event_type"] == "workload_push_dispatched" for event in registry.events())

    monkeypatch.setattr(cli, "package_tree_sha256", original_hash)
    terminal = cli.run_recovery_workload(
        registry, registration["event_index"], source,
        attempt_id="attempt-package-race", implementation_variant_id=variant,
        no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
    )
    types = [event["event_type"] for event in registry.events()]
    assert types.count("workload_preflight") == 1
    assert types.count("workload_push_dispatched") == 1
    assert types.count("workload_push_accepted") == 1
    assert types.count("workload_attempt") == 1
    assert terminal["event_data"]["terminal_status"] == "COMPLETE"
    projected_budget = terminal["event_data"]["result"]["projected_budget"]
    assert projected_budget["quota_gpu_seconds_remaining"] == 39980
    assert projected_budget["campaign_gpu_seconds_projected"] == 8424 + 30 + 7200
    assert len(api.push_calls) == 1


def test_recovery_lost_push_response_leaves_pending_tail_and_never_pushes_twice(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = make_recovery_registry(tmp_path)
    api = FakeRecoveryApi(lose_push_response=True)
    args = (registry, registration["event_index"], make_recovery_source(tmp_path))
    kwargs = {"attempt_id": "attempt-1", "implementation_variant_id": variant,
              "no_external_writer_confirmed": True, "api": api,
              "output_bytes": lambda kernel, version: 0}
    with pytest.raises(KaggleError, match="ambiguous"):
        run_recovery_workload(*args, **kwargs)
    with pytest.raises(KaggleError, match="dispatch-only tail"):
        run_recovery_workload(*args, **kwargs)
    assert len(api.push_calls) == 1
    assert [event["event_type"] for event in registry.events()][-2:] == [
        "workload_preflight", "workload_push_dispatched",
    ]


def test_recovery_adapter_rejects_version_drift_and_unknown_status_without_terminal_event(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    for index, api in enumerate((FakeRecoveryApi(drift_after_status=True),
                                 FakeRecoveryApi(statuses=("COMPLETE", "MYSTERY")))):
        registry, registration, variant = make_recovery_registry(tmp_path / str(index))
        with pytest.raises(KaggleError):
            run_recovery_workload(registry, registration["event_index"],
                                  make_recovery_source(tmp_path / str(index)),
                                  attempt_id="attempt-1", implementation_variant_id=variant,
                                  no_external_writer_confirmed=True, api=api,
                                  output_bytes=lambda kernel, version: 0,
                                  monotonic=lambda: 0, sleep=lambda seconds: None,
                                  reconciliation_slack_seconds=0)
        assert not any(event["event_type"] == "workload_attempt" for event in registry.events())


def test_recovery_adapter_budget_rejection_happens_before_preflight_or_push(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = make_recovery_registry(tmp_path)
    api = FakeRecoveryApi(quota_remaining=10000)
    with pytest.raises(KaggleError, match="quota"):
        run_recovery_workload(registry, registration["event_index"], make_recovery_source(tmp_path),
                              attempt_id="attempt-1", implementation_variant_id=variant,
                              no_external_writer_confirmed=True, api=api,
                              output_bytes=lambda kernel, version: 0)
    assert api.push_calls == []
    assert not any(event["event_type"] == "workload_preflight" for event in registry.events())


def test_recovery_adapter_resolves_terminal_error_and_releases_project_lock(tmp_path):
    from tools.kaggle.cli import _recovery_kernel_lock, run_recovery_workload

    registry, registration, variant = make_recovery_registry(tmp_path)
    api = FakeRecoveryApi(statuses=("COMPLETE", "ERROR"))
    event = run_recovery_workload(
        registry, registration["event_index"], make_recovery_source(tmp_path),
        attempt_id="attempt-error", implementation_variant_id=variant,
        no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 1,
        monotonic=lambda: 2.0, sleep=lambda seconds: None,
    )
    assert event["event_data"]["terminal_status"] == "ERROR"
    with _recovery_kernel_lock(registry.path, "owner/kernel"):
        pass


def test_recovery_timeout_keeps_accepted_reservation_pending_and_blocks_new_attempt(tmp_path):
    from tools.kaggle.cli import _recovery_kernel_lock, run_recovery_workload

    registry, registration, variant = make_recovery_registry(tmp_path)
    api = FakeRecoveryApi(statuses=("COMPLETE",))
    ticks = [0.0]
    def monotonic():
        ticks[0] += 1
        return ticks[0]
    source = make_recovery_source(tmp_path)
    with pytest.raises(KaggleError, match="monotonic timeout deadline"):
        run_recovery_workload(
            registry, registration["event_index"], source,
            attempt_id="attempt-timeout", implementation_variant_id=variant,
            no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
            monotonic=monotonic, sleep=lambda seconds: None,
            reconciliation_slack_seconds=0,
        )
    assert [event["event_type"] for event in registry.events()][-3:] == [
        "workload_preflight", "workload_push_dispatched", "workload_push_accepted",
    ]
    with _recovery_kernel_lock(registry.path, "owner/kernel"):
        pass
    with pytest.raises(KaggleError, match="unresolved attempt"):
        run_recovery_workload(
            registry, registration["event_index"], source,
            attempt_id="attempt-retry", implementation_variant_id=variant,
            no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
        )
    assert len(api.push_calls) == 1


def test_recovery_resumes_accepted_tail_to_terminal_with_original_budget(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = make_recovery_registry(tmp_path)
    source = make_recovery_source(tmp_path)
    first_api = FakeRecoveryApi(statuses=("COMPLETE", "RUNNING"))
    ticks = [0.0]

    def monotonic():
        ticks[0] += 1.0
        return ticks[0]

    with pytest.raises(KaggleError, match="monotonic timeout deadline"):
        run_recovery_workload(
            registry, registration["event_index"], source,
            attempt_id="accepted-resume", implementation_variant_id=variant,
            no_external_writer_confirmed=True, api=first_api, output_bytes=lambda kernel, version: 0,
            monotonic=monotonic, sleep=lambda _: None, reconciliation_slack_seconds=0,
        )
    assert registry.events()[-1]["event_type"] == "workload_push_accepted"

    # The run was already accepted against 40,000 seconds of quota. Its remaining
    # quota can now be below the threshold for a new reservation without making
    # terminal reconciliation invalid.
    resume_api = FakeRecoveryApi(statuses=("COMPLETE",), version=11, assigned=11,
                                quota_remaining=10000, quota_used=30000)
    terminal = run_recovery_workload(
        registry, registration["event_index"], source,
        attempt_id="accepted-resume", implementation_variant_id=variant,
        no_external_writer_confirmed=True, api=resume_api, output_bytes=lambda kernel, version: 0,
    )
    types = [event["event_type"] for event in registry.events()]
    assert types.count("workload_preflight") == 1
    assert types.count("workload_push_dispatched") == 1
    assert types.count("workload_push_accepted") == 1
    assert types.count("workload_attempt") == 1
    assert terminal["event_data"]["terminal_status"] == "COMPLETE"
    assert terminal["event_data"]["result"]["quota_gpu_seconds_before"] == 30000
    assert terminal["event_data"]["result"]["quota_gpu_seconds_remaining_after"] == 9980
    projected_budget = terminal["event_data"]["result"]["projected_budget"]
    assert projected_budget["quota_gpu_seconds_remaining"] == 40000
    assert projected_budget["campaign_gpu_seconds_projected"] == 8424 + 30 + 7200


def test_recovery_explicit_server_rejection_resolves_without_version_assignment(tmp_path):
    from tools.kaggle.cli import run_recovery_workload

    registry, registration, variant = make_recovery_registry(tmp_path)
    api = FakeRecoveryApi()
    api.reject_before_assignment = True
    event = run_recovery_workload(
        registry, registration["event_index"], make_recovery_source(tmp_path),
        attempt_id="attempt-rejected", implementation_variant_id=variant,
        no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
    )
    assert event["event_data"]["outcome"] == "pre_assignment_rejected"
    assert event["event_data"]["assigned_version"] is None
    assert event["event_data"]["actual_gpu_seconds"] == 0


def test_recovery_cli_requires_operator_gates_before_registry_or_kaggle_access(tmp_path, capsys):
    from tools.kaggle.cli import main

    args = ["kaggle", "recovery-run", "--registry", str(tmp_path / "registry.jsonl"),
            "--registration-event-index", "0", "--attempt-id", "cli-attempt",
            "--source-dir", str(tmp_path / "source"), "--output-dir", str(tmp_path / "output")]
    assert main(args) == 2
    assert "--confirm" in capsys.readouterr().err
    assert not (tmp_path / "registry.jsonl").exists()


def _campaign_cli_fixture(kind="train", role="baseline"):
    from tools.kaggle.cli import _campaign_amendment

    campaign_id = "g008-competitive-20261020-v1"
    amendment = {
        "campaign_id": campaign_id, "current_version_number": 14,
        "benchmark_gpu_seconds_cap": 2800, "training_gpu_seconds_cap": 10800,
        "evaluation_gpu_seconds_cap": 1800, "total_gpu_seconds_cap": 15400,
        "protected_final_reserve_gpu_seconds": 7200, "records": [{} for _ in range(9)],
    }
    events = [{"event_index": 0, "event_type": "g008_competitive_campaign_amendment_v1",
               "event_hash": "a" * 64, "event_data": amendment}]
    target = None
    for index in range(9):
        row_kind = "benchmark" if index == 0 else "train" if index < 7 else "validation" if index == 7 else "final"
        row_role = role if row_kind == "train" else "benchmark" if row_kind == "benchmark" else None
        record = {"campaign_id": campaign_id, "competitive_campaign_event_index": 0,
                  "competitive_campaign_event_hash": "a" * 64, "kernel_id": "owner/kernel",
                  "session_timeout_seconds": 100, "workload_gpu_seconds_cap": 2800 if row_kind == "benchmark" else 7200,
                  "workload_output_bytes_cap": 1_000_000, "accelerator_count": 1,
                  "machine_shape": "NvidiaTeslaT4", "strategy_role": row_role,
                  "package_sha256": "b" * 64, "source_sha256": "c" * 64,
                  "config_sha256": "d" * 64, "seed_plan_sha256": "e" * 64,
                  "backend_sha256": "f" * 64, "trainer_sha256": "1" * 64,
                  "engine_sha256": "2" * 64}
        event = {"event_index": index + 1, "event_type": "registration",
                 "event_hash": str(index + 1) * 64,
                 "event_data": {"registration_id": f"campaign-{index}", "registration_kind": row_kind,
                                "cycle_id": "g008-competitive-campaign-v1", "record": record}}
        events.append(event)
        if row_kind == kind and target is None:
            target = event
    assert target is not None
    assert _campaign_amendment(events, target) == amendment
    return events, target


def _campaign_v2_cli_fixture(kind="train", role="baseline"):
    from tools.kaggle.cli import _campaign_amendment

    campaign_id = "g008-competitive-20261021-v2"
    events = [{"event_index": index, "event_type": "history_placeholder",
               "event_hash": f"{index + 1:064x}", "event_data": {}}
              for index in range(55)]
    run_uuid = "12345678-1234-4234-8234-123456789abc"
    selected_record = {"run_uuid": run_uuid,
                       "allowed_implementation_variants": [{"variant_id": "incumbent"}]}
    events[29] = {"event_index": 29, "event_type": "registration", "event_hash": "a" * 64,
                  "event_data": {"registration_id": "selected-training", "registration_kind": "train",
                                 "cycle_id": "g008-recovery-cycle-3", "record": selected_record}}
    events[48] = {**events[48], "event_type": "benchmark_artifact"}
    events[49] = {**events[49], "event_type": "registration_closed"}
    events[50] = {**events[50], "event_type": "benchmark_attestation"}
    events[54] = {"event_index": 54, "event_type": "workload_attempt", "event_hash": "b" * 64,
                  "event_data": {"registration_event_index": 29, "registration_event_hash": "a" * 64,
                                 "implementation_variant_id": "incumbent",
                                 "terminal_status": "COMPLETE", "assigned_version": 17,
                                 "actual_gpu_seconds": 100, "output_bytes": 100}}
    amendment = {"schema_id": "g008-competitive-campaign", "schema_version": 2,
                 "campaign_id": campaign_id, "current_version_number": 17,
                 "prior_final_event_index": 54, "prior_head_hash": "b" * 64,
                 "benchmark_artifact_event_index": 48, "benchmark_artifact_event_hash": events[48]["event_hash"],
                 "benchmark_lock_event_index": 49, "benchmark_lock_event_hash": events[49]["event_hash"],
                 "benchmark_attestation_event_index": 50, "benchmark_attestation_event_hash": events[50]["event_hash"],
                 "selected_training_registration_event_index": 29,
                 "selected_training_registration_event_hash": "a" * 64,
                 "selected_training_attempt_event_index": 54,
                 "selected_training_attempt_event_hash": "b" * 64,
                 "selected_training_run_uuid": run_uuid,
                 "training_gpu_seconds_cap": 7200,
                 "total_gpu_seconds_cap": 11400, "evaluation_gpu_seconds_cap": 4200,
                 "protected_final_reserve_gpu_seconds": 7200,
                 "validation_roster": {"roster_sha256": "5" * 64, "opponent_code_sha256": "6" * 64,
                                       "game_count": 512, "seat_order": [0, 1],
                                       "paired_seeds_sha256": "7" * 64},
                 "candidate_selection_rule_id": "paired-score-gain-t95-lower-positive-no-pair-regresses-over-5pt-then-median-arm-checkpoint-v1",
                 "records": [{} for _ in range(13)],
                 "output_caps_bytes": {"aggregate": 5_000_000_000, "per_run": 100_000_000}}
    events.append({"event_index": 55, "event_type": "g008_competitive_campaign_amendment_v2",
                   "event_hash": "c" * 64, "event_data": amendment})
    target = None
    for offset in range(13):
        row_kind = "train" if offset < 6 else "validation" if offset < 12 else "final"
        row_role = role if row_kind == "train" else None
        timeout = 1000 if row_kind == "train" else 600
        record = {"campaign_id": campaign_id, "competitive_campaign_v2_event_index": 55,
                  "competitive_campaign_v2_event_hash": "c" * 64, "kernel_id": "owner/kernel",
                  "session_timeout_seconds": timeout,
                  "workload_gpu_seconds_cap": 7200, "workload_output_bytes_cap": 100_000_000,
                  "accelerator_count": 1, "machine_shape": "NvidiaTeslaT4", "strategy_role": row_role,
                  "pair_id": f"seed-pair-{offset // 2 + 1}" if row_kind == "train" else None,
                  "run_uuid": f"87654321-1234-4234-8234-{offset:012d}",
                  "package_sha256": "d" * 64, "source_sha256": "e" * 64,
                  "config_sha256": "f" * 64, "seed_plan_sha256": "1" * 64,
                  "backend_sha256": "2" * 64, "trainer_sha256": "3" * 64,
                  "engine_sha256": "4" * 64}
        event = {"event_index": 56 + offset, "event_type": "registration",
                 "event_hash": f"{56 + offset:064x}",
                 "event_data": {"registration_id": f"v2-{offset}", "registration_kind": row_kind,
                                "cycle_id": "g008-competitive-campaign-v2", "record": record}}
        events.append(event)
        if row_kind == kind and target is None:
            target = event
    assert _campaign_amendment(events, target) == amendment
    return events, target


def test_campaign_v2_resolves_exact_thirteen_registration_block_and_selected_v17_history():
    from tools.kaggle.cli import (_campaign_amendment, _expected_recovery_current_version,
                                  _expected_recovery_terminal_status, _recovery_record,
                                  _validate_campaign_v2_benchmark_lineage)

    events, registration = _campaign_v2_cli_fixture("train", "mixed")
    amendment = _campaign_amendment(events, registration)
    _validate_campaign_v2_benchmark_lineage(events, events[55], amendment)
    record, hashes = _recovery_record(registration, "mixed")
    assert amendment["current_version_number"] == 17
    assert record["strategy_role"] == "mixed"
    assert hashes["package_sha256"] == "d" * 64
    assert _expected_recovery_current_version(events, registration) == 17
    assert _expected_recovery_terminal_status(events, registration) == "COMPLETE"


def test_campaign_v2_cursor_tracks_latest_assigned_terminal_attempt_and_ignores_unassigned_retry():
    from tools.kaggle.cli import _expected_recovery_current_version, _expected_recovery_terminal_status

    events, registration = _campaign_v2_cli_fixture()
    events.extend([
        {"event_type": "workload_attempt", "event_data": {"cycle_id": "g008-competitive-campaign-v2",
          "assigned_version": 18, "terminal_status": "COMPLETE", "outcome": "terminal"}},
        {"event_type": "workload_attempt", "event_data": {"cycle_id": "g008-competitive-campaign-v2",
          "assigned_version": 19, "terminal_status": "ERROR", "outcome": "terminal"}},
        {"event_type": "workload_attempt", "event_data": {"cycle_id": "g008-competitive-campaign-v2",
          "assigned_version": None, "outcome": "pre_assignment_rejected"}},
    ])
    assert _expected_recovery_current_version(events, registration) == 19
    assert _expected_recovery_terminal_status(events, registration) == "ERROR"


def test_campaign_v2_budget_counts_exploratory_v15_once_and_reuses_prior_attestation():
    from tools.kaggle.cli import _validate_recovery_budget

    events, registration = _campaign_v2_cli_fixture()
    events.append({"event_type": "exploratory_kernel_observation_v1",
                   "event_data": {"kernel_id": "owner/kernel", "version_number": 15,
                                  "actual_gpu_seconds": 204, "output_bytes": 222}})

    class Registry:
        def events(self):
            return events

    budget = _validate_recovery_budget(Registry(), registration, 1000, 20_000)
    assert budget["cycle_id"] == "g008-competitive-campaign-v2"
    assert budget["actual"]["training"] == 0
    assert budget["exploratory_gpu_seconds"] == 204
    assert budget["output_bytes"]["training"] == 222
    assert budget["campaign_gpu_seconds_projected"] == 10_200
    assert budget["campaign_gpu_seconds_with_final_reserve"] == 17_400
    assert budget["project_gpu_seconds_projected"] == 10_504
    assert budget["project_gpu_seconds_with_final_reserve"] == 17_704


def test_campaign_v2_retry_budget_amendment_covers_one_exact_error_overhead():
    from tools.kaggle.cli import KaggleError, _validate_recovery_budget

    events, registration = _campaign_v2_cli_fixture()
    failed = {"attempt_id": "g008-v2-pair1-baseline-a1", "cycle_id": "g008-competitive-campaign-v2",
              "registration_event_index": registration["event_index"], "actual_gpu_seconds": 516,
              "output_bytes": 80_238_970}
    events.append({"event_type": "workload_attempt", "event_data": failed})
    events.append({"event_type": "exploratory_kernel_observation_v1",
                   "event_data": {"kernel_id": "owner/kernel", "version_number": 15,
                                  "actual_gpu_seconds": 6, "output_bytes": 100}})
    events.append({"event_type": "g008_competitive_campaign_v2_retry_budget_v1", "event_data": {
        "failed_registration_event_index": 56, "retry_attempt_id": "g008-v2-pair1-baseline-a2",
        "effective_training_gpu_seconds_cap": 7_716,
        "effective_evaluation_gpu_seconds_cap": 4_200,
        "effective_total_gpu_seconds_cap": 11_916,
    }})

    class Registry:
        def events(self):
            return events

    budget = _validate_recovery_budget(
        Registry(), registration, 1_000, 20_000, attempt_id="g008-v2-pair1-baseline-a2")
    assert budget["actual"]["training"] == 516
    assert budget["campaign_gpu_seconds_projected"] == 10_716
    assert budget["project_gpu_seconds_projected"] == 10_822
    with pytest.raises(KaggleError, match="exact retry attempt ID"):
        _validate_recovery_budget(Registry(), registration, 1_000, 20_000, attempt_id="other-retry")


def test_campaign_v2_validation_and_final_cli_gates_follow_comparison_lifecycle():
    from tools.kaggle.cli import KaggleError, _validate_campaign_v2_phase

    events, registration = _campaign_v2_cli_fixture("final")
    with pytest.raises(KaggleError, match="six per-arm results"):
        _validate_campaign_v2_phase(events, registration)
    campaign_id = "g008-competitive-20261021-v2"
    for _ in range(6):
        events.append({"event_type": "competitive_validation_result_v2",
                       "event_data": {"campaign_id": campaign_id}})
    events.append({"event_type": "competitive_strategy_comparison_v2",
                   "event_data": {"campaign_id": campaign_id}})
    events.append({"event_type": "competitive_candidate_freeze_v2",
                   "event_data": {"campaign_id": campaign_id}})
    _validate_campaign_v2_phase(events, registration)


def test_campaign_v2_validation_requires_all_six_training_artifact_attestations():
    from tools.kaggle.cli import KaggleError, _validate_campaign_v2_phase

    events, registration = _campaign_v2_cli_fixture("validation")
    with pytest.raises(KaggleError, match="six COMPLETE training runs"):
        _validate_campaign_v2_phase(events, registration)
    campaign_id = "g008-competitive-20261021-v2"
    for index in range(6):
        events.append({"event_type": "workload_attempt", "event_data": {
            "cycle_id": "g008-competitive-campaign-v2", "registration_event_index": 56 + index,
            "terminal_status": "COMPLETE"}})
        events.append({"event_type": "competitive_training_artifact_v2", "event_data": {
            "campaign_id": campaign_id}})
    _validate_campaign_v2_phase(events, registration)


def test_campaign_recovery_record_uses_registered_strategy_and_exact_hashes():
    from tools.kaggle.cli import _recovery_record

    _, registration = _campaign_cli_fixture("train", "mixed")
    record, hashes = _recovery_record(registration, "mixed")
    assert record["strategy_role"] == "mixed"
    assert hashes["package_sha256"] == "b" * 64
    with pytest.raises(KaggleError, match="exact registered implementation variant"):
        _recovery_record(registration, "baseline")


def test_campaign_training_artifact_reconciliation_binds_cuda_telemetry_and_checkpoint_files(tmp_path):
    from training.ppo.protocol import digest_json, file_sha256
    from tools.kaggle.cli import _validate_campaign_training_output

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    run_id = "23456789-1234-4234-8234-123456789abc"
    seed_sha = "a" * 64
    train_config = {"device": "auto", "total_updates": 2, "seed": 1234,
                    "snapshot_opponent_probability": 0.1,
                    "heuristic_opponent_probability": 1 / 3}
    (source / "run-config.json").write_text(json.dumps({
        "run_id": run_id, "train_config": train_config, "g008_train_schedule": {
            "run_uuid": run_id, "root_seed": 2_100_000, "seed_plan_sha256": seed_sha,
        },
    }))
    updates = [{"update": index, "device": "cuda", "cuda_updates": 1, "optimizer_updates": 1,
                "episodes_by_opponent": {}, "partial_episodes_at_rollout_end": 0,
                "policy_loss": 0.1, "value_loss": 0.2, "entropy": 0.3,
                "approx_kl": 0.01, "clip_fraction": 0.02, "mean_step_reward": 0.4,
                "rollout_seconds": 0.5, "optimizer_seconds": 0.6,
                "update_latency_seconds": 0.7} for index in (1, 2)]
    runtime = {"format": "circuit-ai-kaggle-runtime-evidence/1", "run_id": run_id,
               "python_version": "3.12.0 (test)", "torch_version": "2.5.1+cu121",
               "cuda_version": "12.1", "visible_cuda_devices": ["Tesla T4", "Tesla T4"],
               "visible_cuda_device_count": 2, "training_device": "Tesla T4",
               "training_device_index": 0, "training_cuda_device_count": 1}
    (output / "kaggle-execution-evidence.json").write_text(json.dumps(runtime))
    metrics = {"run_id": run_id, "device": {"device": "cuda", "cuda_available": True,
                "cuda_name": "Tesla T4", "optimizer_updates": 2, "cuda_optimizer_updates": 2},
               "software": {"python": "3.12.0", "torch": "2.5.1+cu121", "cuda": "12.1"},
               "training": {"start_update": 0,
                            "config": {**train_config, "device": "cuda"}, "updates": updates}}
    (output / "training-metrics.json").write_text(json.dumps(metrics))
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    for checkpoint_step in (0, 2):
        pt = checkpoints / f"step-{checkpoint_step}.pt"
        onnx = checkpoints / f"step-{checkpoint_step}.onnx"
        pt.write_bytes(f"pytorch checkpoint {checkpoint_step}".encode())
        onnx.write_bytes(f"onnx checkpoint {checkpoint_step}".encode())
        metadata = {"format": "lc-checkpoint/1", "checkpoint_id": f"step-{checkpoint_step}",
                    "step": checkpoint_step,
                    "pytorch": {"file": f"checkpoints/step-{checkpoint_step}.pt", "sha256": file_sha256(pt)},
                    "onnx": {"file": f"checkpoints/step-{checkpoint_step}.onnx", "sha256": file_sha256(onnx)}}
        metadata["metadata_sha256"] = digest_json(metadata, "metadata_sha256")
        (checkpoints / f"step-{checkpoint_step}.metadata.json").write_text(json.dumps(metadata))
    pt = checkpoints / "step-2.pt"
    record = {"campaign_id": "g008-competitive-20261020-v1", "run_uuid": run_id,
              "root_seed": 2_100_000, "seed_plan_sha256": seed_sha, "update_budget": 2,
              "ppo_config": {"total_updates": 2, "seed": 1234},
              "opponent_probabilities": {"latest": 6000, "step0": 1000, "cruise": 1000,
                                          "local_mobility": 1000, "interceptor": 1000}}
    event = {"event_index": 8, "event_hash": "b" * 64, "event_data": {"record": record}}

    artifact = _validate_campaign_training_output(event, output, source)
    assert artifact["checkpoint_sha256"] == file_sha256(pt)
    manifest = json.loads((output / "g008-training-artifact-manifest.json").read_text())
    assert manifest["run_uuid"] == run_id
    assert manifest["manifest_sha256"] == digest_json(manifest, "manifest_sha256")

    metadata["pytorch"]["sha256"] = "0" * 64
    (checkpoints / "step-2.metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(KaggleError, match="checkpoint metadata"):
        _validate_campaign_training_output(event, output, source)


def test_campaign_evaluation_artifact_reconciliation_verifies_replay_and_roster(tmp_path, monkeypatch):
    from training.ppo.protocol import digest_json, file_sha256
    import training.eval.g008_validation as eval_module
    from tools.kaggle.cli import _validate_campaign_evaluation_output

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    run_id = "23456789-1234-4234-8234-123456789abc"
    candidate_sha = "a" * 64
    roster = {"opponent_id": "g007-step-300", "root_range": (10, 265),
              "roster_sha256": "b" * 64, "opponent_code_sha256": "c" * 64}
    monkeypatch.setitem(eval_module._EXPECTED_ROSTERS, "validation", roster)
    settings = {"campaign_id": "g008-competitive-20261020-v1", "registration_kind": "validation",
                "mode": "validation", "run_uuid": run_id, "roster_sha256": roster["roster_sha256"],
                "opponent_code_sha256": roster["opponent_code_sha256"]}
    run_config_path = source / "run-config.json"
    run_config_path.write_text(json.dumps({"g008_evaluation": settings}))
    episodes = [{"root_seed": root, "target_seat": seat,
                 "result": "win" if root == 10 and seat == 0 else "loss",
                 "failure_diagnosis": (None if root == 10 and seat == 0 else {
                     "category": "wall", "terminal_causes": ["out_of_bounds"]})}
                for root in range(10, 266) for seat in (0, 1)]
    replay_dir = output / "replays"
    replay_dir.mkdir()
    replay = json.loads((Path(__file__).resolve().parents[2]
                         / "shared/fixtures/lc-replay-1.valid.json").read_text())
    replay["run_id"] = run_id
    replay.pop("replay_sha256", None)
    replay["replay_sha256"] = digest_json(replay, "replay_sha256")
    replay_path = replay_dir / "representative-00-win-wall.json"
    replay_path.write_text(json.dumps(replay, sort_keys=True, indent=2) + "\n")
    sample = {"root_seed": 10, "target_seat": 0, "result": "win", "terminal_reason": "wall",
              "replay_file": "replays/representative-00-win-wall.json",
              "replay_sha256": replay["replay_sha256"]}
    report = {"format": "g008-registered-evaluation/1", "status": "complete", "game_count": 512,
              "promotion_eligible": False,
              "campaign_id": settings["campaign_id"], "registration_kind": "validation", "mode": "validation",
              "run_id": run_id, "run_uuid": run_id, "roster_sha256": roster["roster_sha256"],
              "opponent_code_sha256": roster["opponent_code_sha256"], "opponent_id": "g007-step-300",
              "seed_plan_sha256": "d" * 64, "candidate_checkpoint_sha256": candidate_sha,
              "runtime_evidence": {"format": "circuit-ai-kaggle-runtime-evidence/1", "run_id": run_id,
                                   "python_version": "3.12.0 (test)", "torch_version": "2.5.1+cu121",
                                   "cuda_version": "12.1", "visible_cuda_devices": ["Tesla T4", "Tesla T4"],
                                   "visible_cuda_device_count": 2, "training_device": "Tesla T4",
                                   "training_device_index": 0, "training_cuda_device_count": 1},
              "run_config_sha256": digest_json(settings, "__no_digest_field__"),
              "run_config_file_sha256": file_sha256(run_config_path), "episodes": episodes,
              "summary": {"game_count": 512, "wins": 1, "draws": 0, "losses": 511,
                          "failure_diagnostics": {"losses_classified": 511,
                                                  "categories": {"wall": 511}}},
              "representative_replays": [sample]}
    report_path = output / "g008-evaluation.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (output / "kaggle-execution-evidence.json").write_text(
        json.dumps(report["runtime_evidence"], sort_keys=True, indent=2) + "\n")
    files = [{"path": "g008-evaluation.json", "sha256": file_sha256(report_path)},
             {"path": "kaggle-execution-evidence.json",
              "sha256": file_sha256(output / "kaggle-execution-evidence.json")},
             {"path": sample["replay_file"], "sha256": file_sha256(replay_path)}]
    manifest = {"format": "g008-evaluation-artifacts/1", "run_uuid": run_id,
                "run_config_sha256": report["run_config_sha256"],
                "run_config_file_sha256": report["run_config_file_sha256"], "files": files}
    manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
    (output / "g008-evaluation-manifest.json").write_text(json.dumps(manifest))
    record = {"campaign_id": settings["campaign_id"], "run_uuid": run_id,
              "roster_sha256": roster["roster_sha256"], "opponent_code_sha256": roster["opponent_code_sha256"],
              "seed_plan_sha256": report["seed_plan_sha256"]}
    registration = {"event_index": 7, "event_hash": "e" * 64,
                   "event_data": {"registration_kind": "validation", "record": record}}
    attempt = {"event_index": 12, "event_hash": "f" * 64}

    result = _validate_campaign_evaluation_output(registration, attempt, output, candidate_sha, source)
    assert (result["wins"], result["draws"], result["losses"]) == (1, 0, 511)
    replay_path.write_text("tampered")
    with pytest.raises(KaggleError, match="replay"):
        _validate_campaign_evaluation_output(registration, attempt, output, candidate_sha, source)


def test_campaign_recovery_budget_reserves_all_unfinished_workloads_and_final_quota():
    from tools.kaggle.cli import _validate_recovery_budget

    events, registration = _campaign_cli_fixture("train", "baseline")

    class Registry:
        def events(self):
            return events

    budget = _validate_recovery_budget(Registry(), registration, 100, 22_000)
    assert budget["campaign_gpu_seconds_with_final_reserve"] == 8_100
    with pytest.raises(KaggleError, match="unfinished campaign workload"):
        _validate_recovery_budget(Registry(), registration, 100, 7_200)


def test_campaign_expected_version_tracks_last_assigned_attempt_after_retry():
    from tools.kaggle.cli import _expected_recovery_current_version

    events, registration = _campaign_cli_fixture("train", "baseline")
    events.extend([
        {"event_type": "workload_attempt", "event_data": {"cycle_id": "g008-competitive-campaign-v1",
          "assigned_version": 15, "outcome": "terminal"}},
        {"event_type": "workload_attempt", "event_data": {"cycle_id": "g008-competitive-campaign-v1",
          "assigned_version": None, "outcome": "pre_assignment_rejected"}},
    ])

    class Registry:
        def events(self):
            return events

    assert _expected_recovery_current_version(events, registration) == 15


def test_g008_v3_importer_rejects_incomplete_frozen_warmup_roster(tmp_path):
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    root = tmp_path / "benchmark"
    root.mkdir()
    report = {"format": "circuit-ai-g008-principal-benchmark/3",
              "device": {"device": "cuda", "cuda_available": True, "cuda_name": "Tesla T4"}}
    report_path = root / "g008-principal-benchmark.json"
    report_path.write_text(json.dumps(report))
    evidence = {"format": "circuit-ai-kaggle-g008-benchmark-evidence/1", "run_id": "g008-v3",
                "benchmark_report": report, "benchmark_report_sha256": file_sha256(report_path),
                "python_version": "3.11", "torch_version": "2.5", "cuda_version": "12.1",
                "visible_cuda_devices": ["Tesla T4", "Tesla T4"], "visible_cuda_device_count": 2,
                "training_device_index": 0, "training_cuda_device_count": 1}
    (root / "kaggle-execution-evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(KaggleError, match="all five pairs and sixteen warmups"):
        validate_principal_benchmark_artifacts(root, "g008-v3")


def test_recovery_selection_projection_matches_registry_shape_for_v3_report():
    from tools.kaggle.cli import _recovery_selection_evidence

    report = {"parity": {"passed": True, "pairs": {"pair-00": {"passed": True}},
                         "implementation_hashes": {"incumbent": "x", "candidate": "y"}},
              "pairs": [{"candidate_speedup_ratio": 1.2} for _ in range(5)],
              "summary": {"candidate_speedup_ratio": {"median": 1.2, "min": 1.1}},
              "selection": "candidate", "selected_variant_id": "candidate"}
    evidence = _recovery_selection_evidence(report)
    assert evidence == {"parity": {"passed": True},
                        "pairs": [{"candidate_speedup_ratio": 1.2} for _ in range(5)],
                        "summary": {"candidate_speedup_ratio": {"median": 1.2}},
                        "selection": "candidate", "selected_variant_id": "candidate"}


def test_recovery_artifact_rejects_download_root_without_principal_report(tmp_path):
    from tools.kaggle.cli import register_recovery_benchmark_artifact

    attempt_event = {"event_index": 1, "event_hash": "a" * 64, "event_type": "workload_attempt",
                     "event_data": {"outcome": "terminal", "terminal_status": "COMPLETE"}}
    with pytest.raises(KaggleError, match="exactly one principal report artifact directory"):
        register_recovery_benchmark_artifact(object(), attempt_event, tmp_path, tmp_path / "envelope.json")


def test_recovery_artifact_binds_valid_v3_report_and_selection_to_registry(tmp_path, monkeypatch):
    import hashlib
    import statistics
    import uuid

    from tools.kaggle.cli import register_recovery_benchmark_artifact
    from training.eval.provenance import benchmark_run_keys, schedule_spec_digest, seed_plan_sha256
    from training.ppo.g008_benchmark import (CONFIG_PATH, CANDIDATE_SOURCE_COMMIT,
                                             INCUMBENT_SOURCE_COMMIT, load_principal_config,
                                             prepare_benchmark_schedules)

    download_root = tmp_path / "downloaded"
    artifact_root = download_root / "arena-output" / "g008-valid-v3"
    artifact_root.mkdir(parents=True)
    run_keys = benchmark_run_keys()
    run_ids = {key: str(uuid.uuid5(uuid.NAMESPACE_URL, f"fake-valid-v3:{key}")) for key in run_keys}
    schedules = prepare_benchmark_schedules(run_ids)
    config, config_doc = load_principal_config()
    implementations = {
        role: {"trainer_sha256": hashlib.sha256(f"{role}-trainer".encode()).hexdigest(),
               "engine_sha256": hashlib.sha256(f"{role}-engine".encode()).hexdigest()}
        for role in ("incumbent", "candidate")
    }
    source_sha = hashlib.sha256(json.dumps(implementations, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False).encode()).hexdigest()
    commits = {"incumbent": INCUMBENT_SOURCE_COMMIT, "candidate": CANDIDATE_SOURCE_COMMIT}
    expected_steps = ((config.arena_count * config.rollout_steps + config.minibatch_size - 1)
                      // config.minibatch_size) * config.update_epochs
    transitions = config.arena_count * config.rollout_steps * config.total_updates
    metric_hashes = {}

    def write_metrics(relative, update_count, latency):
        updates = [{"device": "cuda", "cuda_updates": expected_steps,
                    "optimizer_updates": expected_steps, "optimizer_seconds": 1.0,
                    "checkpoint_seconds": 0.1, "update_latency_seconds": latency}
                   for _ in range(update_count)]
        metrics = {"format": "circuit-ai-g008-benchmark-run-metrics/1",
                   "device": {"device": "cuda", "cuda_available": True, "cuda_name": "Tesla T4",
                              "optimizer_updates": expected_steps * update_count,
                              "cuda_optimizer_updates": expected_steps * update_count},
                   "training": {"updates": updates, "mean_rollout_seconds": 1.0,
                                "update_elapsed_seconds": update_count * latency,
                                "elapsed_seconds": update_count * (latency + 1.0)}}
        path = artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
        digest = file_sha256(path)
        metric_hashes[relative] = digest
        return metrics, digest

    def arm_metrics(run_key, role, latency):
        relative = f"measured/{run_key}/{role}/training-metrics.json"
        _, digest = write_metrics(relative, config.total_updates, latency)
        update_only = config.total_updates * (latency - 0.1)
        rollout = config.total_updates * 1.0
        optimizer = config.total_updates * 1.0
        checkpoint = config.total_updates * 0.1
        update_elapsed = config.total_updates * latency
        wall = config.total_updates * (latency + 1.0)
        speed = transitions / update_only
        decisions = transitions * 2
        return {"implementation": {"role": role, "variant_id": role,
                                    "source_commit": commits[role], **implementations[role]},
                "run_key": run_key, "run_id": run_ids[run_key], "seed": config.seed,
                "config_sha256": config_doc["config_sha256"],
                "expanded_schedule_digest": schedules[run_key]["expanded_schedule_digest"],
                "completed_joint_arena_transitions": transitions, "agent_decisions": decisions,
                "update_only_joint_transitions_per_second": speed,
                "wall_inclusive_joint_transitions_per_second": transitions / wall,
                "update_only_agent_decisions_per_second": decisions / update_only,
                "wall_inclusive_agent_decisions_per_second": decisions / wall,
                "rollout_seconds": rollout, "optimizer_seconds": optimizer,
                "update_only_seconds": update_only,
                "checkpoint_inclusive_update_elapsed_seconds": update_elapsed,
                "checkpoint_seconds": checkpoint, "wall_seconds": wall,
                "optimizer_updates": expected_steps * config.total_updates,
                "cuda_optimizer_updates": expected_steps * config.total_updates,
                "training_metrics_file": relative, "training_metrics_sha256": digest}

    parity_pairs = {}
    for index in range(5):
        key = f"pair-{index:02d}"
        trace = {"sha256": f"{index + 1:064x}", "transition_count": config.arena_count * config.rollout_steps,
                 "reset_count": 7,
                 "action_source": "g008-trainer-model-sampled-collector/1"}
        parity_pairs[key] = {"passed": True, "incumbent": dict(trace), "candidate": dict(trace),
                             "implementation_hashes": implementations}
    pairs, pair_roles = [], {"incumbent": {}, "candidate": {}}
    for index in range(5):
        key = f"pair-{index:02d}"
        incumbent = arm_metrics(key, "incumbent", 4.1)
        candidate = arm_metrics(key, "candidate", 3.1)
        for role, arm in (("incumbent", incumbent), ("candidate", candidate)):
            for field in ("wall_inclusive_joint_transitions_per_second", "update_only_seconds",
                          "wall_seconds", "optimizer_seconds", "rollout_seconds"):
                pair_roles[role].setdefault(field, []).append(arm[field])
        ratio = candidate["update_only_joint_transitions_per_second"] / incumbent["update_only_joint_transitions_per_second"]
        pairs.append({"repeat": index, "run_key": key, "run_id": run_ids[key],
                      "order": ["incumbent", "candidate"] if index % 2 == 0 else ["candidate", "incumbent"],
                      "seed": config.seed, "config_sha256": config_doc["config_sha256"],
                      "expanded_schedule_digest": schedules[key]["expanded_schedule_digest"],
                      "parity_sha256": parity_pairs[key]["incumbent"]["sha256"],
                      "candidate_speedup_ratio": ratio,
                      "arms": {"incumbent": incumbent, "candidate": candidate}})
    ratios = [pair["candidate_speedup_ratio"] for pair in pairs]

    def percentiles(values):
        ordered = sorted(values)
        q1 = statistics.median(ordered[:len(ordered) // 2])
        q3 = statistics.median(ordered[(len(ordered) + 1) // 2:])
        return {"median": statistics.median(ordered), "min": min(ordered),
                "max": max(ordered), "iqr": q3 - q1}

    summary = {
        "candidate_update_only_joint_transitions_per_second": percentiles(
            [pair["arms"]["candidate"]["update_only_joint_transitions_per_second"] for pair in pairs]),
        "incumbent_update_only_joint_transitions_per_second": percentiles(
            [pair["arms"]["incumbent"]["update_only_joint_transitions_per_second"] for pair in pairs]),
        "candidate_speedup_ratio": percentiles(ratios),
    }
    for role, fields in pair_roles.items():
        for field, values in fields.items():
            summary[f"{role}_{field}"] = percentiles(values)

    warmups = []
    for key in run_keys:
        if not key.startswith("warmup-"):
            continue
        backend, role, repeat = key.removeprefix("warmup-").rsplit("-", 2)
        relative = f"warmups/{backend}/{role}/warmup-{repeat}/training-metrics.json"
        _, digest = write_metrics(relative, 1, 4.1)
        warmups.append({"backend_id": backend, "role": role, "run": int(repeat), "run_key": key,
                        "run_id": run_ids[key],
                        "implementation": {"role": role, "variant_id": role,
                                           "source_commit": commits[role], **implementations[role]},
                        "discarded": True,
                        "expanded_schedule_digest": schedules[key]["expanded_schedule_digest"],
                        "training_metrics_file": relative, "training_metrics_sha256": digest,
                        "completed_joint_arena_transitions": config.arena_count * config.rollout_steps})
    report = {
        "format": "circuit-ai-g008-principal-benchmark/3", "selection": "candidate",
        "selected_variant_id": "candidate", "selection_rule": {"parity_required": True},
        "implementations": implementations, "implementation_source_commits": commits,
        "source_sha256": source_sha, "source": config_doc["source"], "config": config.__dict__,
        "config_sha256": config_doc["config_sha256"], "config_artifact_sha256": file_sha256(CONFIG_PATH),
        "seed_plan_sha256": seed_plan_sha256(), "schedule_spec_digest": schedule_spec_digest(),
        "schedule_roster": [{"run_key": key, "run_id": run_ids[key],
                              "expanded_schedule_digest": schedules[key]["expanded_schedule_digest"]}
                             for key in run_keys],
        "warmups": {"count_per_role": 2, "updates_each": 1, "discarded": True, "runs": warmups},
        "measured_repeats": 5, "updates_per_repeat": config.total_updates,
        "parity": {"passed": True, "pairs": parity_pairs, "implementation_hashes": implementations},
        "device": {"device": "cuda", "cuda_available": True, "cuda_name": "Tesla T4",
                   "optimizer_updates": expected_steps, "cuda_optimizer_updates": expected_steps},
        "summary": summary,
        "optimizer_updates_total": sum(arm["optimizer_updates"] for pair in pairs for arm in pair["arms"].values()),
        "cuda_optimizer_updates_total": sum(arm["cuda_optimizer_updates"] for pair in pairs for arm in pair["arms"].values()),
        "pairs": pairs,
    }
    report_path = artifact_root / "g008-principal-benchmark.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    evidence = {"format": "circuit-ai-kaggle-g008-benchmark-evidence/1", "run_id": "g008-valid-v3",
                "python_version": "3.11.11", "torch_version": "2.5.1+cu124", "cuda_version": "12.4",
                "visible_cuda_devices": ["Tesla T4", "Tesla T4"], "visible_cuda_device_count": 2,
                "training_device_index": 0, "training_cuda_device_count": 1,
                "benchmark_report": report,
                "benchmark_report_sha256": file_sha256(report_path)}
    (artifact_root / "kaggle-execution-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    registration_event = {"event_index": 1, "event_hash": "a" * 64,
                          "event_type": "registration", "event_data": {}}
    attempt_data = {
        "attempt_id": "attempt-v3", "outcome": "terminal", "terminal_status": "COMPLETE",
        "registration_event_index": 1, "registration_event_hash": registration_event["event_hash"],
        "assigned_version": 11, "terminal_metadata": {"kernel_id": "owner/kernel", "status": "COMPLETE",
                                                            "current_version_number": 11},
        "status_evidence": [{"kernel_id": "owner/kernel", "status": "COMPLETE",
                             "current_version_number": 11}],
        "implementation_variant_id": "paired", "package_sha256": "b" * 64,
        "source_sha256": source_sha, "config_sha256": config_doc["config_sha256"],
        "seed_plan_sha256": seed_plan_sha256(), "backend_sha256": "c" * 64,
        "trainer_sha256": "d" * 64, "engine_sha256": "e" * 64,
    }
    attempt_event = {"event_index": 3, "event_hash": "f" * 64,
                     "event_type": "workload_attempt", "event_data": attempt_data}

    class ArtifactRegistry:
        path = tmp_path / "registry.jsonl"

        def __init__(self):
            self.recorded = None

        def events(self):
            return [{"event_index": 0, "event_hash": "9" * 64}, registration_event,
                    {"event_index": 2, "event_hash": "0" * 64}, attempt_event]

        def register_benchmark_artifact(self, artifact_sha256, registration_index, registration_hash, **kwargs):
            self.recorded = {"artifact_sha256": artifact_sha256, "registration_event_index": registration_index,
                             "registration_event_hash": registration_hash, **kwargs}
            return self.recorded

    registry = ArtifactRegistry()
    envelope_path = download_root / "g008-recovery-benchmark-artifact.json"
    result = register_recovery_benchmark_artifact(registry, attempt_event, download_root, envelope_path)
    expected_evidence = {"parity": {"passed": True},
                         "pairs": [{"candidate_speedup_ratio": ratio} for ratio in ratios],
                         "summary": {"candidate_speedup_ratio": {"median": statistics.median(ratios)}},
                         "selection": "candidate", "selected_variant_id": "candidate"}
    assert result == registry.recorded
    assert result["selection_evidence"] == expected_evidence
    assert result["report_sha256"] == file_sha256(report_path)
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["raw_metrics_sha256"] == metric_hashes
    assert envelope["attempt"]["assigned_version"] == 11

    duplicate_root = download_root / "other-output"
    duplicate_root.mkdir()
    (duplicate_root / "g008-principal-benchmark.json").write_text(report_path.read_text(encoding="utf-8"),
                                                                    encoding="utf-8")
    with pytest.raises(KaggleError, match="exactly one principal report artifact directory"):
        register_recovery_benchmark_artifact(registry, attempt_event, download_root,
                                             download_root / "ambiguous-envelope.json")

    from training.ppo import g008_benchmark
    monkeypatch.setattr(g008_benchmark, "prepare_benchmark_schedules", lambda _: schedules)
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    def refresh_report_evidence():
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        evidence["benchmark_report"] = report
        evidence["benchmark_report_sha256"] = file_sha256(report_path)
        (artifact_root / "kaggle-execution-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    legacy_fields = {
        "candidate_update_only_joint_transitions_per_second": [
            pair["arms"]["candidate"]["update_only_joint_transitions_per_second"] for pair in pairs],
        "incumbent_update_only_joint_transitions_per_second": [
            pair["arms"]["incumbent"]["update_only_joint_transitions_per_second"] for pair in pairs],
        "candidate_speedup_ratio": [pair["candidate_speedup_ratio"] for pair in pairs],
        "wall_inclusive_joint_transitions_per_second": [
            pair["arms"]["candidate"]["wall_inclusive_joint_transitions_per_second"] for pair in pairs],
        "update_only_seconds": [pair["arms"]["candidate"]["update_only_seconds"] for pair in pairs],
        "wall_seconds": [pair["arms"]["candidate"]["wall_seconds"] for pair in pairs],
        "optimizer_seconds": [pair["arms"]["candidate"]["optimizer_seconds"] for pair in pairs],
        "rollout_seconds": [pair["arms"]["candidate"]["rollout_seconds"] for pair in pairs],
    }
    report["summary"] = {field: percentiles(values) for field, values in legacy_fields.items()}
    refresh_report_evidence()
    validate_principal_benchmark_artifacts(artifact_root, "g008-valid-v3")

    report["summary"]["wall_seconds"]["median"] += 1
    refresh_report_evidence()
    with pytest.raises(KaggleError, match="summary or selected implementation is invalid"):
        validate_principal_benchmark_artifacts(artifact_root, "g008-valid-v3")
    report["summary"] = {field: percentiles(values) for field, values in legacy_fields.items()}
    refresh_report_evidence()

    report["parity"]["pairs"]["pair-00"]["candidate"]["sha256"] = "0" * 64
    refresh_report_evidence()
    with pytest.raises(KaggleError, match="passed flag contradicts its two parity trace hashes"):
        validate_principal_benchmark_artifacts(artifact_root, "g008-valid-v3")

    traces = report["parity"]["pairs"]["pair-00"]
    traces["candidate"]["sha256"] = traces["incumbent"]["sha256"]
    traces["incumbent"]["reset_count"] = config.arena_count * config.rollout_steps + 1
    refresh_report_evidence()
    with pytest.raises(KaggleError, match="parity trace digest/count/action source is invalid"):
        validate_principal_benchmark_artifacts(artifact_root, "g008-valid-v3")

    traces["incumbent"]["reset_count"] = 7
    traces["candidate"]["reset_count"] = 8
    refresh_report_evidence()
    with pytest.raises(KaggleError, match="contradictory reset counts"):
        validate_principal_benchmark_artifacts(artifact_root, "g008-valid-v3")


def test_download_hash_validates_benchmark_report_in_manifest(tmp_path):
    destination = tmp_path / "artifacts/run-42"
    def run(args):
        if args[2] == "status":
            return "Kernel status: Complete"
        root = Path(args[args.index("-p") + 1])
        make_run(root)
        report = root / "benchmark-results.json"
        report.write_text(json.dumps({"format": "lc-throughput-benchmark/1", "selected_trial": 0}))
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"].append({"path": "benchmark-results.json", "sha256": file_sha256(report)})
        manifest["benchmark_records"] = [{"protocol": "lc-throughput-benchmark/1",
                                           "report_file": "benchmark-results.json"}]
        manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
        manifest_path.write_text(json.dumps(manifest))
        return "downloaded"
    result = download("me/light-cycles", 5, "run-42", destination, run=run)
    assert (result / "benchmark-results.json").is_file()
    assert json.loads((result / "manifest.json").read_text())["benchmark_records"][0]["report_file"] == "benchmark-results.json"


def test_download_refuses_tampered_benchmark_report(tmp_path):
    destination = tmp_path / "artifacts/run-42"
    def run(args):
        if args[2] == "status":
            return "Kernel status: Complete"
        root = Path(args[args.index("-p") + 1])
        make_run(root)
        report = root / "benchmark-results.json"
        report.write_text("{}")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"].append({"path": "benchmark-results.json", "sha256": file_sha256(report)})
        manifest["benchmark_records"] = [{"report_file": "benchmark-results.json"}]
        manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
        manifest_path.write_text(json.dumps(manifest))
        report.write_text("tampered")
        return "downloaded"
    with pytest.raises(KaggleError, match="SHA-256 mismatch: benchmark-results.json"):
        download("me/light-cycles", 5, "run-42", destination, run=run)
    assert not destination.exists()


def test_download_refuses_bad_hash_and_preserves_no_partial_target(tmp_path):
    destination = tmp_path / "artifacts/run-42"
    def run(args):
        if args[2] == "status":
            return "Kernel status: Complete"
        stage = Path(args[args.index("-p") + 1])
        make_run(stage)
        (stage / "checkpoints/step-0.pt").write_bytes(b"tampered")
        return "downloaded"
    with pytest.raises(KaggleError, match="SHA-256"):
        download("me/light-cycles", 5, "run-42", destination, run=run)
    assert not destination.exists()
    assert not list(destination.parent.glob(".run-42.staging-*"))


def test_download_never_runs_for_running_or_failed_kernel(tmp_path):
    for response in ("Kernel status: Running", "Kernel status: Error"):
        calls = []
        with pytest.raises(KaggleError, match="no download was attempted"):
            download("me/light-cycles", 2, "run-42", tmp_path / "run",
                     max_attempts=1, run=lambda args: calls.append(list(args)) or response)
        assert len(calls) == 1 and calls[0][2] == "status"


@pytest.mark.parametrize("mutation", ["unlisted", "mismatched"])
def test_download_refuses_missing_or_mismatched_manifest_checkpoint(tmp_path, mutation):
    destination = tmp_path / "artifacts/run-42"
    def run(args):
        if args[2] == "status":
            return "Kernel status: Complete"
        root = Path(args[args.index("-p") + 1])
        make_run(root)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        checkpoint = manifest["checkpoints"][0]
        if mutation == "unlisted":
            manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != checkpoint["file"]]
        else:
            checkpoint["sha256"] = "0" * 64
        manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
        manifest_path.write_text(json.dumps(manifest))
        return "downloaded"
    with pytest.raises(KaggleError, match="Checkpoint is missing from the run file hashes"):
        download("me/light-cycles", 5, "run-42", destination, run=run)
    assert not destination.exists()


@pytest.mark.parametrize("listed", [False, True])
def test_download_refuses_unsupported_onnx_external_data_sidecar(tmp_path, listed):
    destination = tmp_path / "artifacts/run-42"

    def run(args):
        if args[2] == "status":
            return "Kernel status: Complete"
        root = Path(args[args.index("-p") + 1])
        make_run(root)
        sidecar = root / "checkpoints/step-0.onnx.data"
        sidecar.write_bytes(b"external weights")
        if listed:
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"].append({"path": "checkpoints/step-0.onnx.data", "sha256": file_sha256(sidecar)})
            manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
            manifest_path.write_text(json.dumps(manifest))
        return "downloaded"

    with pytest.raises(KaggleError, match="Viewer requires single-file ONNX; external-data sidecar is unsupported"):
        download("me/light-cycles", 5, "run-42", destination, run=run)
    assert not destination.exists()


@pytest.mark.parametrize("mutation", ["missing-onnx", "duplicate-pytorch"])
def test_download_requires_exact_pytorch_onnx_checkpoint_pair(tmp_path, mutation):
    destination = tmp_path / "artifacts/run-42"
    def run(args):
        if args[2] == "status":
            return "Kernel status: Complete"
        root = Path(args[args.index("-p") + 1])
        make_run(root)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        target = next(entry for entry in manifest["checkpoints"] if entry["checkpoint_id"] == "step-0")
        if mutation == "missing-onnx":
            manifest["checkpoints"].remove(next(entry for entry in manifest["checkpoints"]
                                                 if entry["checkpoint_id"] == "step-0" and entry["format"] == "onnx"))
        else:
            duplicate = dict(target)
            duplicate["file"] = "checkpoints/step-0-duplicate.pt"
            manifest["checkpoints"].append(duplicate)
        manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
        manifest_path.write_text(json.dumps(manifest))
        return "downloaded"
    with pytest.raises(KaggleError, match="exactly one PyTorch and one ONNX reference"):
        download("me/light-cycles", 5, "run-42", destination, run=run)
    assert not destination.exists()


def test_download_rejects_schema_valid_but_impossible_empty_replay(tmp_path):
    destination = tmp_path / "artifacts/run-42"
    def run(args):
        if args[2] == "status":
            return "Kernel status: Complete"
        root = Path(args[args.index("-p") + 1])
        make_run(root)
        replay_path = root / "replays/replay-0.json"
        replay = json.loads(replay_path.read_text())
        replay["ticks"] = []
        replay["tick_count"] = 0
        replay["replay_sha256"] = digest_json(replay, "replay_sha256")
        replay_path.write_text(json.dumps(replay), encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["files"]:
            if entry["path"] == "replays/replay-0.json":
                entry["sha256"] = file_sha256(replay_path)
        manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return "downloaded"
    with pytest.raises(KaggleError, match="does not reach its declared terminal result"):
        download("me/light-cycles", 5, "run-42", destination, run=run)
    assert not destination.exists()


def test_download_rejects_mismatched_paired_checkpoint_metadata(tmp_path):
    destination = tmp_path / "artifacts/run-42"
    def run(args):
        if args[2] == "status":
            return "Kernel status: Complete"
        root = Path(args[args.index("-p") + 1])
        make_run(root)
        metadata_path = root / "checkpoints/step-0.metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["onnx"]["sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["files"]:
            if entry["path"] == "checkpoints/step-0.metadata.json":
                entry["sha256"] = file_sha256(metadata_path)
        manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return "downloaded"
    with pytest.raises(KaggleError, match="Checkpoint metadata digest is invalid"):
        download("me/light-cycles", 5, "run-42", destination, run=run)
    assert not destination.exists()


def test_prepare_builds_deterministic_self_contained_package_without_kaggle(tmp_path):
    config = tmp_path / "run-config.json"
    config.write_text(json.dumps({"run_id": "run-local", "train_config": {"device": "cuda", "total_updates": 2}}))
    first = prepare("me/light-cycles", "source-r5", tmp_path / "one", config)
    second = prepare("me/light-cycles", "source-r5", tmp_path / "two", config)
    first_files = {path.relative_to(first).as_posix(): path.read_bytes()
                   for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second).as_posix(): path.read_bytes()
                    for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files
    metadata = json.loads((first / "kernel-metadata.json").read_text())
    assert metadata["id"] == "me/light-cycles"
    assert metadata["code_file"] == "run_kaggle.py"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert (first / ".arena-source-revision").read_text() == "source-r5\n"
    source_manifest = json.loads((first / "source-manifest.json").read_text())
    assert source_manifest["format"] == "circuit-ai-source-manifest/1"
    assert source_manifest["source_revision"] == "source-r5"
    assert source_manifest["git_commit"]
    assert isinstance(source_manifest["git_worktree_dirty"], bool)
    assert source_manifest["package_builder"]["sha256"] == file_sha256(Path("tools/kaggle/cli.py"))
    assert len(source_manifest["runner_template_sha256"]) == 64
    source_paths = {item["path"] for item in source_manifest["files"]}
    assert {"training/requirements-kaggle.txt", "run-config.json", "kernel-metadata.json",
            ".arena-source-revision"} <= source_paths
    assert source_manifest["source_manifest_sha256"] == hashlib.sha256(json.dumps(
        {key: value for key, value in source_manifest.items() if key != "source_manifest_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert any(item["path"] == "training/ppo/train.py" for item in source_manifest["files"])
    assert (first / "training/arena/engine.py").is_file()
    runner = (first / "run_kaggle.py").read_text()
    assert '"/kaggle/working/arena-output"' in runner
    assert "requires an available CUDA GPU" in runner
    assert 'requested_device == "cpu"' in runner
    assert 'settings["train_config"]["device"] = "cuda"' in runner
    requirements = (first / "training/requirements-kaggle.txt").read_text().lower()
    assert "--no-deps" in runner
    assert not any(line.strip().startswith("torch") for line in requirements.splitlines())
    compile(runner, str(first / "run_kaggle.py"), "exec")
    assert "import kaggle" not in runner
    namespace = {"__name__": "runner_bootstrap_test", "__file__": str(first / "run_kaggle.py")}
    exec(compile(runner, str(first / "run_kaggle.py"), "exec"), namespace)
    embedded_root = Path(namespace["ROOT"])
    try:
        assert (embedded_root / "run-config.json").is_file()
        assert (embedded_root / "source-manifest.json").read_bytes() == (first / "source-manifest.json").read_bytes()
        assert (embedded_root / "training/ppo/train.py").is_file()
        assert (embedded_root / "training/ppo/g008_incumbent").is_dir()
        assert (embedded_root / "schemas/g008-split-seed-plan-3.schema.json").is_file()
        incumbent_root = embedded_root / "training/ppo/g008_incumbent"
        assert {"train.py", "engine.py", "manifest.json"} <= {path.name for path in incumbent_root.iterdir()}
        incumbent_manifest = json.loads((incumbent_root / "manifest.json").read_text())
        assert incumbent_manifest["files"]["train.py"] == file_sha256(incumbent_root / "train.py")
        assert incumbent_manifest["files"]["engine.py"] == file_sha256(incumbent_root / "engine.py")
        assert (embedded_root / "schemas/lc-run-1.schema.json").is_file()
    finally:
        namespace["SOURCE_TMP"].cleanup()


def test_prepare_accepts_only_bounded_throughput_benchmark_config(tmp_path):
    config = tmp_path / "benchmark-config.json"
    config.write_text(json.dumps({"run_id": "run-bench", "train_config": {"device": "cuda"},
                                  "throughput_benchmark": {"configs": [{"arena_count": 4}], "updates": 2}}))
    package = prepare("me/light-cycles", "source-bench", tmp_path / "package", config)
    staged = json.loads((package / "run-config.json").read_text())
    assert staged["throughput_benchmark"] == {"configs": [{"arena_count": 4}], "updates": 2}
    runner = (package / "run_kaggle.py").read_text()
    assert "run_benchmark(candidates, benchmark_root)" in runner
    assert "benchmark-results.json" in runner


def test_prepare_accepts_preregistered_g008_principal_benchmark(tmp_path):
    import uuid
    from training.eval.provenance import benchmark_run_keys

    run_ids = {key: str(uuid.uuid5(uuid.NAMESPACE_URL, f"g008-test:{key}"))
               for key in benchmark_run_keys()}
    config = tmp_path / "g008-config.json"
    config.write_text(json.dumps({"run_id": "g008-principal-r1", "train_config": {"device": "auto"},
                                  "g008_principal_benchmark": {"run_ids": run_ids}}))
    package = prepare("me/circuit-ai", "g008-test", tmp_path / "package", config)
    staged = json.loads((package / "run-config.json").read_text())
    assert staged["g008_principal_benchmark"]["run_ids"] == run_ids
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    runner = (package / "run_kaggle.py").read_text()
    assert "run_principal_benchmark(output, run_ids=g008_settings[\"run_ids\"])" in runner
    assert "kaggle-execution-evidence.json" in runner


def test_prepare_rejects_unregistered_g008_run_roster(tmp_path):
    config = tmp_path / "g008-config.json"
    config.write_text(json.dumps({"run_id": "g008-invalid", "train_config": {"device": "auto"},
                                  "g008_principal_benchmark": {"run_ids": {"pair-00": "not-a-uuid"}}}))
    with pytest.raises(KaggleError, match="preregistered G008 benchmark schedule"):
        prepare("me/circuit-ai", "g008-test", tmp_path / "package", config)
    assert not (tmp_path / "package").exists()


def test_prepare_rejects_modified_frozen_seed_plan_before_publishing_package(tmp_path, monkeypatch):
    import uuid
    from training.eval.provenance import SEED_PLAN_PATH, benchmark_run_keys

    altered_plan = json.loads(SEED_PLAN_PATH.read_text())
    altered_plan["protocol"] += " modified"
    plan_path = tmp_path / "altered-seed-plan.json"
    plan_path.write_text(json.dumps(altered_plan))
    run_ids = {key: str(uuid.uuid5(uuid.NAMESPACE_URL, f"modified-plan:{key}"))
               for key in benchmark_run_keys()}
    monkeypatch.setattr("training.eval.provenance.SEED_PLAN_PATH", plan_path)
    config = tmp_path / "g008-config.json"
    config.write_text(json.dumps({"run_id": "g008-invalid-plan", "train_config": {"device": "auto"},
                                  "g008_principal_benchmark": {"run_ids": run_ids}}))

    with pytest.raises(KaggleError, match="preregistered G008 benchmark schedule"):
        prepare("me/circuit-ai", "g008-test", tmp_path / "package", config)
    assert not (tmp_path / "package").exists()


def test_prepare_freezes_g008_train_schedule_and_uuid(tmp_path):
    import uuid
    from training.eval.provenance import build_training_run_schedule, seed_plan_sha256

    run_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "g008-train-package"))
    config = tmp_path / "g008-train-config.json"
    config.write_text(json.dumps({
        "run_id": run_uuid,
        "train_config": g008_principal_train_config(),
        "g008_train_schedule": {"run_uuid": run_uuid, "root_seed": 2_000_000},
    }))

    settings = _load_run_settings(config)
    expected = build_training_run_schedule(run_id=run_uuid, root_seed=2_000_000,
                                           arena_count=128, rollout_steps=32, total_updates=3)
    assert settings["g008_train_schedule"] == {
        "run_uuid": run_uuid,
        "root_seed": 2_000_000,
        "expanded_schedule_digest": expected["expanded_schedule_digest"],
        "schedule_spec_digest": expected["schedule_spec_digest"],
        "seed_plan_sha256": seed_plan_sha256(),
    }
    assert settings == _load_run_settings(config)

    package = prepare("me/circuit-ai", "g008-train", tmp_path / "package", config)
    staged = json.loads((package / "run-config.json").read_text())
    assert staged["g008_train_schedule"] == settings["g008_train_schedule"]
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["machine_shape"] == "NvidiaTeslaT4"


@pytest.mark.parametrize(("pair_id", "role", "ppo_seed", "root_seed"), [
    ("seed-pair-1", "baseline", 3301, 2_100_001),
    ("seed-pair-2", "mixed", 3302, 2_200_001),
])
def test_load_run_settings_binds_v2_campaign_train_config_and_compact_seed_schedule(
        tmp_path, pair_id, role, ppo_seed, root_seed):
    from dataclasses import asdict, replace
    from training.eval.provenance import seed_plan_sha256
    from training.ppo.g008_benchmark import load_principal_config
    from training.ppo.protocol import digest_json
    from training.ppo.train import g008_campaign_schedule_descriptor

    run_uuid = "23456789-1234-4234-8234-123456789abc"
    seed_sha = seed_plan_sha256()
    base, _ = load_principal_config()
    config_obj = replace(base, device="auto", seed=ppo_seed, total_updates=1000,
                         checkpoint_every=50,
                         snapshot_opponent_probability=0.1 if role == "mixed" else 0.0,
                         heuristic_opponent_probability=(1 / 3) if role == "mixed" else 0.0)
    config = asdict(config_obj)
    shared = {name: value for name, value in config.items()
              if name not in {"device", "seed", "snapshot_opponent_probability",
                              "heuristic_opponent_probability"}}
    descriptor = g008_campaign_schedule_descriptor(
        run_uuid=run_uuid, root_seed=root_seed, arena_count=config_obj.arena_count,
        total_updates=config_obj.total_updates, rollout_steps=config_obj.rollout_steps,
        seed_plan_sha=seed_sha)
    probabilities = ({"latest": 10_000, "step0": 0, "cruise": 0,
                      "local_mobility": 0, "interceptor": 0} if role == "baseline" else
                     {"latest": 6_000, "step0": 1_000, "cruise": 1_000,
                      "local_mobility": 1_000, "interceptor": 1_000})
    training = {
        "campaign_id": "g008-competitive-20261021-v2", "registration_id": "v2-train-1",
        "run_uuid": run_uuid, "pair_id": pair_id, "strategy_role": role,
        "root_seed": root_seed, "ppo_seed": ppo_seed, "update_budget": 1_000,
        "seed_plan_sha256": seed_sha, "opponent_probabilities": probabilities,
        "config_sha256": digest_json(config, ""), "shared_config_sha256": digest_json(shared, ""),
        "schedule_descriptor_sha256": descriptor["schedule_descriptor_sha256"],
    }
    path = tmp_path / "g008-campaign-train.json"
    path.write_text(json.dumps({"run_id": run_uuid, "train_config": config,
                                "g008_campaign_train": training}))

    loaded = _load_run_settings(path)

    assert loaded["g008_campaign_train"] == training
    assert loaded["run_id"] == run_uuid
    package = prepare("me/circuit-ai", f"v2-{pair_id}-{role}", tmp_path / "package", path)
    runner_source = (package / "run_kaggle.py").read_text(encoding="utf-8")
    assert 'g008_campaign_train = settings.get("g008_campaign_train")' in runner_source
    assert "run_evaluation=False" in runner_source
    assert 'run_id=g008_campaign_train["run_uuid"]' in runner_source
    assert '"g008_campaign_train"' in runner_source and "Registered G008 workloads require cuda:0 to be a Tesla T4" in runner_source
    assert "from dataclasses import asdict, dataclass, replace" in (
        package / "training/ppo/train.py").read_text(encoding="utf-8")
    _assert_nonfinal_campaign_archive_has_no_final_holdout(package)

    training["config_sha256"] = "f" * 64
    path.write_text(json.dumps({"run_id": run_uuid, "train_config": config,
                                "g008_campaign_train": training}))
    with pytest.raises(KaggleError, match="full TrainConfig hash"):
        _load_run_settings(path)


def test_load_run_settings_rejects_campaign_train_depth_drift(tmp_path):
    from dataclasses import asdict, replace
    from training.eval.provenance import seed_plan_sha256
    from training.ppo.g008_benchmark import load_principal_config
    from training.ppo.protocol import digest_json
    from training.ppo.train import g008_campaign_schedule_descriptor

    run_uuid = "23456789-1234-4234-8234-123456789abc"
    seed_sha = seed_plan_sha256()
    base, _ = load_principal_config()
    config = asdict(replace(base, device="auto", seed=3301, total_updates=3, checkpoint_every=50))
    shared = {name: value for name, value in config.items()
              if name not in {"device", "seed", "snapshot_opponent_probability",
                              "heuristic_opponent_probability"}}
    descriptor = g008_campaign_schedule_descriptor(
        run_uuid=run_uuid, root_seed=2_100_001, arena_count=config["arena_count"],
        total_updates=1000, rollout_steps=config["rollout_steps"], seed_plan_sha=seed_sha)
    path = tmp_path / "invalid-campaign-train.json"
    path.write_text(json.dumps({"run_id": run_uuid, "train_config": config,
        "g008_campaign_train": {
            "campaign_id": "g008-competitive-20261021-v2", "registration_id": "v2-train-1",
            "run_uuid": run_uuid, "pair_id": "seed-pair-1", "strategy_role": "baseline",
            "root_seed": 2_100_001, "ppo_seed": 3301, "update_budget": 1000,
            "seed_plan_sha256": seed_sha,
            "opponent_probabilities": {"latest": 10000, "step0": 0, "cruise": 0,
                                       "local_mobility": 0, "interceptor": 0},
            "config_sha256": digest_json(config, ""), "shared_config_sha256": digest_json(shared, ""),
            "schedule_descriptor_sha256": descriptor["schedule_descriptor_sha256"],
        }}))
    with pytest.raises(KaggleError, match="TrainConfig differs from the frozen v2 training baseline"):
        _load_run_settings(path)


def test_campaign_train_config_registry_wire_round_trips_float_fields():
    from tools.kaggle.cli import _campaign_train_config_native, _campaign_train_config_wire

    source = {"device": "auto", "seed": 3302, "learning_rate": 0.0003,
              "gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2,
              "entropy_coef": 0.01, "value_coef": 0.5,
              "snapshot_opponent_probability": 0.1,
              "heuristic_opponent_probability": 1 / 3}
    wire = _campaign_train_config_wire(source)

    assert wire["learning_rate"] == "0.0003"
    assert wire["heuristic_opponent_probability"] == "0.3333333333333333"
    assert _campaign_train_config_native(wire) == source


@pytest.mark.parametrize(("run_uuid", "root_seed", "run_id", "extra", "match"), [
    ("ABCDEFAB-1234-4234-8234-123456789abc", 2_000_000, None, {}, "canonical lowercase"),
    ("23456789-1234-4234-8234-123456789abc", True, None, {}, "Invalid preregistered"),
    ("23456789-1234-4234-8234-123456789abc", 1_000_000, None, {}, "outside the frozen train partition"),
    ("23456789-1234-4234-8234-123456789abc", 2_000_000, "different-run-id", {}, "must equal"),
    ("23456789-1234-4234-8234-123456789abc", 2_000_000, None,
     {"g008_principal_benchmark": {"run_ids": {}}}, "cannot be combined"),
    ("23456789-1234-4234-8234-123456789abc", 2_000_000, None,
     {"g008_train_schedule_extra": {}}, "unsupported fields"),
])
def test_load_run_settings_rejects_invalid_g008_train_schedule(
        tmp_path, run_uuid, root_seed, run_id, extra, match):
    config = tmp_path / "invalid-g008-train-config.json"
    document = {
        "run_id": run_id or run_uuid,
        "train_config": g008_principal_train_config(),
        "g008_train_schedule": {"run_uuid": run_uuid, "root_seed": root_seed},
        **extra,
    }
    config.write_text(json.dumps(document))
    with pytest.raises(KaggleError, match=match):
        _load_run_settings(config)


def test_load_run_settings_rejects_nonfrozen_g008_model_seed(tmp_path):
    import uuid

    run_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "g008-train-wrong-model-seed"))
    config = tmp_path / "wrong-model-seed.json"
    config.write_text(json.dumps({
        "run_id": run_uuid,
        "train_config": {**g008_principal_train_config(), "seed": 1235},
        "g008_train_schedule": {"run_uuid": run_uuid, "root_seed": 2_000_000},
    }))
    with pytest.raises(KaggleError, match="frozen principal config"):
        _load_run_settings(config)


@pytest.mark.parametrize(("field", "value"), [
    ("arena_count", 127),
    ("rollout_steps", 31),
    ("total_updates", 2),
    ("learning_rate", 0.0002),
])
def test_load_run_settings_rejects_g008_train_config_drift(tmp_path, field, value):
    import uuid

    run_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"g008-train-config-drift:{field}"))
    train_config = {**g008_principal_train_config(), field: value}
    config = tmp_path / f"g008-config-drift-{field}.json"
    config.write_text(json.dumps({
        "run_id": run_uuid,
        "train_config": train_config,
        "g008_train_schedule": {"run_uuid": run_uuid, "root_seed": 2_000_000},
    }))
    with pytest.raises(KaggleError, match="differs from the frozen principal config"):
        _load_run_settings(config)


def test_load_run_settings_normalizes_frozen_cuda_config_for_local_validation(tmp_path):
    import uuid

    run_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "g008-train-frozen-cuda-config"))
    train_config = {**g008_principal_train_config(), "device": "cuda"}
    config = tmp_path / "g008-frozen-cuda-config.json"
    config.write_text(json.dumps({
        "run_id": run_uuid,
        "train_config": train_config,
        "g008_train_schedule": {"run_uuid": run_uuid, "root_seed": 2_000_000},
    }))
    settings = _load_run_settings(config)
    assert settings["train_config"]["device"] == "cuda"


@pytest.mark.parametrize("benchmark", [
    {"configs": [{"arena_count": 4}], "updates": 0},
    {"configs": [{"arena_count": 4}], "updates": 11},
    {"configs": [{"arena_count": 4, "device": "cpu"}], "updates": 1},
    {"configs": [{"arena_count": 4}] * 6, "updates": 1},
    {"configs": [{"arena_count": 4}], "updates": True},
])
def test_prepare_rejects_invalid_or_unbounded_throughput_benchmark(tmp_path, benchmark):
    config = tmp_path / "benchmark-config.json"
    config.write_text(json.dumps({"run_id": "run-bench", "train_config": {"device": "cuda"},
                                  "throughput_benchmark": benchmark}))
    with pytest.raises(KaggleError):
        prepare("me/light-cycles", "source-bench", tmp_path / "package", config)
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize(("device", "cuda_available"), [("cpu", True), ("auto", False), ("cuda", False)])
def test_kaggle_runner_fails_fast_without_required_cuda(tmp_path, monkeypatch, device, cuda_available):
    config = tmp_path / "run-config.json"
    config.write_text(json.dumps({"run_id": "fail-fast", "train_config": {"device": device}}))
    package = prepare("me/light-cycles", "source-r5", tmp_path / f"package-{device}-{cuda_available}", config)
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "torch-test"
    fake_torch.version = types.SimpleNamespace(cuda="cuda-test")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    pip_calls = []
    monkeypatch.setattr("subprocess.check_call", lambda *args, **kwargs: pip_calls.append(args))
    namespace = {"__name__": "runner_test", "__file__": str(package / "run_kaggle.py")}
    source = (package / "run_kaggle.py").read_text()
    exec(compile(source, str(package / "run_kaggle.py"), "exec"), namespace)
    with pytest.raises(RuntimeError, match="requires .*CUDA"):
        namespace["main"]()
    assert pip_calls == []


def test_kaggle_runner_resolves_auto_to_cuda_and_trains_on_gpu(tmp_path, monkeypatch):
    package = prepare("me/light-cycles", "source-r5", tmp_path / "package-auto")
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "torch-test"
    fake_torch.version = types.SimpleNamespace(cuda="cuda-test")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                            get_device_name=lambda index: "Tesla T4", set_device=lambda index: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    output_root = tmp_path / "kaggle-output"
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_root))
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
    trained = []
    training_module = types.ModuleType("training.ppo.train")
    training_module.TrainConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    training_module.train = lambda config, output, export_onnx: trained.append((config.device, str(output), export_onnx))
    monkeypatch.setitem(sys.modules, "training.ppo.train", training_module)
    namespace = {"__name__": "runner_test", "__file__": str(package / "run_kaggle.py")}
    source = (package / "run_kaggle.py").read_text()
    exec(compile(source, str(package / "run_kaggle.py"), "exec"), namespace)
    namespace["main"]()
    assert trained == [("cuda", str(output_root / "kaggle-run"), True)]


def test_kaggle_runner_passes_frozen_g008_train_schedule_and_uuid_to_train(tmp_path, monkeypatch):
    import uuid
    from training.eval.provenance import build_training_run_schedule

    run_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "g008-train-runner"))
    run_config = tmp_path / "g008-train-run-config.json"
    run_config.write_text(json.dumps({
        "run_id": run_uuid,
        "train_config": g008_principal_train_config(),
        "g008_train_schedule": {"run_uuid": run_uuid, "root_seed": 2_000_000},
    }))
    package = prepare("me/circuit-ai", "g008-train-runner", tmp_path / "package", run_config)
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "torch-test"
    fake_torch.version = types.SimpleNamespace(cuda="cuda-test")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                            get_device_name=lambda index: "Tesla T4", set_device=lambda index: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(tmp_path / "kaggle-output"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
    calls = []
    training_module = types.ModuleType("training.ppo.train")
    training_module.TrainConfig = TrainConfig
    training_module.train = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "training.ppo.train", training_module)

    namespace = {"__name__": "runner_test", "__file__": str(package / "run_kaggle.py")}
    source = (package / "run_kaggle.py").read_text()
    exec(compile(source, str(package / "run_kaggle.py"), "exec"), namespace)
    namespace["main"]()

    assert len(calls) == 1
    args, kwargs = calls[0]
    config, output = args[:2]
    assert isinstance(config, TrainConfig) and config.device == "cuda"
    assert output == tmp_path / "kaggle-output" / run_uuid
    assert kwargs == {"export_onnx": True, "run_evaluation": False,
                      "g008_schedule": build_training_run_schedule(
                          run_id=run_uuid, root_seed=2_000_000, arena_count=128,
                          rollout_steps=32, total_updates=3),
                      "run_id": run_uuid}


def test_prepared_kaggle_runner_dispatches_registered_g008_final_evaluation(tmp_path, monkeypatch):
    import importlib.util
    import uuid
    from training.eval.g008_final_roster import EXPECTED_FINAL_ROSTER
    from training.eval.provenance import seed_plan_sha256
    from training.ppo.protocol import file_sha256

    run_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "g008-final-runner-template"))
    roster = EXPECTED_FINAL_ROSTER
    run_config = tmp_path / "g008-final-config.json"
    run_config.write_text(json.dumps({
        "run_id": run_uuid,
        "train_config": {"device": "auto"},
        "g008_evaluation": {
            "campaign_id": "g008-competitive-20261021-v2",
            "registration_kind": "final", "mode": "final", "run_uuid": run_uuid,
            "candidate_dataset_source": "pascalcoimbra/g008-frozen-candidate",
            "candidate_checkpoint_path": "/kaggle/input/g008-frozen-candidate/candidate.pt",
            "roster_sha256": roster["roster_sha256"],
            "opponent_code_sha256": roster["opponent_code_sha256"],
            "game_count": 512, "seat_order": [0, 1],
            "seed_plan_sha256": seed_plan_sha256(), "opponent_id": "open-space",
        },
    }))
    package = prepare("me/circuit-ai", "g008-final-runner-template", tmp_path / "package-final", run_config)
    assert (package / "training/configs/g008-split-seed-plan.json").is_file()
    assert (package / "training/eval/provenance.py").is_file()
    assert (package / "training/eval/registry.py").is_file()
    assert (package / "training/eval/lc_eval.py").is_file()
    assert (package / "training/eval/g008_final_roster.py").is_file()
    final_archive = _embedded_package_files(package)
    assert {
        "training/configs/g008-split-seed-plan.json",
        "training/eval/provenance.py", "training/eval/registry.py",
        "training/eval/lc_eval.py", "training/eval/g008_final_roster.py",
        "training/eval/g008_final_opponent.py",
    } <= final_archive.keys()
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == ["pascalcoimbra/g008-frozen-candidate"]
    assert (package / "schemas/lc-replay-1.schema.json").is_file()
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "torch-test"
    fake_torch.version = types.SimpleNamespace(cuda="cuda-test")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                            get_device_name=lambda index: "Tesla T4", set_device=lambda index: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    output_root = tmp_path / "kaggle-output"
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_root))
    calls = []
    evaluation_module = types.ModuleType("training.eval.g008_validation")
    evaluation_module.evaluate_registered = lambda settings, output, **kwargs: (
        calls.append((settings, output, kwargs)) or {"game_count": 512})
    monkeypatch.setitem(sys.modules, "training.eval.g008_validation", evaluation_module)
    namespace = {"__name__": "runner_test", "__file__": str(package / "run_kaggle.py")}
    source = (package / "run_kaggle.py").read_text()
    exec(compile(source, str(package / "run_kaggle.py"), "exec"), namespace)
    namespace["main"]()

    assert calls[0][0]["mode"] == "final"
    assert calls[0][0]["device"] == "cuda"
    assert calls[0][1] == output_root / run_uuid
    assert calls[0][2]["config_file_sha256"] == file_sha256(package / "run-config.json")
    from training.ppo.protocol import digest_json
    frozen_settings = json.loads((package / "run-config.json").read_text())["g008_evaluation"]
    assert calls[0][2]["settings_sha256"] == digest_json(frozen_settings, "__no_digest_field__")
    assert calls[0][2]["runtime_evidence"]["visible_cuda_devices"] == ["Tesla T4", "Tesla T4"]
    assert calls[0][2]["runtime_evidence"]["visible_cuda_device_count"] == 2
    assert calls[0][2]["runtime_evidence"]["training_device_index"] == 0
    assert calls[0][2]["runtime_evidence"]["training_cuda_device_count"] == 1


def test_prepare_embeds_frozen_g007_opponent_for_registered_validation(tmp_path):
    import uuid
    from training.eval.g008_validation import _EXPECTED_ROSTERS
    from training.eval.provenance import seed_plan_sha256
    from training.ppo.protocol import file_sha256

    run_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "g008-validation-package-assets"))
    opponent_source = Path(__file__).parents[2] / "artifacts/circuit-ai/runs/lc-circuit-ai-selfplay-300-viewer-ready/checkpoints/step-300.pt"
    roster = _EXPECTED_ROSTERS["validation"]
    run_config = tmp_path / "g008-validation-config.json"
    run_config.write_text(json.dumps({
        "run_id": run_uuid, "train_config": {"device": "auto"},
        "g008_evaluation": {
            "campaign_id": "g008-competitive-20261021-v2",
            "registration_kind": "validation", "mode": "validation", "run_uuid": run_uuid,
            "candidate_dataset_source": "pascalcoimbra/g008-frozen-candidate",
            "candidate_checkpoint_path": "/kaggle/input/g008-frozen-candidate/candidate.pt",
            "roster_sha256": roster["roster_sha256"],
            "opponent_code_sha256": roster["opponent_code_sha256"],
            "game_count": 512, "seat_order": [0, 1], "seed_plan_sha256": seed_plan_sha256(),
            "opponent_id": "g007-step-300",
            "opponent_asset_path": "training/eval/assets/g007-step-300.pt",
            "opponent_checkpoint_sha256": file_sha256(opponent_source),
        },
    }))
    package = prepare("me/circuit-ai", "g008-validation-package-assets", tmp_path / "package-validation", run_config)
    embedded_opponent = package / "training/eval/assets/g007-step-300.pt"
    assert file_sha256(embedded_opponent) == file_sha256(opponent_source)
    _assert_nonfinal_campaign_archive_has_no_final_holdout(package)
    assert json.loads((package / "kernel-metadata.json").read_text())["dataset_sources"] == [
        "pascalcoimbra/g008-frozen-candidate"]


def test_kaggle_runner_runs_bounded_cuda_sweep_and_hashes_report_in_manifest(tmp_path, monkeypatch):
    config_path = tmp_path / "run-config.json"
    config_path.write_text(json.dumps({"run_id": "run-bench", "train_config": {"device": "auto"},
                                       "throughput_benchmark": {"configs": [{"arena_count": 4}], "updates": 1}}))
    package = prepare("me/light-cycles", "source-bench", tmp_path / "package", config_path)
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "torch-test"
    fake_torch.version = types.SimpleNamespace(cuda="cuda-test")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                            get_device_name=lambda index: "Tesla T4", set_device=lambda index: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    output_root = tmp_path / "kaggle-output"
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_root))
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
    training_module = types.ModuleType("training.ppo.train")
    training_module.TrainConfig = TrainConfig
    def fake_train(config, output, export_onnx):
        assert config.device == "cuda" and export_onnx is True
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"files": [], "benchmark_records": []}
        manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
        (output / "manifest.json").write_text(json.dumps(manifest))
    training_module.train = fake_train
    monkeypatch.setitem(sys.modules, "training.ppo.train", training_module)
    benchmark_calls = []
    benchmark_module = types.ModuleType("training.ppo.benchmark")
    from training.ppo.benchmark import build_candidates as real_build_candidates
    benchmark_module.build_candidates = real_build_candidates
    def fake_run_benchmark(candidates, output):
        benchmark_calls.append((candidates, output))
        assert all(candidate.device == "cuda" and candidate.total_updates == 1 for candidate in candidates)
        return {"format": "lc-throughput-benchmark/1", "selected_trial": 0,
                "selection_metric": "median_environment_steps_per_second (update-only)",
                "device_mode": "cuda", "execution_model": "single-device"}
    benchmark_module.run_benchmark = fake_run_benchmark
    monkeypatch.setitem(sys.modules, "training.ppo.benchmark", benchmark_module)

    namespace = {"__name__": "runner_test", "__file__": str(package / "run_kaggle.py")}
    source = (package / "run_kaggle.py").read_text()
    exec(compile(source, str(package / "run_kaggle.py"), "exec"), namespace)
    namespace["main"]()

    output = output_root / "run-bench"
    report_path = output / "benchmark-results.json"
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads(report_path.read_text())
    assert report["format"] == "lc-throughput-benchmark/1"
    assert file_sha256(report_path) == next(item["sha256"] for item in manifest["files"]
                                             if item["path"] == "benchmark-results.json")
    assert manifest["benchmark_records"][0]["report_file"] == "benchmark-results.json"
    assert manifest["manifest_sha256"] == digest_json(manifest, "manifest_sha256")
    assert len(benchmark_calls) == 1 and not benchmark_calls[0][1].exists()


def test_kaggle_runner_executes_g008_benchmark_only_and_records_cuda_evidence(tmp_path, monkeypatch):
    import uuid
    from training.eval.provenance import benchmark_run_keys

    run_ids = {key: str(uuid.uuid5(uuid.NAMESPACE_URL, f"g008-runner:{key}"))
               for key in benchmark_run_keys()}
    config_path = tmp_path / "run-config.json"
    config_path.write_text(json.dumps({"run_id": "g008-runner", "train_config": {"device": "auto"},
                                       "g008_principal_benchmark": {"run_ids": run_ids}}))
    package = prepare("me/circuit-ai", "g008-runner", tmp_path / "package", config_path)
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "torch-test"
    fake_torch.version = types.SimpleNamespace(cuda="cuda-test")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                            get_device_name=lambda index: "Tesla T4", set_device=lambda index: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    output_root = tmp_path / "kaggle-output"
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_root))
    training_module = types.ModuleType("training.ppo.train")
    training_module.TrainConfig = TrainConfig
    training_module.train = lambda *args, **kwargs: pytest.fail("G008 benchmark must not run ordinary training")
    monkeypatch.setitem(sys.modules, "training.ppo.train", training_module)
    called = []
    g008_module = types.ModuleType("training.ppo.g008_benchmark")
    def fake_benchmark(output, run_ids):
        called.append((str(output), run_ids))
        Path(output).mkdir(parents=True, exist_ok=True)
        report = {"format": "test-report", "device": {"device": "cuda"}}
        (Path(output) / "g008-principal-benchmark.json").write_text(json.dumps(report))
        return report
    g008_module.run_principal_benchmark = fake_benchmark
    monkeypatch.setitem(sys.modules, "training.ppo.g008_benchmark", g008_module)
    namespace = {"__name__": "runner_test", "__file__": str(package / "run_kaggle.py")}
    source = (package / "run_kaggle.py").read_text()
    exec(compile(source, str(package / "run_kaggle.py"), "exec"), namespace)
    namespace["main"]()
    output = output_root / "g008-runner"
    evidence = json.loads((output / "kaggle-execution-evidence.json").read_text())
    assert called == [(str(output), run_ids)]
    package_source_manifest = json.loads((package / "source-manifest.json").read_text())
    assert evidence["source_manifest_sha256"] == package_source_manifest["source_manifest_sha256"]
    assert evidence["visible_cuda_devices"] == ["Tesla T4", "Tesla T4"]
    assert evidence["visible_cuda_device_count"] == 2
    assert evidence["training_device_index"] == 0
    assert evidence["training_cuda_device_count"] == 1
    assert evidence["benchmark_report"]["format"] == "test-report"
    assert evidence["benchmark_report_sha256"] == file_sha256(output / "g008-principal-benchmark.json")


def test_prepare_output_passes_submit_preflight_and_refuses_overwrite(tmp_path):
    source = prepare("me/light-cycles", "local-r9", tmp_path / "kernel")
    assert "DRY RUN" in submit("me/light-cycles", "local-r9", source, confirm=False, dry_run=True)
    with pytest.raises(KaggleError, match="already exists"):
        prepare("me/light-cycles", "local-r9", source)
    with pytest.raises(KaggleError, match="already exists"):
        prepare("me/other", "local-r9", tmp_path / "kernel")
