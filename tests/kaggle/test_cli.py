import json
import importlib.util
import sys
import types
from dataclasses import fields
from pathlib import Path

import pytest

from tools.kaggle.cli import KaggleError, _load_run_settings, _sdk_metadata, download, poll_status, prepare, submit
from training.arena.engine import ACTION_ORDER, initialize_episode, step
from training.ppo.protocol import digest_json, file_sha256
from training.ppo.train import TrainConfig


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


def make_g008_benchmark(root: Path, run_id="g008-run") -> None:
    import uuid
    from training.eval.provenance import (benchmark_run_keys, schedule_spec_digest,
                                          seed_plan_sha256)
    from training.ppo.g008_benchmark import CONFIG_PATH, load_principal_config, prepare_benchmark_schedules

    root.mkdir(parents=True, exist_ok=True)
    run_keys = benchmark_run_keys()
    run_ids = {key: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{key}")) for key in run_keys}
    schedules = prepare_benchmark_schedules(run_ids)
    principal_config, config_document = load_principal_config()
    roster = [{"run_key": key, "run_id": run_ids[key],
               "expanded_schedule_digest": schedules[key]["expanded_schedule_digest"]}
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
                        "expanded_schedule_digest": schedules[run_key]["expanded_schedule_digest"],
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
                "expanded_schedule_digest": schedules[f"warmup-python-oracle-incumbent-{index:02d}"]["expanded_schedule_digest"],
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
                "visible_cuda_devices": ["Tesla T4"],
                "benchmark_report_sha256": file_sha256(report_path), "benchmark_report": report}
    (root / "kaggle-execution-evidence.json").write_text(json.dumps(evidence))


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


@pytest.mark.parametrize("mutation", ["missing-version", "empty-version", "missing-gpus", "multiple-gpus", "gpu-mismatch"])
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
    elif mutation == "multiple-gpus":
        evidence["visible_cuda_devices"] = ["Tesla T4", "Tesla T4"]
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


def make_recovery_registry(tmp_path, *, timeout=30, kind="benchmark"):
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
        "reservation_metadata": {"campaign_gpu_seconds_before": 8424, "campaign_output_bytes_before": 0,
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

    def change_on_second_hash(path):
        calls.append(path)
        return original_hash(path) if len(calls) == 1 else "0" * 64

    monkeypatch.setattr(cli, "package_tree_sha256", change_on_second_hash)
    with pytest.raises(KaggleError, match="changed after preflight"):
        cli.run_recovery_workload(
            registry, registration["event_index"], source,
            attempt_id="attempt-package-race", implementation_variant_id=variant,
            no_external_writer_confirmed=True, api=api, output_bytes=lambda kernel, version: 0,
        )
    assert len(calls) == 2
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
    assert projected_budget["campaign_gpu_seconds_projected"] == 8424 + 30
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
    assert projected_budget["campaign_gpu_seconds_projected"] == 8424 + 30


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
                "visible_cuda_devices": ["Tesla T4"]}
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


def test_recovery_artifact_binds_valid_v3_report_and_selection_to_registry(tmp_path, monkeypatch):
    import hashlib
    import statistics
    import uuid

    from tools.kaggle.cli import register_recovery_benchmark_artifact
    from training.eval.provenance import benchmark_run_keys, schedule_spec_digest, seed_plan_sha256
    from training.ppo.g008_benchmark import (CONFIG_PATH, CANDIDATE_SOURCE_COMMIT,
                                             INCUMBENT_SOURCE_COMMIT, implementation_hashes,
                                             load_principal_config, prepare_benchmark_schedules)

    artifact_root = tmp_path / "downloaded"
    artifact_root.mkdir()
    run_keys = benchmark_run_keys()
    run_ids = {key: str(uuid.uuid5(uuid.NAMESPACE_URL, f"fake-valid-v3:{key}")) for key in run_keys}
    schedules = prepare_benchmark_schedules(run_ids)
    config, config_doc = load_principal_config()
    implementations = {role: implementation_hashes(role) for role in ("incumbent", "candidate")}
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
                "visible_cuda_devices": ["Tesla T4"], "benchmark_report": report,
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
    envelope_path = artifact_root / "g008-recovery-benchmark-artifact.json"
    result = register_recovery_benchmark_artifact(registry, attempt_event, artifact_root, envelope_path)
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

    report["parity"]["pairs"]["pair-00"]["candidate"]["sha256"] = "0" * 64
    from training.ppo import g008_benchmark
    monkeypatch.setattr(g008_benchmark, "prepare_benchmark_schedules", lambda _: schedules)
    from tools.kaggle.cli import validate_principal_benchmark_artifacts

    def refresh_report_evidence():
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        evidence["benchmark_report"] = report
        evidence["benchmark_report_sha256"] = file_sha256(report_path)
        (artifact_root / "kaggle-execution-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

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
        assert (embedded_root / "training/ppo/train.py").is_file()
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
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                            get_device_name=lambda index: "Tesla T4")
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
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                            get_device_name=lambda index: "Tesla T4")
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


def test_kaggle_runner_runs_bounded_cuda_sweep_and_hashes_report_in_manifest(tmp_path, monkeypatch):
    config_path = tmp_path / "run-config.json"
    config_path.write_text(json.dumps({"run_id": "run-bench", "train_config": {"device": "auto"},
                                       "throughput_benchmark": {"configs": [{"arena_count": 4}], "updates": 1}}))
    package = prepare("me/light-cycles", "source-bench", tmp_path / "package", config_path)
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "torch-test"
    fake_torch.version = types.SimpleNamespace(cuda="cuda-test")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                            get_device_name=lambda index: "Tesla T4")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    output_root = tmp_path / "kaggle-output"
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_root))
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
    training_module = types.ModuleType("training.ppo.train")
    training_module.TrainConfig = TrainConfig
    def fake_train(config, output, export_onnx):
        assert config.device == "cuda" and export_onnx is True
        output.mkdir(parents=True)
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
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                            get_device_name=lambda index: "Tesla T4")
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
        Path(output).mkdir(parents=True)
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
    assert evidence["visible_cuda_devices"] == ["Tesla T4"]
    assert evidence["benchmark_report"]["format"] == "test-report"
    assert evidence["benchmark_report_sha256"] == file_sha256(output / "g008-principal-benchmark.json")


def test_prepare_output_passes_submit_preflight_and_refuses_overwrite(tmp_path):
    source = prepare("me/light-cycles", "local-r9", tmp_path / "kernel")
    assert "DRY RUN" in submit("me/light-cycles", "local-r9", source, confirm=False, dry_run=True)
    with pytest.raises(KaggleError, match="already exists"):
        prepare("me/light-cycles", "local-r9", source)
    with pytest.raises(KaggleError, match="already exists"):
        prepare("me/other", "local-r9", tmp_path / "kernel")
