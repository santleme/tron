import json
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from tools.kaggle.cli import KaggleError, download, poll_status, prepare, submit
from training.arena.engine import ACTION_ORDER, initialize_episode, step
from training.ppo.protocol import digest_json, file_sha256
from training.ppo.train import TrainConfig


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
    root.mkdir(parents=True, exist_ok=True)
    device = {"device": "cuda", "cuda_available": True, "cuda_name": "Tesla T4",
              "optimizer_updates": 2, "cuda_optimizer_updates": 2}
    repeats = []
    for index in range(5):
        metrics = {"device": device, "training": {"updates": []}}
        path = root / "measured" / f"repeat-{index:02d}" / "training-metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics))
        repeats.append({"repeat": index, "optimizer_updates": 2, "cuda_optimizer_updates": 2,
                        "training_metrics_sha256": file_sha256(path)})
    for index in range(2):
        path = root / "warmups" / f"warmup-{index:02d}" / "training-metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"device": device, "training": {"updates": []}}))
    report = {"format": "circuit-ai-g008-principal-benchmark/1", "measured_repeats": 5,
              "device": device, "repeats": repeats}
    report_path = root / "g008-principal-benchmark.json"
    report_path.write_text(json.dumps(report))
    evidence = {"format": "circuit-ai-kaggle-g008-benchmark-evidence/1", "run_id": run_id,
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
