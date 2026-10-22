import json
import hashlib
from types import SimpleNamespace

import pytest

from training.search.tournament import run_experiment, validate_config
from training.search import frozen_opponent
from tools.kaggle.cli import KaggleError, validate_search_output_artifacts


def _config(tmp_path, *, seed=7):
    return {
        "experiment_id": "simsearch-test",
        "run_id": f"simsearch-test-{seed}",
        "width": 4,
        "height": 4,
        "seeds": [seed],
        "planners": ["fixed-depth", "ducb1-tuned", "regret-matching"],
        "heuristics": [
            "heuristic:cruise", "heuristic:local-mobility", "heuristic:interceptor",
        ],
        "edge_budget": 8,
        "max_depth": 1,
        "max_wall_seconds": 60,
        "workers": 1,
    }


def test_experiment_config_rejects_protected_opponents_and_accepts_protocol_budgets(tmp_path):
    config = _config(tmp_path)
    validate_config(config)
    config["heuristics"].append("open-space")
    with pytest.raises(ValueError, match="heuristic roster"):
        validate_config(config)
    config = _config(tmp_path)
    config["edge_budget"] = 100_000
    validate_config(config)
    config = _config(tmp_path)
    config["max_depth"] = 255
    config["edge_budget"] = 100_000
    validate_config(config)
    config["max_depth"] = 256
    with pytest.raises(ValueError, match="bounded config"):
        validate_config(config)


def test_experiment_config_has_no_artificial_seed_count_cap(tmp_path):
    config = _config(tmp_path)
    config["seeds"] = list(range(1, 1001))
    validate_config(config)


def test_experiment_config_accepts_bounded_per_planner_budgets(tmp_path):
    config = _config(tmp_path)
    config["agent_edge_budgets"] = {
        "fixed-depth": 16,
        "ducb1-tuned": 40,
        "regret-matching": 40,
    }
    validate_config(config)
    config["agent_edge_budgets"]["fixed-depth"] = config["max_depth"]
    with pytest.raises(ValueError, match="per-planner edge budgets"):
        validate_config(config)


def test_experiment_config_can_select_only_registered_development_matchups(tmp_path):
    config = _config(tmp_path)
    config["matchups"] = [
        "fixed-depth-vs-ducb1-tuned",
        "fixed-depth-vs-regret-matching",
        "ducb1-tuned-vs-regret-matching",
    ]
    validate_config(config)
    config["matchups"].append("fixed-depth-vs-open-space")
    with pytest.raises(ValueError, match="selected matchups"):
        validate_config(config)


def test_policy_value_checkpoint_is_pinned_and_limited_to_matching_board(tmp_path):
    config = _config(tmp_path)
    config.update({
        "width": 12,
        "height": 12,
        "policy_value_checkpoint": "pv-scale-20261022",
        "matchups": ["fixed-depth-vs-pv-scale-20261022"],
    })
    validate_config(config)
    config["policy_value_checkpoint"] = "some-local-checkpoint"
    with pytest.raises(ValueError, match="pinned experimental checkpoint"):
        validate_config(config)
    config["policy_value_checkpoint"] = "pv-scale-20261022"
    config["width"] = 10
    with pytest.raises(ValueError, match="12x12 board"):
        validate_config(config)


def test_native_fixed_depth_is_opt_in_and_rostered_only_for_pinned_policy_runs(tmp_path):
    config = _config(tmp_path)
    config.update({
        "width": 12,
        "height": 12,
        "policy_value_checkpoint": "pv-scale-20261022",
        "native_fixed_depth": True,
        "matchups": [
            "native-fixed-depth-vs-pv-scale-20261022",
            "native-fixed-depth-vs-fixed-depth",
        ],
    })
    validate_config(config)
    config["native_fixed_depth"] = False
    with pytest.raises(ValueError, match="selected matchups"):
        validate_config(config)


def test_experiment_config_accepts_requested_workers_and_runtime_clamps(tmp_path):
    config = _config(tmp_path)
    config["workers"] = 100
    validate_config(config)
    config["workers"] = 0
    with pytest.raises(ValueError, match="workers"):
        validate_config(config)


def test_spatial_checkpoint_is_hash_pinned_and_native_rust_is_rostered(tmp_path):
    config = _config(tmp_path)
    config.update({
        "width": 12,
        "height": 12,
        "spatial_policy_checkpoint": "dagger1-selected-20261021",
        "native_fixed_depth": True,
        "matchups": ["native-fixed-depth-vs-dagger1-selected-20261021"],
    })
    validate_config(config)
    config["spatial_policy_checkpoint"] = "unreviewed-checkpoint"
    with pytest.raises(ValueError, match="pinned experimental checkpoint"):
        validate_config(config)


def test_native_fixed_depth_supports_multiple_bounded_arena_sizes(tmp_path):
    config = _config(tmp_path)
    config.update({
        "width": 8,
        "height": 8,
        "native_fixed_depth": True,
        "matchups": ["native-fixed-depth-vs-heuristic:interceptor"],
    })
    validate_config(config)
    config["width"] = 17
    config["height"] = 16
    with pytest.raises(ValueError, match="256 cells"):
        validate_config(config)


def test_frozen_opponent_is_exactly_whitelisted_and_requires_bound_matchup(tmp_path):
    config = _config(tmp_path)
    config["frozen_opponent"] = "g007-step300"
    with pytest.raises(ValueError, match="protocol-permitted"):
        validate_config(config)
    config = _config(tmp_path)
    config.update({
        "width": 12, "height": 12, "frozen_opponent": "v15-step236",
        "matchups": ["fixed-depth-vs-v15-step236"],
    })
    validate_config(config)
    config["matchups"] = ["fixed-depth-vs-open-space"]
    with pytest.raises(ValueError, match="single explicit matchup"):
        validate_config(config)
    config["matchups"] = ["fixed-depth-vs-v15-step236"]
    config["width"] = 4
    with pytest.raises(ValueError, match="12x12 board"):
        validate_config(config)


def test_frozen_checkpoint_resolution_prefers_pinned_kaggle_input(tmp_path, monkeypatch):
    kaggle_checkpoint = tmp_path / "kaggle" / "step-236.pt"
    local_checkpoint = tmp_path / "assets" / "v15-step236.pt"
    kaggle_checkpoint.parent.mkdir()
    local_checkpoint.parent.mkdir()
    kaggle_checkpoint.write_bytes(b"pinned checkpoint")
    local_checkpoint.write_bytes(b"local staged checkpoint")
    monkeypatch.setattr(frozen_opponent, "KAGGLE_CHECKPOINT_PATH", kaggle_checkpoint)
    monkeypatch.setattr(frozen_opponent, "LOCAL_CHECKPOINT_PATH", local_checkpoint)

    assert frozen_opponent._checkpoint_path() == kaggle_checkpoint


def test_frozen_checkpoint_resolution_uses_local_stage_when_kaggle_input_is_absent(
        tmp_path, monkeypatch):
    kaggle_checkpoint = tmp_path / "missing-kaggle" / "step-236.pt"
    local_checkpoint = tmp_path / "assets" / "v15-step236.pt"
    local_checkpoint.parent.mkdir()
    local_checkpoint.write_bytes(b"local staged checkpoint")
    monkeypatch.setattr(frozen_opponent, "KAGGLE_CHECKPOINT_PATH", kaggle_checkpoint)
    monkeypatch.setattr(frozen_opponent, "LOCAL_CHECKPOINT_PATH", local_checkpoint)

    assert frozen_opponent._checkpoint_path() == local_checkpoint


def test_frozen_checkpoint_resolution_fails_closed_when_missing(tmp_path, monkeypatch):
    kaggle_checkpoint = tmp_path / "missing-kaggle" / "step-236.pt"
    local_checkpoint = tmp_path / "missing-assets" / "v15-step236.pt"
    monkeypatch.setattr(frozen_opponent, "KAGGLE_CHECKPOINT_PATH", kaggle_checkpoint)
    monkeypatch.setattr(frozen_opponent, "LOCAL_CHECKPOINT_PATH", local_checkpoint)

    with pytest.raises(FileNotFoundError, match="protocol-permitted v15-step236 checkpoint is missing"):
        frozen_opponent._checkpoint_path()


def test_tournament_pairs_both_seats_and_writes_provenance(tmp_path):
    config = _config(tmp_path)
    output = tmp_path / "result"
    report = run_experiment(config, output, source_manifest_sha256="a" * 64)

    assert report["status"] == "complete"
    assert report["frozen_opponents"] == []
    assert "torch_version" not in report["runtime"]
    assert len(report["matchups"]) == 12
    games = [json.loads(line) for line in (output / "games.jsonl").read_text().splitlines()]
    assert len(games) == 24
    for matchup in report["matchups"]:
        pair = [game for game in games if game["matchup_id"] == matchup["matchup_id"]]
        assert [game["target_seat"] for game in pair] == [0, 1]
        assert matchup["summary"]["games"] == 2
        assert matchup["summary"]["left_decisions"] > 0
        assert matchup["summary"]["process_cpu_ns"] > 0
        assert "left_avoidable_deaths" in matchup["summary"]
        assert "mean_ticks_after_separation" in matchup["summary"]

    manifest = json.loads((output / "manifest.json").read_text())
    for entry in manifest["files"]:
        actual = __import__("hashlib").sha256((output / entry["path"]).read_bytes()).hexdigest()
        assert entry["sha256"] == actual

    with pytest.raises(FileExistsError):
        run_experiment(config, output, source_manifest_sha256="a" * 64)


def test_frozen_tournament_report_records_checkpoint_provenance(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.update({
        "width": 12,
        "height": 12,
        "frozen_opponent": "v15-step236",
        "matchups": ["fixed-depth-vs-v15-step236"],
    })
    monkeypatch.setattr("training.search.tournament._selected_matchups", lambda *_: [])
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None)),
    )

    report = run_experiment(config, tmp_path / "frozen-result", source_manifest_sha256="b" * 64)

    assert report["frozen_opponents"] == [{
        "id": "v15-step236",
        "sha256": frozen_opponent.FROZEN_OPPONENTS["v15-step236"],
    }]
    assert report["runtime"]["torch_version"] == "test"


def test_download_validation_rejects_unapproved_frozen_opponent_config(tmp_path):
    config = _config(tmp_path)
    config["matchups"] = ["fixed-depth-vs-ducb1-tuned"]
    output = tmp_path / "tampered-report"
    run_experiment(config, output, source_manifest_sha256="a" * 64)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report["config"]["frozen_opponent"] = "g007-step300"
    report.pop("report_sha256", None)
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                   allow_nan=False).encode("utf-8")
    ).hexdigest()
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    with pytest.raises(KaggleError, match="invalid tournament config"):
        validate_search_output_artifacts(output, config["run_id"])


def test_process_pool_preserves_paired_game_order_and_decisions(tmp_path):
    config = _config(tmp_path, seed=17)
    config["seeds"] = [17, 23]
    config["matchups"] = ["fixed-depth-vs-ducb1-tuned"]
    serial = _config(tmp_path, seed=17)
    serial.update(config)
    serial["workers"] = 1
    parallel = _config(tmp_path, seed=17)
    parallel.update(config)
    parallel["workers"] = 2

    serial_dir, parallel_dir = tmp_path / "serial", tmp_path / "parallel"
    run_experiment(serial, serial_dir, source_manifest_sha256="a" * 64)
    parallel_report = run_experiment(parallel, parallel_dir, source_manifest_sha256="a" * 64)
    serial_games = [json.loads(line) for line in (serial_dir / "games.jsonl").read_text().splitlines()]
    parallel_games = [json.loads(line) for line in (parallel_dir / "games.jsonl").read_text().splitlines()]

    assert parallel_report["runtime"]["workers"] == 2
    assert [(game["seed"], game["target_seat"]) for game in parallel_games] == [
        (game["seed"], game["target_seat"]) for game in serial_games
    ]
    for serial_game, parallel_game in zip(serial_games, parallel_games):
        for field in ("outcome", "ticks", "trace", "first_separation_tick", "ticks_after_separation"):
            assert parallel_game[field] == serial_game[field]
