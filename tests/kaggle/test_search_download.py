import json

import pytest

from tools.kaggle.cli import KaggleError, validate_downloaded_output
from training.search.tournament import run_experiment


def _config():
    return {
        "experiment_id": "simsearch-download-test",
        "run_id": "simsearch-download-test",
        "width": 4,
        "height": 4,
        "seeds": [17],
        "planners": ["fixed-depth", "ducb1-tuned", "regret-matching"],
        "heuristics": ["heuristic:cruise", "heuristic:local-mobility", "heuristic:interceptor"],
        "edge_budget": 8,
        "max_depth": 1,
        "max_wall_seconds": 60,
    }


def test_search_output_download_validation_checks_hashes_and_summaries(tmp_path):
    config = _config()
    output = tmp_path / "run"
    run_experiment(config, output, source_manifest_sha256="a" * 64)
    validate_downloaded_output(output, config["run_id"])

    games = output / "games.jsonl"
    first = json.loads(games.read_text().splitlines()[0])
    first["outcome"] = "draw" if first["outcome"] != "draw" else "left_win"
    lines = games.read_text().splitlines()
    lines[0] = json.dumps(first)
    games.write_text("\n".join(lines) + "\n")
    with pytest.raises(KaggleError, match="hash mismatch"):
        validate_downloaded_output(output, config["run_id"])


def test_search_output_download_accepts_selected_planner_only_matchups(tmp_path):
    config = _config()
    config["matchups"] = [
        "fixed-depth-vs-ducb1-tuned",
        "fixed-depth-vs-regret-matching",
        "ducb1-tuned-vs-regret-matching",
    ]
    output = tmp_path / "selected"
    run_experiment(config, output, source_manifest_sha256="c" * 64)
    validate_downloaded_output(output, config["run_id"])
    report = json.loads((output / "report.json").read_text())
    assert len(report["matchups"]) == 3
