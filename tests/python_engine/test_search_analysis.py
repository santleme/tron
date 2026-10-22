from training.search.analysis import analyze_output
from training.search.tournament import run_experiment


def test_analysis_reports_seed_clusters_seats_and_deterministic_intervals(tmp_path):
    config = {
        "experiment_id": "simsearch-analysis-test",
        "run_id": "simsearch-analysis-test",
        "width": 4,
        "height": 4,
        "seeds": [19, 23],
        "planners": ["fixed-depth", "ducb1-tuned", "regret-matching"],
        "heuristics": ["heuristic:cruise", "heuristic:local-mobility", "heuristic:interceptor"],
        "edge_budget": 8,
        "max_depth": 1,
        "max_wall_seconds": 60,
    }
    output = tmp_path / "run"
    run_experiment(config, output, source_manifest_sha256="b" * 64)

    analysis = analyze_output(output, bootstrap_samples=50, seed=71)
    repeated = analyze_output(output, bootstrap_samples=50, seed=71)
    assert analysis == repeated
    assert len(analysis["matchups"]) == 12
    for matchup in analysis["matchups"]:
        assert matchup["games"] == 4
        assert matchup["seed_clusters"] == 2
        assert matchup["seat_results"]["0"]["games"] == 2
        assert matchup["seat_results"]["1"]["games"] == 2
        assert len(matchup["seed_cluster_score_ci95"]) == 2
        assert len(matchup["seed_cluster_score_ci95_hoeffding"]) == 2
