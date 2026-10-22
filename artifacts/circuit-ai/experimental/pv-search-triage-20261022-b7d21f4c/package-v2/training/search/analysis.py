"""Cluster-aware summaries for verified simultaneous-search tournament outputs."""

from __future__ import annotations

import json
import argparse
import random
import statistics
from math import log, sqrt
from pathlib import Path
from typing import Any


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))]


def _bootstrap_interval(cluster_scores: list[float], *, samples: int, seed: int) -> list[float]:
    if not cluster_scores or samples < 1:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(rng.choices(cluster_scores, k=len(cluster_scores))))
    means.sort()
    return [means[int((samples - 1) * 0.025)], means[int((samples - 1) * 0.975)]]


def analyze_output(output_dir: str | Path, *, bootstrap_samples: int = 10_000,
                   seed: int = 0) -> dict[str, Any]:
    """Summarize outcomes with seed-cluster intervals and per-seat timing."""
    root = Path(output_dir)
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    games = [json.loads(line) for line in (root / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for game in games:
        grouped.setdefault(game["matchup_id"], []).append(game)

    results = []
    for index, matchup in enumerate(report["matchups"]):
        records = grouped.get(matchup["matchup_id"], [])
        wins = sum(game["outcome"] == "left_win" for game in records)
        draws = sum(game["outcome"] == "draw" for game in records)
        losses = len(records) - wins - draws
        by_seed: dict[int, list[dict[str, Any]]] = {}
        for game in records:
            by_seed.setdefault(game["seed"], []).append(game)
        clusters = [
            (sum(game["outcome"] == "left_win" for game in pair) +
             0.5 * sum(game["outcome"] == "draw" for game in pair)) / len(pair)
            for pair in by_seed.values() if pair
        ]
        cluster_mean = statistics.fmean(clusters) if clusters else 0.0
        hoeffding_radius = sqrt(log(2 / 0.05) / (2 * len(clusters))) if clusters else 1.0
        seat_results = {}
        for seat in (0, 1):
            seat_games = [game for game in records if game["target_seat"] == seat]
            seat_results[str(seat)] = {
                "games": len(seat_games),
                "wins": sum(game["outcome"] == "left_win" for game in seat_games),
                "draws": sum(game["outcome"] == "draw" for game in seat_games),
                "losses": sum(game["outcome"] == "right_win" for game in seat_games),
            }
        left_ns = [value for game in records for value in game["left_stats"]["decision_ns"]]
        right_ns = [value for game in records for value in game["right_stats"]["decision_ns"]]
        def causes_for(side: str) -> dict[str, int]:
            totals: dict[str, int] = {}
            for game in records:
                for cause, count in game[f"{side}_stats"].get("death_causes", {}).items():
                    totals[cause] = totals.get(cause, 0) + count
            return dict(sorted(totals.items()))

        collision_metrics_recorded = bool(records) and all(
            "death_causes" in game["left_stats"]
            and "avoidable_deaths" in game["left_stats"]
            and "death_causes" in game["right_stats"]
            and "avoidable_deaths" in game["right_stats"]
            and "first_separation_tick" in game
            and "ticks_after_separation" in game for game in records)

        results.append({
            "matchup_id": matchup["matchup_id"],
            "left_agent": matchup["left_agent"],
            "right_agent": matchup["right_agent"],
            "games": len(records), "seed_clusters": len(clusters),
            "wins": wins, "draws": draws, "losses": losses,
            "observed_win_rate": wins / max(1, len(records)),
            "score_rate_draw_half": (wins + 0.5 * draws) / max(1, len(records)),
            "seed_cluster_score_ci95_hoeffding": [
                max(0.0, cluster_mean - hoeffding_radius),
                min(1.0, cluster_mean + hoeffding_radius),
            ],
            "seed_cluster_score_ci95": _bootstrap_interval(
                clusters, samples=bootstrap_samples, seed=seed + index),
            "seat_results": seat_results,
            "left_decision_ns_mean": int(statistics.fmean(left_ns)) if left_ns else 0,
            "left_decision_ns_p95": _percentile(left_ns, 0.95),
            "right_decision_ns_mean": int(statistics.fmean(right_ns)) if right_ns else 0,
            "right_decision_ns_p95": _percentile(right_ns, 0.95),
            "process_cpu_seconds": round(
                sum(game["process_cpu_ns"] for game in records) / 1e9, 3),
            "collision_metrics_recorded": collision_metrics_recorded,
            "left_avoidable_deaths": (sum(game["left_stats"]["avoidable_deaths"] for game in records)
                                       if collision_metrics_recorded else None),
            "right_avoidable_deaths": (sum(game["right_stats"]["avoidable_deaths"] for game in records)
                                        if collision_metrics_recorded else None),
            "left_death_causes": causes_for("left") if collision_metrics_recorded else None,
            "right_death_causes": causes_for("right") if collision_metrics_recorded else None,
            "games_with_territorial_separation": (
                sum(game["first_separation_tick"] is not None for game in records)
                if collision_metrics_recorded else None),
            "mean_ticks_after_territorial_separation": (
                statistics.fmean(game["ticks_after_separation"] for game in records)
                if collision_metrics_recorded else None),
        })
    return {
        "format": "circuit-ai-simultaneous-search-analysis/1",
        "experiment_id": report["experiment_id"], "run_id": report["run_id"],
        "bootstrap": {"unit": "seed with both seat assignments", "samples": bootstrap_samples,
                      "seed": seed},
        "matchups": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    result = analyze_output(args.output_dir, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            parser.error(f"refusing to overwrite analysis output: {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
