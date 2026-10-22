"""Reproducible exploratory tournaments for simultaneous-search agents.

This module intentionally imports only the canonical arena, local heuristics,
and the experimental search package. It does not load official split rosters,
checkpoints, or the final-only open-space policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from training.arena.engine import initialize_episode, step
from training.arena.heuristics import HEURISTIC_STYLES, heuristic_action
from .bitboard import from_arena_state, legal_actions, transition as compact_transition
from .planners import decoupled_ucb1_tuned, fixed_depth, regret_matching
from .territory import flood_region

PLANNERS = {
    "fixed-depth": fixed_depth,
    "ducb1-tuned": decoupled_ucb1_tuned,
    "regret-matching": regret_matching,
}

FROZEN_OPPONENTS = {
    "v15-step236": "a21a9e00d7e6eb9ad9366502d64b0eebe0cd2e46b191dcee45bf8394a5b5f450",
}

POLICY_VALUE_CHECKPOINTS = {
    "pv-scale-20261022": {
        "sha256": "200e5b74ca3ee61861694eaba0d36021dcfe5b9868f8c9d7d370bfd9517571d7",
        "dataset": "pascalcoimbra/circuit-ai-pv-policy-value-scale-20261022",
        "filename": "policy-value.pt",
    },
}

_POLICY_VALUE_MODEL = None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any], *, digest_field: str | None = None) -> str:
    payload = dict(value)
    if digest_field:
        payload.pop(digest_field, None)
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        payload[digest_field] = digest
    else:
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return digest


def _decision_seed(master_seed: int, matchup: str, seat: int, tick: int) -> int:
    preimage = f"circuit-ai-experimental-search-v1\0{master_seed}\0{matchup}\0{seat}\0{tick}".encode()
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big")


def _new_stats() -> dict[str, Any]:
    return {"decisions": 0, "elapsed_ns": 0, "transition_edges": 0,
            "decision_ns": [], "leaf_evaluations": 0, "simulations": 0,
            "completed_depth_sum": 0, "death_causes": {}, "avoidable_deaths": 0}


def _choose(agent_id: str, state, compact, seat: int, matchup_id: str, seed: int,
            edge_budget: int, max_depth: int) -> tuple[int | str, int, dict[str, Any]]:
    if agent_id in PLANNERS:
        started = time.perf_counter_ns()
        result = PLANNERS[agent_id](
            compact, player=seat, max_depth=max_depth,
            edge_budget=edge_budget,
            seed=_decision_seed(seed, matchup_id, seat, state.tick),
        )
        elapsed = time.perf_counter_ns() - started
        return (result.action if result.action >= 0 else "INVALID", elapsed, {
            "transition_edges": result.transition_edges,
            "leaf_evaluations": result.leaf_evaluations,
            "simulations": result.simulations,
            "completed_depth": result.completed_depth,
        })
    if agent_id in POLICY_VALUE_CHECKPOINTS:
        global _POLICY_VALUE_MODEL
        if _POLICY_VALUE_MODEL is None:
            _POLICY_VALUE_MODEL = _load_policy_value(agent_id)
        import numpy as np
        import torch
        from training.arena.engine import observe

        observation = torch.as_tensor(
            np.asarray(observe(state, seat), dtype=np.float32)[None, ...],
            dtype=torch.float32,
        )
        started = time.perf_counter_ns()
        with torch.inference_mode():
            logits, _value = _POLICY_VALUE_MODEL(observation)
            action = int(torch.argmax(logits[0]).item())
        elapsed = time.perf_counter_ns() - started
        return action, elapsed, {}
    style = agent_id.removeprefix("heuristic:")
    if style in HEURISTIC_STYLES:
        started = time.perf_counter_ns()
        action = heuristic_action(state, seat, style)
        elapsed = time.perf_counter_ns() - started
        return action, elapsed, {}
    if agent_id in FROZEN_OPPONENTS:
        from .frozen_opponent import choose_action as choose_frozen
        started = time.perf_counter_ns()
        action = choose_frozen(state, seat, agent_id)
        elapsed = time.perf_counter_ns() - started
        return action, elapsed, {}
    raise ValueError(f"unknown experimental tournament agent {agent_id!r}")


def _load_policy_value(agent_id: str):
    """Load the exact, pinned experimental model mounted by Kaggle."""
    import torch

    checkpoint_spec = POLICY_VALUE_CHECKPOINTS[agent_id]
    candidates = list(Path("/kaggle/input").rglob(checkpoint_spec["filename"]))
    checkpoint_path = next((path for path in candidates
                            if _file_sha256(path) == checkpoint_spec["sha256"]), None)
    if checkpoint_path is None:
        raise RuntimeError(f"pinned Policy-Value checkpoint {agent_id!r} is missing or has a bad SHA-256")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("model_architecture", {})
    if (checkpoint.get("format") != "lc-checkpoint/1"
            or architecture.get("name") != "mlp-hwc-flatten"
            or architecture.get("version") != "2"
            or architecture.get("config", {}).get("width") != 12
            or architecture.get("config", {}).get("height") != 12):
        raise RuntimeError("Policy-Value checkpoint metadata does not match the experimental v2 contract")
    from .policy_value import model_class
    model = model_class(torch)(12, 12, int(architecture["config"]["hidden_size"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    torch.set_num_threads(1)
    return model


def _agent_ids(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    planners = config.get("planners")
    heuristics = config.get("heuristics")
    if planners != list(PLANNERS):
        raise ValueError("planner roster must include the three planners in canonical order")
    expected = [f"heuristic:{style}" for style in HEURISTIC_STYLES]
    if heuristics != expected:
        raise ValueError("heuristic roster must include only the three development heuristics")
    return planners, heuristics


def validate_config(config: dict[str, Any]) -> None:
    """Fail closed on unbounded or protected experimental tournament inputs."""
    required = {"experiment_id", "run_id", "width", "height", "seeds", "planners",
                "heuristics", "edge_budget", "max_depth", "max_wall_seconds"}
    optional = {"agent_edge_budgets", "matchups", "workers", "native_territory",
                "frozen_opponent", "policy_value_checkpoint"}
    if (not isinstance(config, dict)
            or not required <= set(config) or set(config) - required - optional):
        raise ValueError("experimental search config fields differ from the frozen runner contract")
    if (not isinstance(config["experiment_id"], str) or not config["experiment_id"]
            or not isinstance(config["run_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", config["run_id"])):
        raise ValueError("experiment_id must be non-empty and run_id must be a safe artifact ID")
    planners, _ = _agent_ids(config)
    if ("frozen_opponent" in config
            and (not isinstance(config["frozen_opponent"], str)
                 or config["frozen_opponent"] not in FROZEN_OPPONENTS)):
        raise ValueError("frozen_opponent must be the protocol-permitted v15-step236 checkpoint")
    width, height = config["width"], config["height"]
    seeds = config["seeds"]
    edge_budget, max_depth = config["edge_budget"], config["max_depth"]
    wall_cap = config["max_wall_seconds"]
    if (type(width) is not int or type(height) is not int or width < 2 or height < 2
            or type(edge_budget) is not int or not 8 <= edge_budget <= 512
            or type(max_depth) is not int or not 1 <= max_depth <= 12
            or edge_budget < max_depth + 1
            or type(wall_cap) is not int or not 60 <= wall_cap <= 10_800
            or not isinstance(seeds, list) or len(seeds) < 1 or len(seeds) > 256
            or any(type(seed) is not int or not 0 <= seed < (1 << 53) for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("experimental search workload is outside its bounded config limits")
    if "frozen_opponent" in config:
        expected_matchup = f"fixed-depth-vs-{config['frozen_opponent']}"
        if (width, height) != (12, 12) or config.get("matchups") != [expected_matchup]:
            raise ValueError("v15-step236 comparison requires a 12x12 board and its single explicit matchup")
    if ("policy_value_checkpoint" in config
            and config["policy_value_checkpoint"] not in POLICY_VALUE_CHECKPOINTS):
        raise ValueError("policy_value_checkpoint must identify a pinned experimental checkpoint")
    if "policy_value_checkpoint" in config and (width, height) != (12, 12):
        raise ValueError("the pinned Policy-Value checkpoint requires a 12x12 board")
    if "agent_edge_budgets" in config:
        budgets = config["agent_edge_budgets"]
        if (not isinstance(budgets, dict) or set(budgets) != set(planners)
                or any(type(value) is not int or not max_depth + 1 <= value <= 512
                       for value in budgets.values())):
            raise ValueError("per-planner edge budgets must cover each planner and allow a depth-limited path")
    if "matchups" in config:
        available = _matchups(planners, config["heuristics"], config.get("frozen_opponent"),
                              config.get("policy_value_checkpoint"))
        allowed = {f"{left}-vs-{right}" for left, right in available}
        selected = config["matchups"]
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(item, str) for item in selected)
                or len(set(selected)) != len(selected)
                or any(item not in allowed for item in selected)):
            raise ValueError("selected matchups must be unique members of the bounded development roster")
    if "workers" in config and (type(config["workers"]) is not int
                                or not 1 <= config["workers"] <= 4):
        raise ValueError("workers must be an integer from 1 to 4")
    if "native_territory" in config and type(config["native_territory"]) is not bool:
        raise ValueError("native_territory must be a boolean")


def _matchups(planners: list[str], heuristics: list[str],
              frozen_opponent: str | None = None,
              policy_value_checkpoint: str | None = None) -> list[tuple[str, str]]:
    planner_pairs = [(planners[i], planners[j])
                     for i in range(len(planners)) for j in range(i + 1, len(planners))]
    result = planner_pairs + [(planner, heuristic) for planner in planners for heuristic in heuristics]
    if frozen_opponent:
        # The protocol permits this frozen opponent for fixed-depth comparison only.
        result.append(("fixed-depth", frozen_opponent))
    if policy_value_checkpoint:
        result.extend((planner, policy_value_checkpoint) for planner in planners)
    return result


def _selected_matchups(config: dict[str, Any], planners: list[str],
                       heuristics: list[str]) -> list[tuple[str, str]]:
    available = _matchups(planners, heuristics, config.get("frozen_opponent"),
                          config.get("policy_value_checkpoint"))
    selected_ids = config.get("matchups")
    if selected_ids is None:
        return available
    by_id = {f"{left}-vs-{right}": (left, right) for left, right in available}
    return [by_id[matchup_id] for matchup_id in selected_ids]


def _run_game(left_id: str, right_id: str, matchup_id: str, seed: int, target_seat: int,
              width: int, height: int, edge_budgets: dict[str, int], max_depth: int) -> dict[str, Any]:
    initial, _ = initialize_episode(width, height, seed, seed)
    current = initial
    left_stats, right_stats = _new_stats(), _new_stats()
    trace: list[dict[str, Any]] = []
    first_separation_tick = None
    process_started = time.process_time_ns()
    while current.result is None:
        compact = from_arena_state(current)
        regions = (flood_region(compact, compact.heads[0]), flood_region(compact, compact.heads[1]))
        if first_separation_tick is None and not (regions[0] & regions[1]):
            first_separation_tick = current.tick
        actions: list[int | str] = []
        for seat in (0, 1):
            agent_id = left_id if seat == target_seat else right_id
            action, elapsed, counts = _choose(agent_id, current, compact, seat, matchup_id, seed,
                                               edge_budgets.get(agent_id, 8), max_depth)
            stats = left_stats if seat == target_seat else right_stats
            stats["decisions"] += 1
            stats["elapsed_ns"] += elapsed
            stats["decision_ns"].append(elapsed)
            for key in ("transition_edges", "leaf_evaluations", "simulations"):
                stats[key] += counts.get(key, 0)
            stats["completed_depth_sum"] += counts.get("completed_depth", 0)
            actions.append(action)
        transition = step(current, actions)
        for seat, causes in enumerate(transition.causes):
            if not causes:
                continue
            stats = left_stats if seat == target_seat else right_stats
            for cause in causes:
                stats["death_causes"][cause] = stats["death_causes"].get(cause, 0) + 1
            alternatives = legal_actions(compact, seat)
            for alternative in alternatives:
                if alternative == actions[seat]:
                    continue
                candidate = list(actions)
                candidate[seat] = alternative
                successor, _, _ = compact_transition(compact, candidate)
                if successor.alive[seat]:
                    stats["avoidable_deaths"] += 1
                    break
        trace.append({
            "tick": current.tick,
            "actions_by_seat": actions,
            "causes_by_seat": [list(causes) for causes in transition.causes],
        })
        current = transition.state
    process_ns = time.process_time_ns() - process_started
    separation_ticks_survived = (max(0, len(trace) - first_separation_tick)
                                  if first_separation_tick is not None else 0)
    left_stats["ticks_after_separation"] = separation_ticks_survived
    right_stats["ticks_after_separation"] = separation_ticks_survived
    if current.result == "draw":
        outcome = "draw"
    elif current.result == f"agent_{target_seat}_win":
        outcome = "left_win"
    else:
        outcome = "right_win"
    return {
        "seed": seed, "target_seat": target_seat, "outcome": outcome,
        "ticks": len(trace), "left_stats": left_stats, "right_stats": right_stats,
        "first_separation_tick": first_separation_tick,
        "ticks_after_separation": separation_ticks_survived,
        "process_cpu_ns": process_ns, "trace": trace,
    }


def _run_game_task(arguments: tuple[Any, ...]) -> dict[str, Any]:
    """Top-level process-pool entry point for an isolated deterministic game."""
    return _run_game(*arguments)


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))]


def _summarize(games: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(game["outcome"] == "left_win" for game in games)
    draws = sum(game["outcome"] == "draw" for game in games)
    losses = len(games) - wins - draws
    left_decisions = [duration for game in games for duration in game["left_stats"]["decision_ns"]]
    right_decisions = [duration for game in games for duration in game["right_stats"]["decision_ns"]]
    left_ns = [game["left_stats"]["elapsed_ns"] for game in games]
    right_ns = [game["right_stats"]["elapsed_ns"] for game in games]
    return {
        "games": len(games), "wins": wins, "draws": draws, "losses": losses,
        "observed_win_rate": wins / max(1, len(games)),
        "score_rate_draw_half": (wins + 0.5 * draws) / max(1, len(games)),
        "left_decisions": len(left_decisions),
        "left_decision_ns_mean": int(statistics.fmean(left_decisions)) if left_decisions else 0,
        "left_decision_ns_p95": _percentile(left_decisions, 0.95),
        "right_decisions": len(right_decisions),
        "right_decision_ns_mean": int(statistics.fmean(right_decisions)) if right_decisions else 0,
        "right_decision_ns_p95": _percentile(right_decisions, 0.95),
        "left_total_decision_ns": sum(left_ns),
        "right_total_decision_ns": sum(right_ns),
        "process_cpu_ns": sum(game["process_cpu_ns"] for game in games),
        "left_transition_edges": sum(game["left_stats"]["transition_edges"] for game in games),
        "right_transition_edges": sum(game["right_stats"]["transition_edges"] for game in games),
        "left_leaf_evaluations": sum(game["left_stats"]["leaf_evaluations"] for game in games),
        "right_leaf_evaluations": sum(game["right_stats"]["leaf_evaluations"] for game in games),
        "left_simulations": sum(game["left_stats"]["simulations"] for game in games),
        "right_simulations": sum(game["right_stats"]["simulations"] for game in games),
        "left_avoidable_deaths": sum(game["left_stats"]["avoidable_deaths"] for game in games),
        "right_avoidable_deaths": sum(game["right_stats"]["avoidable_deaths"] for game in games),
        "left_death_causes": _sum_causes(games, "left_stats"),
        "right_death_causes": _sum_causes(games, "right_stats"),
        "games_with_separation": sum(game["first_separation_tick"] is not None for game in games),
        "mean_ticks_after_separation": statistics.fmean(
            game["ticks_after_separation"] for game in games) if games else 0.0,
    }


def _sum_causes(games: list[dict[str, Any]], side: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for game in games:
        for cause, count in game[side]["death_causes"].items():
            totals[cause] = totals.get(cause, 0) + count
    return dict(sorted(totals.items()))


def _runtime_cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _available_cpu_count() -> int:
    count = os.cpu_count() or 1
    try:
        count = min(count, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="ascii").split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return max(1, count)


def run_experiment(config: dict[str, Any], output_dir: str | Path, *,
                   source_manifest_sha256: str) -> dict[str, Any]:
    """Run the configured, bounded tournament and write replay/provenance artifacts."""
    if (not isinstance(source_manifest_sha256, str) or len(source_manifest_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_manifest_sha256)):
        raise ValueError("source manifest digest must be lowercase SHA-256")
    validate_config(config)
    planners, heuristics = _agent_ids(config)
    width, height = config["width"], config["height"]
    seeds = config["seeds"]
    edge_budget, max_depth = config["edge_budget"], config["max_depth"]
    edge_budgets = config.get("agent_edge_budgets", {agent_id: edge_budget for agent_id in planners})
    wall_cap = config["max_wall_seconds"]
    available_cpus = _available_cpu_count()
    worker_count = min(config.get("workers", 4), available_cpus, 4)

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experimental output {output}")
    output.mkdir(parents=True)
    matchups = _selected_matchups(config, planners, heuristics)
    started_wall = time.monotonic()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    matchup_reports = []
    game_records = []
    completed = True
    pool = ProcessPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    try:
        for left_id, right_id in matchups:
            matchup_id = f"{left_id}-vs-{right_id}"
            games = []
            # Keep both seat-swapped games for each seed in the same task
            # window; a time cap can only cut between complete seed clusters.
            task_batch: list[tuple[Any, ...]] = []

            def flush_batch() -> None:
                if not task_batch:
                    return
                tasks = map(_run_game_task, task_batch)
                results = pool.map(_run_game_task, task_batch, chunksize=1) if pool else tasks
                for game in results:
                    game["matchup_id"] = matchup_id
                    game["left_agent"] = left_id
                    game["right_agent"] = right_id
                    games.append(game)
                    game_records.append(game)
                task_batch.clear()

            for seed in seeds:
                if time.monotonic() - started_wall >= wall_cap:
                    completed = False
                    break
                task_batch.extend(
                    (left_id, right_id, matchup_id, seed, target_seat,
                     width, height, edge_budgets, max_depth)
                    for target_seat in (0, 1)
                )
                if len(task_batch) >= worker_count * 2:
                    flush_batch()
            flush_batch()
            matchup_reports.append({"matchup_id": matchup_id, "left_agent": left_id,
                                    "right_agent": right_id, "summary": _summarize(games)})
            if not completed:
                break
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)

    from .native.territory import available as _native_available
    native_loaded = _native_available()
    if config.get("native_territory", False) and not native_loaded:
        raise RuntimeError("native_territory was requested but its shared library did not load")
    runtime = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_name": _runtime_cpu_name(), "cpu_count": available_cpus,
        "workers": worker_count,
        "native_territory": native_loaded,
        "gpu_enabled": False, "internet_enabled": False,
        "wall_seconds": round(time.monotonic() - started_wall, 3),
        "cpu_process_seconds": round(sum(game["process_cpu_ns"] for game in game_records) / 1e9, 3),
    }
    if config.get("frozen_opponent") or config.get("policy_value_checkpoint"):
        # Frozen inference uses PyTorch supplied by Kaggle's base image. Keep
        # the actual runtime evidence with the result instead of assuming it.
        import torch

        runtime["torch_version"] = torch.__version__
        runtime["torch_cuda_build"] = torch.version.cuda
        runtime["kaggle_kernel_run_type"] = os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        try:
            runtime["host_os"] = platform.freedesktop_os_release().get("PRETTY_NAME")
        except (AttributeError, OSError):
            runtime["host_os"] = None
    if config.get("policy_value_checkpoint"):
        runtime["policy_value_checkpoint"] = config["policy_value_checkpoint"]
    report = {
        "format": "circuit-ai-simultaneous-search-experiment/1",
        "experiment_id": config["experiment_id"], "run_id": config["run_id"],
        "status": "complete" if completed else "partial-time-cap",
        "source_manifest_sha256": source_manifest_sha256,
        "frozen_opponents": ([{"id": config["frozen_opponent"],
                                "sha256": FROZEN_OPPONENTS[config["frozen_opponent"]]}]
                             if config.get("frozen_opponent") else []),
        "policy_value_checkpoints": ([{"id": config["policy_value_checkpoint"],
                                        "sha256": POLICY_VALUE_CHECKPOINTS[
                                            config["policy_value_checkpoint"]]["sha256"]}]
                                      if config.get("policy_value_checkpoint") else []),
        "started_at_utc": started_utc, "config": config, "runtime": runtime,
        "matchups": matchup_reports,
    }
    _write_json(output / "report.json", report, digest_field="report_sha256")
    games_path = output / "games.jsonl"
    with games_path.open("w", encoding="utf-8") as stream:
        for game in game_records:
            stream.write(json.dumps(game, sort_keys=True, ensure_ascii=False) + "\n")
    files = [
        {"path": name, "sha256": _file_sha256(output / name)}
        for name in ("report.json", "games.jsonl")
    ]
    manifest = {
        "format": "circuit-ai-simultaneous-search-output-manifest/1",
        "experiment_id": config["experiment_id"], "run_id": config["run_id"],
        "source_manifest_sha256": source_manifest_sha256, "files": files,
    }
    _write_json(output / "manifest.json", manifest, digest_field="manifest_sha256")
    print(json.dumps({"run_id": config["run_id"], "status": report["status"],
                      "matchups": matchup_reports, "runtime": runtime},
                     sort_keys=True, ensure_ascii=False))
    return report
