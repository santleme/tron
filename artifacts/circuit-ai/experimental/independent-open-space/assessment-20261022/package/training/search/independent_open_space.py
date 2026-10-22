"""One-shot, non-official independent evaluation of frozen Rust search vs open-space.

This entrypoint is packaged separately from search-training/tournament modes.
It never reads or writes the official G008 registry or its consumed seed roster.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import time
from typing import Any

from training.arena.engine import ACTION_ORDER, initialize_episode, step
from training.eval.g008_final_opponent import final_opponent_policy
from training.eval.lc_eval import wilson_interval
from training.search.bitboard import from_arena_state
from training.search.native_planner import ensure_loaded, plan


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decision_seed(seed: int, seat: int, tick: int) -> int:
    payload = f"circuit-ai-independent-open-space-v1\0{seed}\0{seat}\0{tick}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _wire_state(state) -> dict[str, Any]:
    return {
        "width": state.width,
        "height": state.height,
        "positions": [{"x": a.position[0], "y": a.position[1]} for a in state.agents],
        "headings": [ACTION_ORDER[a.heading] for a in state.agents],
        "trails": [[{"x": x, "y": y} for x, y in agent.trail] for agent in state.agents],
    }


def _game(seed: int, candidate_seat: int, config: dict[str, Any]) -> dict[str, Any]:
    initial, streams = initialize_episode(config["width"], config["height"], seed, seed)
    current = initial
    opponent = final_opponent_policy()
    trace: list[dict[str, Any]] = []
    decision_ns: list[int] = []
    completed_depth = transition_edges = leaf_evaluations = 0
    while current.result is None:
        compact = from_arena_state(current)
        actions: list[int | str] = [-1, -1]
        started = time.perf_counter_ns()
        action, depth, edges, leaves = plan(
            compact, player=candidate_seat, max_depth=config["max_depth"],
            edge_budget=config["edge_budget"],
            seed=_decision_seed(seed, candidate_seat, current.tick),
        )
        decision_ns.append(time.perf_counter_ns() - started)
        completed_depth += depth
        transition_edges += edges
        leaf_evaluations += leaves
        actions[candidate_seat] = action if action >= 0 else "INVALID"
        opponent_seat = 1 - candidate_seat
        actions[opponent_seat] = opponent(current, opponent_seat)
        transition = step(current, actions)
        terminal = (["agent_0_dead", "agent_1_dead"]
                    if all(not agent.alive for agent in transition.state.agents)
                    else [f"agent_{i}_dead" for i, agent in enumerate(transition.state.agents)
                          if not agent.alive])
        trace.append({
            "tick": current.tick,
            "joint_actions": [item if type(item) is int and item in range(4) else "INVALID"
                              for item in actions],
            "collision_causes": [list(causes) for causes in transition.causes],
            "terminal_causes": terminal,
        })
        current = transition.state

    won = current.result == f"agent_{candidate_seat}_win"
    outcome = "win" if won else ("draw" if current.result == "draw" else "loss")
    replay = {
        "format": "lc-replay/1",
        "run_id": config["run_id"],
        "match_id": f"{config['run_id']}-{seed}-{candidate_seat}",
        "engine_version": "lc-engine/1",
        "observation_schema": "lc-observation/1",
        "action_schema": "lc-action/1",
        "master_seed": str(seed),
        "arena_index": seed,
        "derived_streams": [
            {"label": stream.label, "initstate": str(stream.initstate),
             "initseq": str(stream.initseq)} for stream in streams
        ],
        "agents": [
            {"agent_id": "fixed-depth-rust", "checkpoint_id": config["candidate_id"]}
            if seat == candidate_seat else
            {"agent_id": "open-space", "checkpoint_id": config["opponent_id"]}
            for seat in (0, 1)
        ],
        "initial_state": _wire_state(initial),
        "ticks": trace,
        "result": current.result,
        "tick_count": len(trace),
    }
    replay["replay_sha256"] = _sha256_bytes(_canonical(replay))
    return {
        "seed": seed,
        "candidate_seat": candidate_seat,
        "outcome": outcome,
        "ticks": len(trace),
        "candidate_decision_count": len(decision_ns),
        "candidate_decision_ns_mean": int(statistics.fmean(decision_ns)) if decision_ns else 0,
        "candidate_decision_ns_p95": sorted(decision_ns)[int((len(decision_ns) - 1) * 0.95)] if decision_ns else 0,
        "candidate_completed_depth_mean": completed_depth / max(1, len(decision_ns)),
        "candidate_transition_edges": transition_edges,
        "candidate_leaf_evaluations": leaf_evaluations,
        "replay": replay,
    }


def _run_seed(seed: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    return [_game(seed, seat, config) for seat in (0, 1)]


def _write_json(path: Path, value: dict[str, Any], digest_field: str | None = None) -> str:
    payload = dict(value)
    if digest_field:
        payload.pop(digest_field, None)
        digest = _sha256_bytes(_canonical(payload))
        payload[digest_field] = digest
    else:
        digest = _sha256_bytes(_canonical(payload))
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return digest


def run(config: dict[str, Any], output: Path, source_manifest_sha256: str) -> dict[str, Any]:
    seeds = config["seeds"]
    package = Path(__file__).parent
    candidate_dir = package / "candidate"
    rust_source = package / "native" / "fixed_depth.rs"
    wasm_path = candidate_dir / "fixed_depth.wasm"
    opponent_paths = (Path(__file__).parents[1] / "eval" / "lc_eval.py",
                      Path(__file__).parents[1] / "eval" / "g008_final_opponent.py")
    seed_plan_path = package / "independent-seed-plan.json"
    if (not seed_plan_path.is_file() or rust_source.is_symlink() or wasm_path.is_symlink()
            or not rust_source.is_file() or not wasm_path.is_file()):
        raise RuntimeError("Frozen candidate or committed seed plan is absent from the submitted package.")
    candidate = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))
    if (candidate.get("candidate_id") != config["candidate_id"]
            or candidate["artifacts"]["rust_source"]["sha256"] != config["candidate_source_sha256"]
            or candidate["artifacts"]["wasm"]["sha256"] != config["candidate_wasm_sha256"]):
        raise RuntimeError("Submitted candidate manifest does not bind the configured artifact hashes.")
    seed_plan = json.loads(seed_plan_path.read_text(encoding="utf-8"))
    if (seed_plan.get("run_id") != config["run_id"] or seed_plan.get("seeds") != seeds
            or seed_plan.get("seed_plan_sha256") != config["seed_plan_sha256"]
            or _sha256_bytes(_canonical({key: value for key, value in seed_plan.items()
                                         if key != "seed_plan_sha256"})) != config["seed_plan_sha256"]):
        raise RuntimeError("Submitted seed plan does not match the exact run config/hash.")
    if (_sha256_bytes(rust_source.read_bytes()) != config["candidate_source_sha256"]
            or _sha256_bytes(wasm_path.read_bytes()) != config["candidate_wasm_sha256"]):
        raise RuntimeError("Submitted native source/WASM bytes differ from the frozen candidate hashes.")
    opponent_hash = _sha256_bytes(_canonical({path.name: _sha256_bytes(path.read_bytes())
                                              for path in opponent_paths}))
    if opponent_hash != config["opponent_source_sha256"]:
        raise RuntimeError("Submitted open-space policy bytes differ from the committed opponent source hash.")
    if len(seeds) != 256 or len(set(seeds)) != 256:
        raise ValueError("Independent assessment requires 256 unique seeds and both candidate seats.")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite independent-evaluation output {output}")
    output.mkdir(parents=True)
    shutil.copy2(seed_plan_path, output / "seed-plan.json")
    shutil.copy2(package.parent.parent / "source-manifest.json", output / "source-manifest.json")
    replay_dir = output / "replays"
    replay_dir.mkdir()
    started = time.monotonic()
    workers = min(config.get("workers", os.cpu_count() or 1), os.cpu_count() or 1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=ensure_loaded) as pool:
            grouped = pool.map(_run_seed, seeds, [config] * len(seeds), chunksize=1)
            games = [row for pair in grouped for row in pair]
    else:
        ensure_loaded()
        games = [row for seed in seeds for row in _run_seed(seed, config)]
    replay_rows = []
    for row in games:
        replay = row.pop("replay")
        path = f"replays/{row['seed']}-{row['candidate_seat']}.json"
        replay_path = output / path
        _write_json(replay_path, replay, "replay_sha256")
        row["replay_file"] = path
        row["replay_sha256"] = replay["replay_sha256"]
        replay_rows.append(row)
    games_path = output / "games.jsonl"
    with games_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in replay_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False) + "\n")
    elapsed = time.monotonic() - started
    wins = sum(row["outcome"] == "win" for row in replay_rows)
    draws = sum(row["outcome"] == "draw" for row in replay_rows)
    losses = len(replay_rows) - wins - draws
    seat_summaries = {}
    for seat in (0, 1):
        seat_rows = [row for row in replay_rows if row["candidate_seat"] == seat]
        seat_summaries[str(seat)] = {
            "games": len(seat_rows),
            "wins": sum(row["outcome"] == "win" for row in seat_rows),
            "draws": sum(row["outcome"] == "draw" for row in seat_rows),
            "losses": sum(row["outcome"] == "loss" for row in seat_rows),
        }
    report = {
        "format": "circuit-ai-independent-open-space-evaluation/1",
        "run_id": config["run_id"],
        "official": False,
        "purpose": "fresh independent one-shot assessment; not used for tuning",
        "candidate_id": config["candidate_id"],
        "candidate_wasm_sha256": config["candidate_wasm_sha256"],
        "candidate_source_sha256": config["candidate_source_sha256"],
        "opponent_id": "open-space",
        "opponent_source_sha256": config["opponent_source_sha256"],
        "seed_plan_sha256": config["seed_plan_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "seed_count": len(seeds),
        "game_count": len(replay_rows),
        "seat_order": [0, 1],
        "summary": {
            "wins": wins, "draws": draws, "losses": losses,
            "observed_win_rate": wins / len(replay_rows),
            "score_rate_draw_half": (wins + 0.5 * draws) / len(replay_rows),
            "wilson_95_win_rate": wilson_interval(wins, draws, losses),
            "seat_results": seat_summaries,
            "mean_survival_ticks": statistics.fmean(row["ticks"] for row in replay_rows),
            "candidate_decision_ns_mean": int(statistics.fmean(
                row["candidate_decision_ns_mean"] for row in replay_rows)),
            "candidate_decision_ns_p95": sorted(
                row["candidate_decision_ns_p95"] for row in replay_rows)[int((len(replay_rows) - 1) * 0.95)],
            "candidate_completed_depth_mean": statistics.fmean(
                row["candidate_completed_depth_mean"] for row in replay_rows),
        },
        "runtime": {
            "platform": platform.platform(), "python": platform.python_version(),
            "cpu_name": os.environ.get("CPU_MODEL_NAME", "Kaggle runtime CPU"),
            "cpu_count": os.cpu_count() or 1, "workers": workers,
            "native_library_sha256": _sha256_bytes((package / "native/libcircuit_planner.so").read_bytes()),
            "gpu_enabled": False, "internet_enabled": False,
            "elapsed_seconds": round(elapsed, 3),
        },
        "status": "complete",
    }
    _write_json(output / "report.json", report, "report_sha256")
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": path.relative_to(output).as_posix(),
                          "sha256": _sha256_bytes(path.read_bytes())})
    manifest = {
        "format": "circuit-ai-independent-open-space-output-manifest/1",
        "run_id": config["run_id"],
        "source_manifest_sha256": source_manifest_sha256,
        "seed_plan_sha256": config["seed_plan_sha256"],
        "files": files,
    }
    _write_json(output / "manifest.json", manifest, "manifest_sha256")
    print(json.dumps(report["summary"], sort_keys=True))
    return report
