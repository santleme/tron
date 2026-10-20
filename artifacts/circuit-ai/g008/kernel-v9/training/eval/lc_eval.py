"""The fixed lc-eval/1 benchmark: 64 games per opponent, both seats and seeds 0..31."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

from training.arena.engine import ACTION_ORDER, DELTA, initialize_episode, observe, step
from training.ppo.protocol import digest_json, write_json

HELD_OUT_SEEDS = tuple(range(32))
OPPONENTS = ("open-space", "step-0")


def wilson_interval(wins: int, draws: int, losses: int, z: float = 1.959963984540054) -> list[float]:
    """Wilson 95% interval for the proportion of games won (draws are not wins)."""
    n = wins + draws + losses
    if n == 0:
        raise ValueError("Wilson interval requires at least one game")
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def open_space_action(state, seat: int) -> int:
    """Choose legal destination maximizing reachable empty cells; action order breaks ties."""
    me = state.agents[seat]
    occupied = {cell for agent in state.agents for cell in agent.trail}

    def reachable(start: tuple[int, int]) -> int:
        seen = {start}
        todo = [start]
        while todo:
            x, y = todo.pop()
            for dx, dy in DELTA:
                cell = (x + dx, y + dy)
                if (0 <= cell[0] < state.width and 0 <= cell[1] < state.height
                        and cell not in occupied and cell not in seen):
                    seen.add(cell)
                    todo.append(cell)
        return len(seen)

    candidates = []
    for action, (dx, dy) in enumerate(DELTA):
        if action == (me.heading + 2) % 4:
            continue
        dest = (me.position[0] + dx, me.position[1] + dy)
        if not (0 <= dest[0] < state.width and 0 <= dest[1] < state.height) or dest in occupied:
            continue
        candidates.append((reachable(dest), -action, action))
    return max(candidates)[2] if candidates else "INVALID"


def _as_wire_state(state):
    return {"width": state.width, "height": state.height,
            "positions": [{"x": a.position[0], "y": a.position[1]} for a in state.agents],
            "headings": [ACTION_ORDER[a.heading] for a in state.agents],
            "trails": [[{"x": x, "y": y} for x, y in agent.trail] for agent in state.agents]}


def _episode_replay(state, seed: int, arena_index: int, run_id: str, match_id: str,
                    target_id: str, opponent_id: str, target_seat: int, initial_state,
                    target_policy: Callable, step0_policy: Callable):
    ticks = []
    collision_causes: set[str] = set()
    current = state
    while current.result is None:
        actions = []
        for seat in range(2):
            if seat == target_seat:
                actions.append(_policy_action(target_policy, current, seat))
            elif opponent_id == "open-space":
                actions.append(open_space_action(current, seat))
            else:
                actions.append(_policy_action(step0_policy, current, seat))
        transition = step(current, actions)
        for causes in transition.causes:
            collision_causes.update(causes)
        terminal = (["agent_0_dead", "agent_1_dead"] if all(not a.alive for a in transition.state.agents)
                    else [f"agent_{i}_dead" for i, agent in enumerate(transition.state.agents) if not agent.alive])
        ticks.append({"tick": current.tick, "joint_actions": [action if action in range(4) else "INVALID" for action in actions],
                      "collision_causes": [list(cause) for cause in transition.causes], "terminal_causes": terminal})
        current = transition.state
    streams = initialize_episode(state.width, state.height, seed, arena_index)[1]
    replay = {"format": "lc-replay/1", "run_id": run_id, "match_id": match_id,
              "engine_version": "lc-engine/1", "observation_schema": "lc-observation/1", "action_schema": "lc-action/1",
              "master_seed": str(seed), "arena_index": arena_index,
              "derived_streams": [{"label": s.label, "initstate": str(s.initstate), "initseq": str(s.initseq)} for s in streams],
              "agents": [{"agent_id": "target" if i == target_seat else opponent_id,
                          "checkpoint_id": target_id if i == target_seat else opponent_id} for i in range(2)],
              "initial_state": _as_wire_state(initial_state), "ticks": ticks, "result": current.result, "tick_count": len(ticks)}
    replay["replay_sha256"] = digest_json(replay, "replay_sha256")
    target_won = current.result == f"agent_{target_seat}_win"
    outcome = "win" if target_won else ("draw" if current.result == "draw" else "loss")
    return replay, outcome, len(ticks), sorted(collision_causes)


def _policy_action(policy: Callable, state, seat: int) -> int:
    result = policy(observe(state, seat))
    if type(result) is int and result in range(4):
        return result
    if hasattr(result, "__len__") and len(result) == 4:
        # Lowest action index wins ties.
        return max(range(4), key=lambda i: float(result[i]))
    raise ValueError("policy must return action index 0..3 or four action scores")


def evaluate(target_policy: Callable, step0_policy: Callable, run_id: str, checkpoint_id: str,
             output_dir: str | Path, width: int = 12, height: int = 12) -> dict:
    """Run the exact held-out sample matrix and emit report plus one replay per game."""
    if width < 2 or height < 2:
        raise ValueError("board dimensions must be at least 2")
    output_dir = Path(output_dir)
    replay_dir = output_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    matchups = []
    for opponent_id in OPPONENTS:
        games, survival, causes = [], [], {}
        for seed in HELD_OUT_SEEDS:
            for target_seat in (0, 1):
                initial, _ = initialize_episode(width, height, seed, seed)
                replay, result, ticks, game_causes = _episode_replay(
                    initial, seed, seed, run_id, f"{opponent_id}-{seed:02d}-{target_seat}",
                    checkpoint_id, opponent_id, target_seat, initial, target_policy, step0_policy)
                replay_name = f"{opponent_id}-{seed:02d}-{target_seat}.json"
                write_json(replay_dir / replay_name, replay)
                games.append({"seed": str(seed), "target_seat": target_seat, "result": result,
                              "ticks": ticks, "collision_causes": game_causes,
                              "replay_file": f"evaluation/replays/{replay_name}"})
                survival.append(ticks)
                for cause in game_causes:
                    causes[cause] = causes.get(cause, 0) + 1
        wins = sum(g["result"] == "win" for g in games)
        draws = sum(g["result"] == "draw" for g in games)
        losses = sum(g["result"] == "loss" for g in games)
        matchups.append({"opponent_id": opponent_id, "games": games,
                         "summary": {"wins": wins, "draws": draws, "losses": losses,
                                     "win_rate": wins / len(games),
                                     "wilson_95": wilson_interval(wins, draws, losses),
                                     "mean_survival_ticks": statistics.fmean(survival),
                                     "median_survival_ticks": statistics.median(survival),
                                     "collision_causes": dict(sorted(causes.items()))}})
    report = {"format": "lc-eval/1", "run_id": run_id, "checkpoint_id": checkpoint_id,
              "protocol": "lc-eval/1", "held_out_seeds": [str(i) for i in HELD_OUT_SEEDS], "matchups": matchups}
    write_json(output_dir / "report.json", report, "report_sha256")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--step-0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    from training.ppo.train import load_policy
    target, target_id = load_policy(args.target, args.device)
    baseline, _ = load_policy(args.step_0, args.device)
    report = evaluate(target, baseline, args.run_id, args.checkpoint_id or target_id, args.output,
                      target.width, target.height)
    print(json.dumps({m["opponent_id"]: m["summary"] for m in report["matchups"]}, indent=2))


if __name__ == "__main__":
    main()
