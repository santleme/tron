"""Run registered G008 validation or final evaluation on Kaggle.

The complete 512-game suite is executed once. Per-turn conditional one-step
counterfactuals and small replay samples are captured during that simulation;
this module never recreates games from a report. Counterfactual survival is a
local diagnostic, not evidence that an alternate policy is strategically better.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable

from jsonschema import Draft202012Validator

from training.arena.engine import ACTION_ORDER, DELTA, initialize_episode, observe, step
from training.ppo.protocol import digest_json, file_sha256, write_json


ROOT = Path(__file__).resolve().parents[2]
SEED_PLAN_PATH = ROOT / "training/configs/g008-split-seed-plan.json"
_CANDIDATE_DATASET_SOURCE = "pascalcoimbra/g008-frozen-candidate"
_CANDIDATE_DATASET_PATH = Path("/kaggle/input/g008-frozen-candidate/candidate.pt")
_G007_OPPONENT_PATH = Path(__file__).resolve().parent / "assets/g007-step-300.pt"
G008_FROZEN_DATASET_SOURCE = _CANDIDATE_DATASET_SOURCE
G008_CANDIDATE_CHECKPOINT_PATH = _CANDIDATE_DATASET_PATH
G007_STEP_300_CHECKPOINT_PATH = _G007_OPPONENT_PATH
_EXPECTED_ROSTERS = {
    "validation": {
        "opponent_id": "g007-step-300",
        "run_key": "validation-suite-v1",
        "root_range": (1_000_000, 1_000_255),
        "roster_sha256": "974adf788d5519633ad1fe7130a6deb634a0cc55d6af2d549e06d64c0f5496c0",
        "opponent_code_sha256": "f79ed460afc6d4419e2e1b0bd084a4f1cf77058fcb8cdcebb7122a2864304412",
    },
}
G007_STEP_300_CHECKPOINT_SHA256 = _EXPECTED_ROSTERS["validation"]["opponent_code_sha256"]
_REPLAY_SAMPLE_LIMIT = 8
_VALIDATION_REFERENCE = {
    "run_id": "g008-v15-step236",
    "report_sha256": "22f7e0fca065eba24ab57c5fae4ea78223fa6d01153a7a8d90e83ac8269228d7",
    "wins": 217, "draws": 51, "losses": 244,
}
_NONINFERIORITY_MARGIN = 0.065
_CAMPAIGN_V2_ID = "g008-competitive-20261021-v2"


def wilson_interval(wins: int, draws: int, losses: int,
                    z: float = 1.959963984540054) -> list[float]:
    """Wilson 95% interval for win proportion, counting draws as non-wins."""
    values = (wins, draws, losses)
    if any(type(value) is not int or value < 0 for value in values) or sum(values) == 0:
        raise ValueError("Wilson interval requires nonnegative W/D/L and at least one game")
    n = sum(values)
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _terminal_reason(causes: Iterable[str]) -> str:
    causes = set(causes)
    if "out_of_bounds" in causes:
        return "wall"
    if causes & {"occupied_head", "occupied_trail", "same_destination", "head_swap"}:
        return "collision"
    return "other"


def _free_actions(state, seat: int) -> list[int]:
    agent = state.agents[seat]
    occupied = {cell for player in state.agents for cell in player.trail}
    actions = []
    for action, (dx, dy) in enumerate(DELTA):
        if action == (agent.heading + 2) % 4:
            continue
        cell = (agent.position[0] + dx, agent.position[1] + dy)
        if 0 <= cell[0] < state.width and 0 <= cell[1] < state.height and cell not in occupied:
            actions.append(action)
    return actions


def _reachable_area(state, start: tuple[int, int]) -> int:
    occupied = {cell for player in state.agents for cell in player.trail}
    occupied.discard(start)
    seen, pending = {start}, [start]
    while pending:
        x, y = pending.pop()
        for dx, dy in DELTA:
            cell = (x + dx, y + dy)
            if (0 <= cell[0] < state.width and 0 <= cell[1] < state.height
                    and cell not in occupied and cell not in seen):
                seen.add(cell)
                pending.append(cell)
    return len(seen)


def _policy_action(policy: Callable, state, seat: int, *, state_input: bool = False) -> int | str:
    result = policy(state, seat) if state_input else policy(observe(state, seat))
    if result == "INVALID":
        return result
    if type(result) is int and result in range(4):
        return result
    if hasattr(result, "__len__") and len(result) == 4:
        return max(range(4), key=lambda action: float(result[action]))
    raise ValueError("policy must return action index 0..3 or four action scores")


def _wire_state(state) -> dict[str, Any]:
    return {
        "width": state.width,
        "height": state.height,
        "positions": [{"x": a.position[0], "y": a.position[1]} for a in state.agents],
        "headings": [ACTION_ORDER[a.heading] for a in state.agents],
        "trails": [[{"x": x, "y": y} for x, y in player.trail] for player in state.agents],
    }


def _episode(initial, seed: int, arena_index: int, target_seat: int,
             candidate: Callable, opponent: Callable, *, run_id: str,
             candidate_checkpoint_id: str, opponent_id: str,
             opponent_checkpoint_id: str, keep_replay: bool,
             opponent_uses_state: bool = False
             ) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    current = initial
    _, streams = initialize_episode(initial.width, initial.height, seed, arena_index)
    ticks: list[dict[str, Any]] = []
    decision_diagnostics: list[dict[str, Any]] = []
    candidate_causes: set[str] = set()
    opponent_causes: set[str] = set()
    target_decisions = 0
    avoidable_terminal = False
    locally_dominated = 0
    final_failure_diagnosis: dict[str, Any] | None = None
    while current.result is None:
        actions = [None, None]
        for seat in (0, 1):
            policy = candidate if seat == target_seat else opponent
            actions[seat] = _policy_action(policy, current, seat,
                                           state_input=opponent_uses_state and seat != target_seat)
        transition = step(current, actions)
        candidate_causes.update(transition.causes[target_seat])
        opponent_causes.update(transition.causes[1 - target_seat])

        if transition.causes[target_seat]:
            terminal_actions = _free_actions(current, target_seat)
            terminal_action = actions[target_seat]
            terminal_avoidable = any(
                alternative != terminal_action
                and not step(current, [alternative if seat == target_seat else actions[seat]
                                       for seat in (0, 1)]).causes[target_seat]
                for alternative in terminal_actions)
            terminal_causes = list(transition.causes[target_seat])
            if terminal_avoidable:
                category = "avoidable_decision"
            elif "out_of_bounds" in terminal_causes:
                category = "wall"
            elif not terminal_actions:
                category = "no_safe_move_trapped"
            elif set(terminal_causes) & {"occupied_head", "occupied_trail", "same_destination", "head_swap"}:
                category = "collision_without_safe_alternative"
            else:
                category = "other_terminal_failure"
            final_failure_diagnosis = {
                "category": category,
                "terminal_causes": terminal_causes,
                "safe_actions_before_terminal_move": terminal_actions,
                "reachable_free_cells_before_terminal_move": _reachable_area(
                    current, current.agents[target_seat].position),
                "conditional_avoidable_if_opponent_action_held": terminal_avoidable,
            }

        decision: dict[str, Any] | None = None
        if keep_replay:
            alternatives = _free_actions(current, target_seat)
            selected = actions[target_seat]
            dead = bool(transition.causes[target_seat])
            reachable = _reachable_area(transition.state, transition.state.agents[target_seat].position) if not dead else 0
            safe_after = len(_free_actions(transition.state, target_seat)) if not dead else 0
            avoidable = False
            dominated = False
            for alternative in alternatives:
                if alternative == selected:
                    continue
                counterfactual_actions = list(actions)
                counterfactual_actions[target_seat] = alternative
                counterfactual = step(current, counterfactual_actions)
                if dead:
                    if not counterfactual.causes[target_seat]:
                        avoidable = True
                elif counterfactual.state.result is None and counterfactual.state.agents[target_seat].alive:
                    other_safe = len(_free_actions(counterfactual.state, target_seat))
                    other_area = _reachable_area(counterfactual.state,
                                                 counterfactual.state.agents[target_seat].position)
                    if (other_safe >= safe_after and other_area >= reachable
                            and (other_safe > safe_after or other_area > reachable)):
                        dominated = True
            target_decisions += 1
            avoidable_terminal = avoidable_terminal or (dead and avoidable)
            locally_dominated += int(dominated)
            decision = {
                "tick": current.tick,
                "candidate_position": list(current.agents[target_seat].position),
                "candidate_heading": ACTION_ORDER[current.agents[target_seat].heading],
                "candidate_action": selected,
                "candidate_safe_actions": alternatives,
                "candidate_reachable_cells_before_move": _reachable_area(
                    current, current.agents[target_seat].position),
                "candidate_collision_causes": list(transition.causes[target_seat]),
                "conditional_avoidable_if_opponent_action_held": dead and avoidable,
                "conditional_locally_dominated": dominated,
            }
            decision_diagnostics.append(decision)
        if keep_replay:
            dead = [f"agent_{seat}_dead" for seat, player in enumerate(transition.state.agents)
                    if not player.alive]
            ticks.append({"tick": current.tick,
                          "joint_actions": [a if type(a) is int and a in range(4) else "INVALID" for a in actions],
                          "collision_causes": [list(causes) for causes in transition.causes],
                          "terminal_causes": dead})
        current = transition.state

    result = ("draw" if current.result == "draw" else
              "win" if current.result == f"agent_{target_seat}_win" else "loss")
    row = {
        "root_seed": None,
        "derived_seed_uint64": str(seed),
        "arena_index": arena_index,
        "target_seat": target_seat,
        "result": result,
        "terminal_reason": _terminal_reason(candidate_causes or opponent_causes),
        "terminal_causes": sorted(candidate_causes or opponent_causes),
        "ticks_completed": current.tick,
        "target_decisions": target_decisions,
        "conditional_avoidable_terminal": avoidable_terminal,
        "conditional_locally_dominated_decisions": locally_dominated,
        "failure_diagnosis": final_failure_diagnosis if result == "loss" else None,
    }
    replay = None
    if keep_replay:
        replay = {
            "format": "lc-replay/1", "run_id": run_id,
            "match_id": f"{seed}-{arena_index}-{target_seat}",
            "engine_version": "lc-engine/1", "observation_schema": "lc-observation/1",
            "action_schema": "lc-action/1", "master_seed": str(seed), "arena_index": arena_index,
            "derived_streams": [{"label": stream.label, "initstate": str(stream.initstate),
                                 "initseq": str(stream.initseq)} for stream in streams],
            "agents": [{"agent_id": "candidate" if seat == target_seat else opponent_id,
                        "checkpoint_id": candidate_checkpoint_id if seat == target_seat else opponent_checkpoint_id}
                       for seat in range(2)],
            "initial_state": _wire_state(initial), "ticks": ticks,
            "result": current.result, "tick_count": len(ticks),
        }
        replay["replay_sha256"] = digest_json(replay, "replay_sha256")
    return row, replay, decision_diagnostics


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation requires at least one game")
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    losses = len(rows) - wins - draws
    causes: Counter[str] = Counter(cause for row in rows for cause in row["terminal_causes"])
    reasons: Counter[str] = Counter(row["terminal_reason"] for row in rows)
    durations = [row["ticks_completed"] for row in rows]
    failures = [row for row in rows if row["result"] == "loss" and row.get("failure_diagnosis")]
    failure_categories = Counter(row["failure_diagnosis"]["category"] for row in failures)
    failure_causes = Counter(cause for row in failures
                             for cause in row["failure_diagnosis"]["terminal_causes"])
    failure_categories_by_seat = {
        f"seat-{seat}": dict(sorted(Counter(
            row["failure_diagnosis"]["category"] for row in failures
            if row["target_seat"] == seat).items())) for seat in (0, 1)}
    return {
        "game_count": len(rows), "wins": wins, "draws": draws, "losses": losses,
        "win_rate": wins / len(rows), "wilson_95": wilson_interval(wins, draws, losses),
        "duration_ticks": {"mean": statistics.fmean(durations),
                            "median": statistics.median(durations),
                            "min": min(durations), "max": max(durations)},
        "terminal_reasons": dict(sorted(reasons.items())),
        "terminal_causes": dict(sorted(causes.items())),
        "conditional_counterfactuals": {
            "games_with_avoidable_terminal_decision": sum(
                row["conditional_avoidable_terminal"] for row in rows),
            "locally_dominated_decisions": sum(
                row["conditional_locally_dominated_decisions"] for row in rows),
            "interpretation": "One-step alternatives held the actual opponent action fixed; this is a local diagnostic, not proof of strategic superiority.",
        },
        "failure_diagnostics": {
            "losses_classified": len(failures),
            "categories": dict(sorted(failure_categories.items())),
            "categories_by_seat": failure_categories_by_seat,
            "terminal_causes": dict(sorted(failure_causes.items())),
            "avoidable_final_decisions": sum(
                row["failure_diagnosis"]["conditional_avoidable_if_opponent_action_held"]
                for row in failures),
            "interpretation": "Each loss is classified from its terminal decision. One-step alternatives keep the observed opponent action fixed and do not establish strategic superiority.",
        },
    }


def compare_validation_reference(candidate: dict[str, Any]) -> dict[str, Any]:
    """Newcombe score interval for candidate minus historical G008 win rate.

    Draws count as non-wins, matching the campaign criterion. This is a
    conservative two-independent-proportions comparison even though suites
    share the frozen roster; it does not claim a paired-game test.
    """
    ref = _VALIDATION_REFERENCE
    reference = {"wins": ref["wins"], "draws": ref["draws"], "losses": ref["losses"]}
    candidate_wins = candidate["wins"]
    candidate_n = candidate["wins"] + candidate["draws"] + candidate["losses"]
    reference_n = sum(reference.values())
    if candidate_n != 512 or reference_n != 512:
        raise ValueError("validation comparison requires two frozen 512-game suites")
    p1, p0 = candidate_wins / candidate_n, reference["wins"] / reference_n
    lo1, hi1 = wilson_interval(candidate["wins"], candidate["draws"], candidate["losses"])
    lo0, hi0 = wilson_interval(reference["wins"], reference["draws"], reference["losses"])
    delta = p1 - p0
    lower = delta - math.sqrt((p1 - lo1) ** 2 + (hi0 - p0) ** 2)
    upper = delta + math.sqrt((hi1 - p1) ** 2 + (p0 - lo0) ** 2)
    return {
        "method": "Newcombe-Wilson-score-difference/independent-proportions",
        "reference_run_id": ref["run_id"], "reference_report_sha256": ref["report_sha256"],
        "reference_wins": reference["wins"],
        "reference_draws": reference["draws"], "reference_losses": reference["losses"],
        "candidate_win_rate": p1, "reference_win_rate": p0,
        "win_rate_delta": delta, "difference_95": [lower, upper],
        "noninferiority_margin": _NONINFERIORITY_MARGIN,
        "noninferiority_passed": lower >= -_NONINFERIORITY_MARGIN,
        "limitation": "Draws count as non-wins. The conservative independent-proportions interval does not exploit paired seed outcomes.",
    }


def _sample_key(row: dict[str, Any]) -> str:
    identity = {"root_seed": row["root_seed"], "derived_seed_uint64": row["derived_seed_uint64"],
                "seat": row["target_seat"], "result": row["result"],
                "terminal_reason": row["terminal_reason"]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_replay_samples(items: list[tuple[dict[str, Any], dict[str, Any]]], limit: int = _REPLAY_SAMPLE_LIMIT):
    """Pick one deterministic representative per observed outcome/cause, capped at limit."""
    if limit < 0:
        raise ValueError("sample limit cannot be negative")
    groups: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
    for row, replay in items:
        category = f"{row['result']}:{row['terminal_reason']}"
        key = _sample_key(row)
        previous = groups.get(category)
        if previous is None or key < previous[0]:
            groups[category] = (key, row, replay)
    selected = sorted(groups.values(), key=lambda item: (item[0], item[1]["target_seat"]))[:limit]
    return [(row, replay) for _, row, replay in selected]


def _consider_replay_sample(groups: dict[str, tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
                            row: dict[str, Any], replay: dict[str, Any],
                            diagnostics: list[dict[str, Any]] | None = None) -> None:
    category = f"{row['result']}:{row['terminal_reason']}"
    key = _sample_key(row)
    previous = groups.get(category)
    if previous is None or key < previous[0]:
        groups[category] = (key, row, replay, diagnostics or [])


def _registered_roster(settings: dict[str, Any], seed_plan_path: Path = SEED_PLAN_PATH):
    mode = settings.get("mode")
    if mode not in ("validation", "final"):
        raise ValueError("G008 evaluation mode must be validation or final")
    campaign_v2_validation = (mode == "validation"
                              and settings.get("campaign_id") == _CAMPAIGN_V2_ID)
    if mode == "final":
        from training.eval.g008_final_roster import EXPECTED_FINAL_ROSTER
        expected = EXPECTED_FINAL_ROSTER
    else:
        expected = _EXPECTED_ROSTERS["validation"]
    fields = ("campaign_id", "registration_kind", "run_uuid", "roster_sha256",
              "opponent_code_sha256", "game_count", "seat_order", "seed_plan_sha256",
              "candidate_dataset_source", "candidate_checkpoint_path")
    missing = [key for key in fields if key not in settings]
    if missing:
        raise ValueError(f"registered G008 evaluation config missing: {', '.join(missing)}")
    if (settings["registration_kind"] != mode or settings["game_count"] != 512
            or settings["seat_order"] != [0, 1]
            or settings["roster_sha256"] != expected["roster_sha256"]
            or settings["opponent_code_sha256"] != expected["opponent_code_sha256"]):
        raise ValueError("evaluation config differs from the frozen registered campaign roster")
    if campaign_v2_validation:
        from training.eval.g008_campaign_seed import derive_environment_seed, seed_plan_sha256

        plan_sha = seed_plan_sha256()
        lower, upper = expected["root_range"]
    else:
        from training.eval.provenance import derive_environment_seed

        plan_raw = seed_plan_path.read_bytes()
        plan_sha = hashlib.sha256(plan_raw).hexdigest()
        plan = json.loads(plan_raw)
        partition = plan["partitions"][mode]
        lower, upper = partition["root_seed_range_inclusive"]
    if settings["seed_plan_sha256"] != plan_sha:
        raise ValueError("registered seed-plan hash differs from packaged seed plan")
    if not campaign_v2_validation:
        if ((lower, upper) != expected["root_range"]
                or partition.get("target_seats") != [0, 1]):
            raise ValueError("frozen G008 evaluation seed range/seats changed")
    slots = []
    for root in range(lower, upper + 1):
        for seat in (0, 1):
            seed_tuple = {"split": mode, "domain_label": f"g008/eval/{mode}/initial",
                          "env_root_seed": root, "run_key": expected["run_key"],
                          "update_index": 0, "rollout_index": 0, "arena_index": 0,
                          "game_index": seat, "reset_index": 0}
            _, derived_seed, _ = derive_environment_seed(seed_tuple)
            slots.append((seed_tuple, derived_seed))
    if len(slots) != 512:
        raise ValueError(f"frozen {mode} roster must contain exactly 512 games")
    return expected, slots, plan_sha


def evaluate_registered(settings: dict[str, Any], output_dir: Path,
                        *, seed_plan_path: Path = SEED_PLAN_PATH,
                        policy_loader: Callable | None = None,
                        game_limit: int | None = None,
                        config_file_sha256: str | None = None,
                        settings_sha256: str | None = None,
                        runtime_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute registered G008 512-game evaluation and emit compact artifacts.

    ``game_limit`` exists only for local unit smoke tests; campaign mode rejects
    it so it cannot accidentally produce a partial official result.
    """
    expected, slots, seed_plan_sha = _registered_roster(settings, seed_plan_path)
    if settings.get("registered_execution") is True and (
            runtime_evidence is None
            or not isinstance(runtime_evidence.get("visible_cuda_devices"), list)
            or not runtime_evidence.get("visible_cuda_devices")
            or runtime_evidence["visible_cuda_devices"][0] != "Tesla T4"
            or runtime_evidence.get("visible_cuda_device_count") != len(runtime_evidence["visible_cuda_devices"])
            or runtime_evidence.get("training_device") != "Tesla T4"
            or runtime_evidence.get("training_device_index") != 0
            or runtime_evidence.get("training_cuda_device_count") != 1):
        raise ValueError("registered evaluation requires evidence that cuda:0 is one Tesla T4")
    if game_limit is not None and game_limit != 512 and settings.get("registered_execution") is True:
        raise ValueError("registered G008 evaluations cannot be truncated")
    if game_limit is not None:
        slots = slots[:game_limit]
    if not slots:
        raise ValueError("evaluation roster is empty")
    if settings.get("candidate_dataset_source") != _CANDIDATE_DATASET_SOURCE:
        raise ValueError("candidate checkpoint must come from the predeclared frozen Kaggle dataset")
    candidate_path = Path(settings.get("candidate_checkpoint_path", _CANDIDATE_DATASET_PATH))
    if game_limit is None and candidate_path != _CANDIDATE_DATASET_PATH:
        raise ValueError("registered candidate checkpoint path differs from the frozen Kaggle dataset mount")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate checkpoint missing: {candidate_path}")
    candidate_sha = file_sha256(candidate_path)
    mode = settings["mode"]
    if mode == "validation":
        for key in ("opponent_asset_path", "opponent_checkpoint_sha256"):
            if key not in settings:
                raise ValueError(f"validation config missing frozen opponent field: {key}")
        if settings.get("opponent_id") != expected["opponent_id"]:
            raise ValueError("validation must use the frozen G007 step-300 opponent")
        opponent_asset_path = settings.get("opponent_asset_path", "training/eval/assets/g007-step-300.pt")
        if game_limit is None and opponent_asset_path != "training/eval/assets/g007-step-300.pt":
            raise ValueError("validation opponent path differs from the frozen package asset")
        opponent_path = (Path(settings["opponent_checkpoint_path"]) if game_limit is not None
                         and "opponent_checkpoint_path" in settings
                         else _G007_OPPONENT_PATH)
        if not opponent_path.is_file():
            raise FileNotFoundError(f"frozen G007 checkpoint missing: {opponent_path}")
        opponent_sha = file_sha256(opponent_path)
        if game_limit is None and opponent_sha != expected["opponent_code_sha256"]:
            raise ValueError("mounted G007 step-300 checkpoint differs from the frozen checkpoint identity")
        if settings.get("opponent_checkpoint_sha256", opponent_sha) != opponent_sha:
            raise ValueError("G007 checkpoint hash differs from registered validation config")
        if policy_loader is None:
            from training.ppo.train import load_policy
            policy_loader = load_policy
        candidate, candidate_id = policy_loader(candidate_path, settings.get("device", "cpu"))
        opponent, opponent_id = policy_loader(opponent_path, settings.get("device", "cpu"))
        if (candidate.width, candidate.height) != (12, 12) or (opponent.width, opponent.height) != (12, 12):
            raise ValueError("frozen G008 validation board size must be 12x12")
    else:
        # Final policy identity is only introduced on the registered final execution path.
        if settings.get("opponent_id") != expected["opponent_id"]:
            raise ValueError("final evaluation must bind its frozen opponent identity")
        if policy_loader is None:
            from training.ppo.train import load_policy
            policy_loader = load_policy
        candidate, candidate_id = policy_loader(candidate_path, settings.get("device", "cpu"))
        from training.eval.g008_final_opponent import final_opponent_policy
        opponent = final_opponent_policy()
        opponent_uses_state = True
        opponent_id = expected["opponent_code_sha256"]
        opponent_sha = expected["opponent_code_sha256"]
    if mode == "validation":
        opponent_uses_state = False

    run_id = settings.get("run_id", settings["run_uuid"])
    rows: list[dict[str, Any]] = []
    replay_groups: dict[str, tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = {}
    for seed_tuple, seed in slots:
        seat = seed_tuple["game_index"]
        initial, _ = initialize_episode(12, 12, seed, seed_tuple["arena_index"])
        row, replay, diagnostics = _episode(
            initial, seed, seed_tuple["arena_index"], seat, candidate, opponent,
            run_id=run_id, candidate_checkpoint_id=candidate_id,
            opponent_id=expected["opponent_id"], opponent_checkpoint_id=opponent_id,
            keep_replay=True, opponent_uses_state=opponent_uses_state)
        row["root_seed"] = seed_tuple["env_root_seed"]
        rows.append(row)
        _consider_replay_sample(replay_groups, row, replay, diagnostics)
    if game_limit is None and len(rows) != 512:
        raise ValueError("registered G008 evaluation did not complete all 512 games")

    summary = summarize(rows)
    campaign_v2_validation = (mode == "validation"
                              and settings.get("campaign_id") == _CAMPAIGN_V2_ID)
    if mode == "validation" and len(rows) == 512 and not campaign_v2_validation:
        summary["historical_reference_comparison"] = compare_validation_reference(summary)
        summary["validation_passed"] = summary["historical_reference_comparison"]["noninferiority_passed"]
    elif mode == "final" and len(rows) == 512:
        summary["competitive_target_passed"] = summary["wins"] / 512 >= 0.60
    eligible = (summary.get("validation_passed", False) if mode == "validation"
                else summary.get("competitive_target_passed", False))
    selected_samples = sorted(replay_groups.values(), key=lambda item: (item[0], item[1]["target_seat"]))[:_REPLAY_SAMPLE_LIMIT]
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = output_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    diagnostics_rows = []
    # `prepare()` puts shared schemas at package-root/schemas and omits the
    # repository's training/kaggle subtree from the Kaggle source package.
    replay_schema = json.loads((ROOT / "schemas/lc-replay-1.schema.json").read_text())
    replay_validator = Draft202012Validator(replay_schema)
    for index, (_, row, replay, diagnostics) in enumerate(selected_samples):
        errors = sorted(replay_validator.iter_errors(replay), key=lambda error: list(error.absolute_path))
        if errors:
            raise ValueError(f"generated representative replay violates lc-replay/1: {errors[0].message}")
        name = f"representative-{index:02d}-{row['result']}-{row['terminal_reason']}.json"
        path = replay_dir / name
        write_json(path, replay)
        sample_rows.append({"root_seed": row["root_seed"], "target_seat": row["target_seat"],
                            "result": row["result"], "terminal_reason": row["terminal_reason"],
                            "replay_file": f"replays/{name}", "replay_sha256": replay["replay_sha256"]})
        diagnostics_rows.append({"replay_file": f"replays/{name}", "root_seed": row["root_seed"],
                                 "target_seat": row["target_seat"],
                                 "decisions": diagnostics})

    report = {
        "format": "g008-registered-evaluation/1",
        "status": "complete" if len(rows) == 512 else "partial_smoke",
        "campaign_id": settings["campaign_id"],
        "registration_kind": mode, "mode": mode, "run_id": run_id,
        "run_uuid": settings["run_uuid"], "roster_sha256": settings["roster_sha256"],
        "opponent_code_sha256": settings["opponent_code_sha256"],
        "seed_plan_sha256": seed_plan_sha,
        # Runtime-only overrides (for example device="cuda") must not change
        # the identity of the frozen experiment configuration.
        "run_config_sha256": settings_sha256 or digest_json(settings, "__no_digest_field__"),
        "run_config_file_sha256": config_file_sha256,
        "candidate_checkpoint_path": str(candidate_path),
        "candidate_dataset_source": _CANDIDATE_DATASET_SOURCE,
        "candidate_checkpoint_sha256": candidate_sha, "candidate_model_id": candidate_id,
        "runtime_evidence": runtime_evidence,
        "opponent_id": expected["opponent_id"], "opponent_checkpoint_sha256": opponent_sha,
        "opponent_asset_path": (opponent_asset_path if mode == "validation" else None),
        "opponent_checkpoint_id": opponent_id,
        "game_count": len(rows), "seat_order": [0, 1],
        "summary": summary, "episodes": rows, "representative_replays": sample_rows,
        "representative_decision_diagnostics": diagnostics_rows,
        "replay_sample_limit": _REPLAY_SAMPLE_LIMIT,
        "counterfactual_scope": "Computed during the original evaluation simulation, holding the observed opponent action fixed for one step. Conditional results are diagnostics, not proof of strategic superiority.",
    }
    # V2 validation is descriptive evidence for the preregistered comparison
    # between its own candidates. The historical 244-loss report belongs to
    # the v1 selection rule and must not leak into v2 decisions. Final v2
    # evaluation still reports the frozen competitive success criterion.
    if not campaign_v2_validation:
        report["promotion_eligible"] = bool(eligible and len(rows) == 512)
    write_json(output_dir / "g008-evaluation.json", report)
    files = [{"path": "g008-evaluation.json", "sha256": file_sha256(output_dir / "g008-evaluation.json")}]
    runtime_path = output_dir / "kaggle-execution-evidence.json"
    if runtime_evidence is not None:
        write_json(runtime_path, runtime_evidence)
        files.append({"path": runtime_path.name, "sha256": file_sha256(runtime_path)})
    files.extend({"path": row["replay_file"], "sha256": file_sha256(output_dir / row["replay_file"])}
                 for row in sample_rows)
    manifest = {"format": "g008-evaluation-artifacts/1", "run_uuid": settings["run_uuid"],
                "run_config_sha256": report["run_config_sha256"],
                "run_config_file_sha256": config_file_sha256, "files": files}
    write_json(output_dir / "g008-evaluation-manifest.json", manifest,
               "manifest_sha256")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="registered Kaggle run-config JSON containing g008_evaluation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config_bytes = args.config.read_bytes()
    settings = json.loads(config_bytes)["g008_evaluation"]
    report = evaluate_registered(settings, args.output,
                                  config_file_sha256=hashlib.sha256(config_bytes).hexdigest())
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
