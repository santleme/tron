import hashlib
import json

import pytest

from training.ppo.protocol import digest_json
from training.eval.g008_validation import (
    _EXPECTED_ROSTERS,
    _policy_action,
    compare_validation_reference,
    evaluate_registered,
    select_replay_samples,
    summarize,
    wilson_interval,
)


def test_wilson_interval_and_validation_reference_are_reproducible():
    interval = wilson_interval(307, 51, 154)
    assert interval == pytest.approx([0.5568, 0.6412], abs=0.001)
    comparison = compare_validation_reference({"wins": 307, "draws": 51, "losses": 154})
    assert comparison["reference_win_rate"] == pytest.approx(217 / 512)
    assert comparison["win_rate_delta"] == pytest.approx(90 / 512)
    assert comparison["method"] == "Newcombe-Wilson-score-difference/independent-proportions"
    assert comparison["noninferiority_margin"] == pytest.approx(0.065)
    assert comparison["noninferiority_passed"] is True


def test_summary_counts_draws_as_nonwins_and_aggregates_diagnostics():
    rows = [
        {"result": "win", "terminal_causes": [], "terminal_reason": "other",
         "ticks_completed": 15, "conditional_avoidable_terminal": False,
         "conditional_locally_dominated_decisions": 2},
        {"result": "draw", "terminal_causes": ["same_destination"], "terminal_reason": "collision",
         "ticks_completed": 10, "conditional_avoidable_terminal": False,
         "conditional_locally_dominated_decisions": 0},
        {"result": "loss", "terminal_causes": ["occupied_trail"], "terminal_reason": "collision",
         "ticks_completed": 8, "conditional_avoidable_terminal": True,
         "conditional_locally_dominated_decisions": 1,
         "target_seat": 0,
         "failure_diagnosis": {"category": "avoidable_decision",
                               "terminal_causes": ["occupied_trail"],
                               "conditional_avoidable_if_opponent_action_held": True}},
    ]
    result = summarize(rows)
    assert (result["wins"], result["draws"], result["losses"]) == (1, 1, 1)
    assert result["win_rate"] == pytest.approx(1 / 3)
    assert result["terminal_causes"] == {"occupied_trail": 1, "same_destination": 1}
    assert result["conditional_counterfactuals"]["games_with_avoidable_terminal_decision"] == 1
    assert "not proof of strategic superiority" in result["conditional_counterfactuals"]["interpretation"]
    assert result["failure_diagnostics"]["losses_classified"] == 1
    assert result["failure_diagnostics"]["categories"] == {"avoidable_decision": 1}


def test_failure_diagnostics_summarizes_report_without_replaying_games():
    from training.eval.g008_failure_diagnostics import summarize_report

    episodes = [{"result": "loss", "target_seat": 1,
                 "failure_diagnosis": {"category": "wall", "terminal_causes": ["out_of_bounds"],
                                       "conditional_avoidable_if_opponent_action_held": False}}
                for _ in range(2)]
    episodes.extend({"result": "win", "target_seat": 0} for _ in range(510))
    result = summarize_report({"format": "g008-registered-evaluation/1", "run_uuid": "run",
                               "opponent_id": "g007-step-300", "candidate_checkpoint_sha256": "a" * 64,
                               "episodes": episodes, "summary": {"losses": 2}})
    assert result["losses_classified"] == 2
    assert result["categories_by_seat"]["seat-1"] == {"wall": 2}
    assert result["terminal_causes"] == {"out_of_bounds": 2}


def test_replay_sample_is_deterministic_and_bounded():
    items = []
    for seed in range(10):
        row = {"root_seed": seed, "derived_seed_uint64": seed + 100,
               "target_seat": seed % 2, "result": "loss" if seed % 2 else "win",
               "terminal_reason": "collision"}
        items.append((row, {"sample": seed}))
    forward = select_replay_samples(items, limit=2)
    backward = select_replay_samples(list(reversed(items)), limit=2)
    assert forward == backward
    assert len(forward) == 2


def test_state_based_opponent_adapter_passes_state_and_seat():
    state = object()
    seen = []

    def opponent(received_state, seat):
        seen.append((received_state, seat))
        return 2

    assert _policy_action(opponent, state, 1, state_input=True) == 2
    assert seen == [(state, 1)]


def test_policy_can_return_open_space_invalid_terminal_action():
    assert _policy_action(lambda _state, _seat: "INVALID", object(), 0,
                          state_input=True) == "INVALID"


@pytest.mark.parametrize(("campaign_id", "uses_v1_selection_fields"), [
    ("g008-competitive-20261020-v1", True),
    ("g008-competitive-20261021-v2", False),
])
def test_tiny_smoke_evaluates_games_and_writes_metrics_and_sampled_replay(
        tmp_path, campaign_id, uses_v1_selection_fields):
    seed_plan = tmp_path / "seed-plan.json"
    seed_plan.write_bytes((__import__("pathlib").Path(__file__).parents[2]
                           / "training/configs/g008-split-seed-plan.json").read_bytes())
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"test-candidate")
    opponent = tmp_path / "opponent.pt"
    opponent.write_bytes(b"test-opponent")

    class Policy:
        width = height = 12

        def __call__(self, _observation):
            return 0

    policy = Policy()

    def loader(_path, _device):
        return policy, "test-policy"

    plan_sha = hashlib.sha256(seed_plan.read_bytes()).hexdigest()
    roster = _EXPECTED_ROSTERS["validation"]
    settings = {
        "campaign_id": campaign_id,
        "registration_kind": "validation", "mode": "validation",
        "run_id": "tiny-smoke", "run_uuid": "tiny-smoke-uuid",
        "roster_sha256": roster["roster_sha256"],
        "opponent_code_sha256": roster["opponent_code_sha256"],
        "game_count": 512, "seat_order": [0, 1], "seed_plan_sha256": plan_sha,
        "candidate_dataset_source": "pascalcoimbra/g008-frozen-candidate",
        "candidate_checkpoint_path": str(checkpoint),
        "opponent_id": "g007-step-300", "opponent_checkpoint_path": str(opponent),
        "opponent_asset_path": "training/eval/assets/g007-step-300.pt",
        "opponent_checkpoint_sha256": hashlib.sha256(opponent.read_bytes()).hexdigest(),
        "device": "cpu",
    }
    frozen_settings_sha = digest_json({key: value for key, value in settings.items()
                                       if key != "device"}, "__no_digest_field__")
    report = evaluate_registered(settings, tmp_path / "out", seed_plan_path=seed_plan,
                                 policy_loader=loader, game_limit=2,
                                 settings_sha256=frozen_settings_sha)
    assert report["game_count"] == 2
    assert len(report["episodes"]) == 2
    assert len(report["representative_replays"]) <= 2
    assert report["candidate_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["roster_sha256"] == roster["roster_sha256"]
    assert report["opponent_code_sha256"] == roster["opponent_code_sha256"]
    assert report["opponent_checkpoint_sha256"] == hashlib.sha256(opponent.read_bytes()).hexdigest()
    assert set(("wins", "draws", "losses", "win_rate", "wilson_95")) <= set(report["summary"])
    assert ("promotion_eligible" in report) is uses_v1_selection_fields
    assert ("historical_reference_comparison" in report["summary"]) is False
    assert ("validation_passed" in report["summary"]) is False
    assert report["run_config_sha256"] == frozen_settings_sha
    saved = json.loads((tmp_path / "out/g008-evaluation.json").read_text())
    manifest = json.loads((tmp_path / "out/g008-evaluation-manifest.json").read_text())
    assert saved["summary"]["game_count"] == 2
    assert ("promotion_eligible" in saved) is uses_v1_selection_fields
    assert saved["summary"]["failure_diagnostics"]["losses_classified"] == saved["summary"]["losses"]
    assert all(row["failure_diagnosis"] is None or row["result"] == "loss"
               for row in saved["episodes"])
    assert manifest["run_config_sha256"] == saved["run_config_sha256"]
    assert len(list((tmp_path / "out/replays").glob("*.json"))) <= 2
    assert saved["representative_decision_diagnostics"]
    replay_schema = json.loads((__import__("pathlib").Path(__file__).parents[2]
                                / "training/kaggle/schemas/lc-replay-1.schema.json").read_text())
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(replay_schema)
    for path in (tmp_path / "out/replays").glob("*.json"):
        assert validator.is_valid(json.loads(path.read_text()))
        assert all("candidate_decision" not in tick for tick in json.loads(path.read_text())["ticks"])


def test_final_roster_identity_mismatch_is_rejected(tmp_path):
    from training.eval.g008_final_roster import EXPECTED_FINAL_ROSTER

    roster = _EXPECTED_ROSTERS["validation"]
    final = EXPECTED_FINAL_ROSTER
    settings = {"mode": "final", "registration_kind": "final", "campaign_id": "campaign",
                "run_uuid": "uuid", "roster_sha256": roster["roster_sha256"],
                "opponent_code_sha256": final["opponent_code_sha256"], "game_count": 512,
                "seat_order": [0, 1], "seed_plan_sha256": "0" * 64,
                "candidate_dataset_source": "pascalcoimbra/g008-frozen-candidate",
                "candidate_checkpoint_path": "/kaggle/input/g008-frozen-candidate/candidate.pt"}
    with pytest.raises(ValueError, match="frozen registered campaign roster"):
        from training.eval.g008_validation import _registered_roster
        _registered_roster(settings)
