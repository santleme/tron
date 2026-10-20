import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from training.arena.engine import observe
from training.eval.lc_eval import evaluate, open_space_action, wilson_interval
from training.arena.engine import initialize_episode
from training.ppo.protocol import digest_json


def straight_policy(observation):
    # Find self-head heading planes in the fixed HWC observation.
    for row in observation:
        for cell in row:
            if cell[0]:
                return next(i for i in range(4) if cell[4 + i])
    return [0.0, 0.0, 0.0, 0.0]


class EvaluationProtocolTests(unittest.TestCase):
    def test_wilson_uses_wins_over_all_games_and_excludes_draws_from_wins(self):
        interval = wilson_interval(16, 32, 16)
        self.assertEqual(len(interval), 2)
        self.assertTrue(0 < interval[0] < 0.25 < interval[1] < 1)
        self.assertAlmostEqual(wilson_interval(64, 0, 0)[0], 0.9433759402071946)
        self.assertEqual(wilson_interval(64, 0, 0)[1], 1.0)

    def test_open_space_chooses_legal_action_with_action_order_tie_break(self):
        state, _ = initialize_episode(8, 6, 5, 0)
        action = open_space_action(state, 0)
        self.assertIn(action, range(4))
        self.assertNotEqual(action, (state.agents[0].heading + 2) % 4)

    def test_evaluate_emits_exact_complete_reports_and_replay_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate(straight_policy, straight_policy, "test-run", "target", directory, 6, 6)
            self.assertEqual(report["held_out_seeds"], [str(i) for i in range(32)])
            self.assertEqual({m["opponent_id"] for m in report["matchups"]}, {"open-space", "step-0"})
            expected = {(str(seed), seat) for seed in range(32) for seat in (0, 1)}
            for matchup in report["matchups"]:
                self.assertEqual({(g["seed"], g["target_seat"]) for g in matchup["games"]}, expected)
                self.assertEqual(sum(matchup["summary"][key] for key in ("wins", "draws", "losses")), 64)
                for game in matchup["games"]:
                    replay = json.loads((Path(directory) / game["replay_file"].replace("evaluation/", "")).read_text())
                    self.assertEqual(replay["tick_count"], len(replay["ticks"]))
            schema = json.loads((Path(__file__).resolve().parents[2] / "schemas/lc-eval-1.schema.json").read_text())
            Draft202012Validator(schema).validate(report)
            replay_schema = json.loads((Path(__file__).resolve().parents[2] / "schemas/lc-replay-1.schema.json").read_text())
            sample = json.loads((Path(directory) / "replays/open-space-00-0.json").read_text())
            Draft202012Validator(replay_schema).validate(sample)
            self.assertEqual(sample["replay_sha256"], digest_json(sample, "replay_sha256"))


if __name__ == "__main__":
    unittest.main()
