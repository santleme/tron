import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from training.eval.provenance import (
    ProvenanceError,
    SEED_TUPLE_FIELDS,
    build_benchmark_reservation_roster,
    benchmark_run_keys,
    build_benchmark_run_schedules,
    build_schedule_aggregate,
    build_expanded_schedule,
    build_training_run_schedule,
    collision_scan,
    derive_environment_seed,
    encode_optional_uint64,
    _encode_string,
    schedule_spec_digest,
    validate_realization,
    validate_schedule_binding,
)


def test_training_run_schedule_freezes_every_initial_and_reset_slot():
    from training.ppo.train import G008EnvironmentSeedSchedule

    run_id = "12345678-1234-4234-8234-123456789abc"
    schedule = build_training_run_schedule(run_id=run_id, root_seed=2_000_017,
                                          arena_count=2, total_updates=3, rollout_steps=4)
    assert schedule["suite_id"] == "g008-train"
    assert schedule["run_key"] == f"train-{run_id}"
    assert len(schedule["slots"]) == 2 * 3 * 5
    adapter = G008EnvironmentSeedSchedule(schedule, run_id=run_id, arena_count=2,
                                          total_updates=3, rollout_steps=4)
    assert adapter.seed_for(update=2, arena=1, reset_index=0) > 0
    assert adapter.seed_for(update=2, arena=1, reset_index=4) > 0


def test_training_run_schedule_rejects_out_of_partition_and_oversized_rosters():
    run_id = "12345678-1234-4234-8234-123456789abc"
    with pytest.raises(ProvenanceError, match="outside the frozen train partition"):
        build_training_run_schedule(run_id=run_id, root_seed=1_999_999,
                                    arena_count=1, total_updates=1, rollout_steps=1)
    with pytest.raises(ProvenanceError, match="one-million-slot"):
        build_training_run_schedule(run_id=run_id, root_seed=2_000_000,
                                    arena_count=1_001, total_updates=1, rollout_steps=1_000)


def _tuple(root=5_000_000, arena=0):
    return {
        "split": "benchmark",
        "domain_label": "g008/benchmark/measured/initial",
        "env_root_seed": root,
        "run_key": "pair-00",
        "update_index": 0,
        "rollout_index": 0,
        "arena_index": arena,
        "game_index": 0,
        "reset_index": 0,
    }


def _slot(seed_tuple, backend="shared", suite_ordinal=2, root_ordinal=0):
    return {
        **seed_tuple,
        "suite_ordinal": suite_ordinal,
        "root_ordinal": root_ordinal,
        "status": "reserved",
        "protocol_exclusion_reason": "",
        "backend_id": backend,
    }


def _expanded(slots):
    return build_expanded_schedule(suite_id="benchmark-v1", run_id="run-abc",
                                   run_key=slots[0]["run_key"], slots=slots)


class G008SeedCodecTests(unittest.TestCase):
    def test_golden_vector_matches_frozen_plan_exactly(self):
        plan_path = Path(__file__).resolve().parents[2] / "training/configs/g008-split-seed-plan.json"
        vector = json.loads(plan_path.read_text())["environment_seed_codec"]["golden_vector"]
        seed_tuple = dict(zip(SEED_TUPLE_FIELDS, vector["tuple"]))
        digest, derived, payload = derive_environment_seed(seed_tuple)
        self.assertEqual(payload.hex(), vector["payload_hex"])
        self.assertEqual(digest, vector["sha256"])
        self.assertEqual(derived, vector["derived_seed_uint64"])

    def test_split_root_run_key_and_domain_mismatches_fail_closed(self):
        invalid = _tuple(root=5_000_005)
        invalid["run_key"] = "pair-05"
        with self.assertRaisesRegex(ProvenanceError, "measured benchmark root/run_key"):
            derive_environment_seed(invalid)
        invalid = _tuple()
        invalid["split"] = "imitation_demo"
        invalid["domain_label"] = "g008/imitation_demo/initial"
        invalid["env_root_seed"] = 3_000_000
        run_id = "12345678-1234-4234-8234-123456789abc"
        invalid["run_key"] = f"imitation-{run_id}"
        initial_digest = derive_environment_seed(invalid)[0]
        invalid["domain_label"] = "g008/imitation_demo/reset"
        reset_digest = derive_environment_seed(invalid)[0]
        self.assertNotEqual(initial_digest, reset_digest)
        invalid["domain_label"] = "g008/train/initial"
        with self.assertRaisesRegex(ProvenanceError, "domain label does not match"):
            derive_environment_seed(invalid)
        invalid["domain_label"] = "g008/imitation_demo/initial"
        invalid["run_key"] = "imitation-12345678-1234-0000-8234-123456789abc"
        with self.assertRaisesRegex(ProvenanceError, "lowercase canonical UUID"):
            derive_environment_seed(invalid)

    def test_imitation_schedule_binds_run_key_to_registered_run_id(self):
        run_id = "12345678-1234-4234-8234-123456789abc"
        imitation = _tuple()
        imitation.update({
            "split": "imitation_demo",
            "domain_label": "g008/imitation_demo/initial",
            "env_root_seed": 3_000_000,
            "run_key": f"imitation-{run_id}",
        })
        slot = _slot(imitation, suite_ordinal=1, root_ordinal=0)
        expanded = build_expanded_schedule(
            suite_id="g008-imitation-demo", run_id=run_id,
            run_key=f"imitation-{run_id}", slots=[slot])
        self.assertEqual(expanded["slots"][0]["domain_label"], "g008/imitation_demo/initial")

        mismatched = "12345678-1234-4234-8234-123456789abd"
        with self.assertRaisesRegex(ProvenanceError, "bind exactly to run_id"):
            build_expanded_schedule(
                suite_id="g008-imitation-demo", run_id=run_id,
                run_key=f"imitation-{mismatched}", slots=[{**slot, "run_key": f"imitation-{mismatched}"}])
        invalid = _tuple()
        invalid["domain_label"] = "g008/eval/final/initial"
        with self.assertRaisesRegex(ProvenanceError, "domain label does not match"):
            derive_environment_seed(invalid)

    def test_nfc_encoding_domain_separation_and_optional_na_sentinel(self):
        self.assertEqual(_encode_string("caf\u00e9"), _encode_string("cafe\u0301"))
        composed = _tuple()
        other_domain = copy.deepcopy(composed)
        other_domain["domain_label"] = "g008/benchmark/measured/reset"
        self.assertNotEqual(derive_environment_seed(composed)[0], derive_environment_seed(other_domain)[0])
        self.assertEqual(encode_optional_uint64(None).hex(), "ffffffffffffffff")
        self.assertEqual(encode_optional_uint64(0).hex(), "0000000000000000")

    def test_benchmark_schedule_spec_digest_is_stable_and_bound_to_plan(self):
        first = schedule_spec_digest()
        self.assertEqual(first, schedule_spec_digest())
        self.assertEqual(first, "9b3878688d8494211cbde92c7f671653239956a97161263277e0db19b2e7ffec")
        self.assertEqual(len(first), 64)
        with self.assertRaisesRegex(ProvenanceError, "stale seed-plan"):
            schedule_spec_digest("0" * 64)


class G008ScheduleTests(unittest.TestCase):
    def test_seed_plan_freezes_measured_warmup_and_unused_benchmark_ranges(self):
        plan_path = Path(__file__).resolve().parents[2] / "training/configs/g008-split-seed-plan.json"
        plan = json.loads(plan_path.read_text())
        benchmark = plan["partitions"]["benchmark"]
        self.assertEqual(benchmark["root_seed_range_inclusive"], [5_000_000, 5_000_271])
        self.assertEqual([row["root_seed"] for row in benchmark["measured_roots"]],
                         [5_000_000, 5_000_001, 5_000_002, 5_000_003, 5_000_004])
        self.assertEqual(benchmark["reserved_unused_root_seed_range_inclusive"], [5_000_005, 5_000_255])
        warmups = [root for assignment in benchmark["warmup_assignments"] for root in assignment["root_seeds"]]
        self.assertEqual(warmups, list(range(5_000_256, 5_000_272)))

    def test_aggregate_maps_ordered_individual_run_schedule_digests(self):
        keys = ("pair-00", "pair-01")
        run_ids = {key: f"benchmark-run-{index:02d}" for index, key in enumerate(keys)}
        schedules = {}
        realized = {}
        for index, key in enumerate(keys):
            run_id = run_ids[key]
            seed_tuple = _tuple(root=5_000_000 + index)
            seed_tuple["run_key"] = key
            schedule = build_expanded_schedule(suite_id="benchmark-v1", run_id=run_id,
                                               run_key=key, slots=[_slot(seed_tuple)])
            schedules[run_id] = schedule
            realized[run_id] = validate_realization(
                schedule, realized_slots=[{"slot_id": schedule["slots"][0]["slot_id"],
                                           "status": "terminal-unreachable"}],
                realized_tuple_list=[], episode_outcomes=[])
        aggregate = build_schedule_aggregate(expanded_schedules=schedules, realized_schedules=realized)
        self.assertEqual(aggregate["schedule_spec_digest"], schedules[run_ids[keys[0]]]["schedule_spec_digest"])
        self.assertEqual([entry["run_id"] for entry in aggregate["runs"]], list(run_ids.values()))
        self.assertEqual([entry["expanded_schedule_digest"] for entry in aggregate["runs"]],
                         [schedules[run_id]["expanded_schedule_digest"] for run_id in run_ids.values()])

    def test_exact_paired_tuple_reuse_is_the_only_allowed_duplicate(self):
        pair = [_slot(_tuple(), "incumbent"), _slot(_tuple(), "candidate")]
        expanded = _expanded(pair)
        self.assertTrue(expanded["collision_scan"]["passed"])
        self.assertEqual(expanded["collision_scan"]["total_slots"], 2)
        self.assertEqual(len(expanded["collision_scan"]["duplicate_groups"]), 1)
        self.assertEqual({member["backend_id"] for member in
                          expanded["collision_scan"]["duplicate_groups"][0]["members"]},
                         {"incumbent", "candidate"})

    def test_collision_scan_rejects_distinct_tuple_collision_and_sorts_groups(self):
        slots = [_slot(_tuple(), "shared"), _slot(_tuple(root=5_000_001), "shared", root_ordinal=1)]
        for index, slot in enumerate(slots):
            slot["derived_seed_uint64"] = 7
            slot["slot_id"] = f"{index:064x}"
        with patch("training.eval.provenance.derive_environment_seed", side_effect=[
                ("a" * 64, 7, b"a"), ("b" * 64, 7, b"b")]):
            result = collision_scan(slots)
        self.assertFalse(result["passed"])
        self.assertEqual([g["derived_seed_uint64"] for g in result["duplicate_groups"]], [7])
        self.assertEqual(result["duplicate_groups"][0]["members"], sorted(
            result["duplicate_groups"][0]["members"], key=lambda member: member["slot_id"].encode()))

    def test_expanded_digest_is_invariant_and_sensitive_to_roots_and_preregistration(self):
        slots = [_slot(_tuple(), root_ordinal=0), _slot(_tuple(arena=1), root_ordinal=0)]
        baseline = _expanded(slots)
        self.assertEqual(baseline["expanded_schedule_digest"], _expanded(copy.deepcopy(slots))["expanded_schedule_digest"])
        self.assertEqual(baseline["expanded_schedule_digest"], _expanded(list(reversed(slots)))["expanded_schedule_digest"])
        changed_root = copy.deepcopy(slots)
        for slot in changed_root:
            slot["env_root_seed"] += 1
            slot["run_key"] = "pair-01"
        self.assertNotEqual(baseline["expanded_schedule_digest"], _expanded(changed_root)["expanded_schedule_digest"])
        changed_status = copy.deepcopy(slots)
        changed_status[1]["status"] = "protocol-excluded"
        changed_status[1]["protocol_exclusion_reason"] = "frozen exclusion"
        self.assertNotEqual(baseline["expanded_schedule_digest"], _expanded(changed_status)["expanded_schedule_digest"])
        self.assertNotEqual(baseline["expanded_schedule_digest"], build_expanded_schedule(
            suite_id="benchmark-v1", run_id="run-other", run_key="pair-00", slots=slots)["expanded_schedule_digest"])
        reordered = copy.deepcopy(slots)
        reordered[1]["root_ordinal"] = 1
        self.assertNotEqual(baseline["expanded_schedule_digest"], _expanded(reordered)["expanded_schedule_digest"])

    def test_realized_outcomes_are_order_invariant_but_change_digest(self):
        expanded = _expanded([_slot(_tuple(), root_ordinal=0),
                              _slot(_tuple(arena=1), root_ordinal=0)])
        slots = expanded["slots"]
        statuses = [{"slot_id": slot["slot_id"], "status": "consumed"} for slot in slots]
        tuples = [{key: slot[key] for key in SEED_TUPLE_FIELDS} for slot in slots]
        outcomes = [
            {"initial_slot_id": slot["slot_id"], "seat0_agent_id": "a", "seat1_agent_id": "b",
             "result": "seat0_win", "terminal_reason": "collision", "ticks_completed": 10}
            for slot in slots
        ]
        first = validate_realization(expanded, realized_slots=statuses,
                                     realized_tuple_list=tuples, episode_outcomes=outcomes)
        shuffled = validate_realization(expanded, realized_slots=list(reversed(statuses)),
                                       realized_tuple_list=tuples, episode_outcomes=list(reversed(outcomes)))
        self.assertEqual(first["realized_schedule_digest"], shuffled["realized_schedule_digest"])
        self.assertEqual(first["expanded_schedule_digest"], expanded["expanded_schedule_digest"])
        changed = copy.deepcopy(outcomes)
        changed[0]["result"] = "draw"
        modified = validate_realization(expanded, realized_slots=statuses,
                                        realized_tuple_list=tuples, episode_outcomes=changed)
        self.assertEqual(modified["expanded_schedule_digest"], first["expanded_schedule_digest"])
        self.assertNotEqual(modified["realized_schedule_digest"], first["realized_schedule_digest"])

    def test_schedule_binding_rejects_missing_stale_and_wrong_parent_digests(self):
        expanded = _expanded([_slot(_tuple())])
        slot = expanded["slots"][0]
        realized = validate_realization(
            expanded,
            realized_slots=[{"slot_id": slot["slot_id"], "status": "consumed"}],
            realized_tuple_list=[{key: slot[key] for key in SEED_TUPLE_FIELDS}],
            episode_outcomes=[{"initial_slot_id": slot["slot_id"], "seat0_agent_id": "a",
                               "seat1_agent_id": "b", "result": "draw",
                               "terminal_reason": "timeout", "ticks_completed": 32}],
        )
        from training.eval.provenance import seed_plan_sha256
        binding = {"seed_plan_sha256": seed_plan_sha256(),
                   "schedule_spec_digest": expanded["schedule_spec_digest"],
                   "expanded_schedule_digest": expanded["expanded_schedule_digest"],
                   "realized_schedule_digest": realized["realized_schedule_digest"]}
        validate_schedule_binding(binding, expanded=expanded, realized=realized)
        bad = dict(binding)
        bad["expanded_schedule_digest"] = "0" * 64
        with self.assertRaisesRegex(ProvenanceError, "expanded digest"):
            validate_schedule_binding(bad, expanded=expanded, realized=realized)
        bad = dict(binding)
        bad["seed_plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(ProvenanceError, "stale seed-plan"):
            validate_schedule_binding(bad, expanded=expanded, realized=realized)
        bad = dict(binding)
        bad["schedule_spec_digest"] = "0" * 64
        with self.assertRaisesRegex(ProvenanceError, "stale"):
            validate_schedule_binding(bad, expanded=expanded, realized=realized)
        with self.assertRaisesRegex(ProvenanceError, "four exact digest fields"):
            validate_schedule_binding({key: value for key, value in binding.items() if key != "realized_schedule_digest"},
                                      expanded=expanded, realized=realized)

    def test_realized_schedule_rejects_missing_duplicate_bad_and_unfinished_records(self):
        expanded = _expanded([_slot(_tuple())])
        slot = expanded["slots"][0]
        record = {"slot_id": slot["slot_id"], "status": "consumed"}
        tuple_item = {key: slot[key] for key in SEED_TUPLE_FIELDS}
        outcome = {"initial_slot_id": slot["slot_id"], "seat0_agent_id": "a", "seat1_agent_id": "b",
                   "result": "invalid", "terminal_reason": "other", "ticks_completed": 1}
        args = {"realized_slots": [record], "realized_tuple_list": [tuple_item], "episode_outcomes": [outcome]}
        for altered in (
            {**args, "realized_slots": []},
            {**args, "realized_slots": [record, record]},
            {**args, "episode_outcomes": [{**outcome, "extra": 1}]},
            {**args, "episode_outcomes": [{**outcome, "result": "invalid", "terminal_reason": "wall"}]},
            {**args, "realized_slots": [{"slot_id": slot["slot_id"], "status": "run-aborted-before-reach"}]},
        ):
            with self.subTest(altered=altered), self.assertRaises(ProvenanceError):
                validate_realization(expanded, **altered)


if __name__ == "__main__":
    unittest.main()
