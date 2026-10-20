import importlib.util
import copy
import uuid
import tempfile
import unittest

import pytest

from training.ppo.train import TrainConfig, train


def test_contiguous_observation_batch_matches_public_observation_contract():
    import numpy as np
    from training.arena.engine import initialize_episode, observe, step
    from training.ppo.train import _observation_batch

    states = [initialize_episode(12, 12, 41, index)[0] for index in range(3)]
    # Extend trails so the test covers non-head occupied cells as well as heads.
    for index in range(3):
        state = states[index]
        for _ in range(3):
            if state.result is not None:
                break
            state = step(state, tuple(agent.heading for agent in state.agents)).state
        states[index] = state
    seats = [0, 1, 0]
    actual = _observation_batch(states, seats)
    expected = [observe(state, seat) for state, seat in zip(states, seats)]
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    assert actual.tolist() == expected


def test_incremental_rollout_cache_matches_rebuilt_observations_and_occupancy():
    from training.arena.engine import initialize_episode, step
    from training.ppo.train import (_observation_batch, _observation_cache,
                                    _update_observation_cache)

    states = [initialize_episode(8, 8, 73, index)[0] for index in range(4)]
    cache = _observation_cache(states)
    occupied = [{cell for agent in state.agents for cell in agent.trail} for state in states]
    for tick in range(12):
        previous = states
        states = []
        for arena, state in enumerate(previous):
            actions = tuple(agent.heading for agent in state.agents)
            expected = step(state, actions)
            actual = step(state, actions, occupied_cells=occupied[arena])
            assert actual == expected
            for old_agent, new_agent in zip(state.agents, actual.state.agents):
                if old_agent.position != new_agent.position:
                    occupied[arena].add(new_agent.position)
            if actual.state.result is None:
                states.append(actual.state)
            else:
                reset = initialize_episode(8, 8, 9_000 + tick, arena)[0]
                states.append(reset)
                occupied[arena] = {cell for agent in reset.agents for cell in agent.trail}
        _update_observation_cache(cache, previous, states)
        expected_batch = _observation_batch([state for state in states for _ in range(2)],
                                           [seat for _ in states for seat in range(2)])
        assert cache.reshape(expected_batch.shape).tolist() == expected_batch.tolist()


def test_heuristic_training_requires_frozen_g008_training_schedule(tmp_path):
    config = TrainConfig(device="cpu", heuristic_opponent_probability=0.5)
    with pytest.raises(ValueError, match="frozen g008-train schedule"):
        train(config, tmp_path / "heuristic-run", export_onnx=False,
              run_evaluation=False)


def test_g008_training_cannot_call_g007_open_space_evaluator(tmp_path):
    run_id = "12345678-1234-4234-8234-123456789abc"
    config = TrainConfig(width=6, height=6, arena_count=1, rollout_steps=1,
                         total_updates=1, update_epochs=1, minibatch_size=4,
                         hidden_size=8, device="cpu")
    with pytest.raises(ValueError, match="must not invoke the G007 open-space evaluator"):
        train(config, tmp_path / "g008-run", g008_schedule=_g008_train_schedule(
            arena_count=1, reset_indices=(1,), run_id=run_id), run_id=run_id)


def _g008_train_schedule(arena_count=1, reset_indices=(1,), run_id="12345678-1234-4234-8234-123456789abc"):
    from training.eval.provenance import build_expanded_schedule

    slots = []
    for arena in range(arena_count):
        for reset in (0, *reset_indices):
            slots.append({
                "split": "train",
                "domain_label": "g008/train/initial" if reset == 0 else "g008/train/reset",
                "env_root_seed": 2_000_000 + arena,
                "run_key": f"train-{run_id}",
                "update_index": 0,
                "rollout_index": 0,
                "arena_index": arena,
                "game_index": 0,
                "reset_index": reset,
                "suite_ordinal": 0,
                "root_ordinal": arena,
                "status": "reserved",
                "protocol_exclusion_reason": "",
            })
    return build_expanded_schedule(suite_id="g008-train", run_id=run_id,
                                   run_key=f"train-{run_id}", slots=slots)


def test_g008_schedule_binds_initial_and_reset_seed_domains_reproducibly():
    from training.arena.engine import initialize_episode
    from training.eval.provenance import derive_environment_seed
    from training.ppo.train import G008EnvironmentSeedSchedule

    run_id = "12345678-1234-4234-8234-123456789abc"
    expanded = _g008_train_schedule(run_id=run_id)
    schedule = G008EnvironmentSeedSchedule(expanded, run_id=run_id, arena_count=1, total_updates=1)
    initial_seed = schedule.seed_for(update=0, arena=0, reset_index=0)
    reset_seed = schedule.seed_for(update=0, arena=0, reset_index=1)
    initial_slot, reset_slot = expanded["slots"]
    assert initial_seed == derive_environment_seed({key: initial_slot[key] for key in (
        "split", "domain_label", "env_root_seed", "run_key", "update_index", "rollout_index",
        "arena_index", "game_index", "reset_index")})[1]
    assert reset_seed == derive_environment_seed({key: reset_slot[key] for key in (
        "split", "domain_label", "env_root_seed", "run_key", "update_index", "rollout_index",
        "arena_index", "game_index", "reset_index")})[1]
    assert initial_seed != reset_seed
    assert initialize_episode(5, 5, initial_seed, 0)[0] == initialize_episode(5, 5, initial_seed, 0)[0]
    assert initialize_episode(5, 5, reset_seed, 0)[0] == initialize_episode(5, 5, reset_seed, 0)[0]


def test_g008_schedule_fails_closed_for_missing_stale_wrong_run_and_unreserved_slots():
    from training.eval.provenance import ProvenanceError, build_expanded_schedule
    from training.ppo.train import G008EnvironmentSeedSchedule

    run_id = "12345678-1234-4234-8234-123456789abc"
    with pytest.raises(ProvenanceError):
        G008EnvironmentSeedSchedule(None, run_id=run_id, arena_count=1, total_updates=1)
    expanded = _g008_train_schedule(run_id=run_id)
    with pytest.raises(ProvenanceError):
        G008EnvironmentSeedSchedule(expanded, run_id="12345678-1234-4234-8234-123456789abd",
                                    arena_count=1, total_updates=1)
    stale = copy.deepcopy(expanded)
    stale["run_key"] = "train-12345678-1234-4234-8234-123456789abd"
    with pytest.raises(ProvenanceError):
        G008EnvironmentSeedSchedule(stale, run_id=run_id, arena_count=1, total_updates=1)
    missing_reset = _g008_train_schedule(reset_indices=())
    adapter = G008EnvironmentSeedSchedule(missing_reset, run_id=run_id, arena_count=1, total_updates=1)
    with pytest.raises(ProvenanceError, match="no reserved seed"):
        adapter.seed_for(update=0, arena=0, reset_index=1)
    unreserved = copy.deepcopy(expanded)
    unreserved["slots"][0]["status"] = "protocol-excluded"
    # The schedule digest is intentionally left stale; either invariant rejects it.
    with pytest.raises(ProvenanceError):
        G008EnvironmentSeedSchedule(unreserved, run_id=run_id, arena_count=1, total_updates=1)
    out_of_partition = _g008_train_schedule(run_id=run_id)
    slots = []
    for source in out_of_partition["slots"]:
        slot = dict(source)
        slot["env_root_seed"] = 3_000_000
        slots.append(slot)
    with pytest.raises(ProvenanceError, match="outside frozen train partition"):
        build_expanded_schedule(suite_id="g008-train", run_id=run_id,
                                run_key=f"train-{run_id}", slots=slots)


def test_g008_run_keeps_config_seed_for_ppo_and_requires_frozen_run_binding(tmp_path):
    from training.eval.provenance import ProvenanceError

    run_id = "12345678-1234-4234-8234-123456789abc"
    schedule = _g008_train_schedule(run_id=run_id)
    config = TrainConfig(width=5, height=5, arena_count=1, rollout_steps=1, total_updates=1,
                         update_epochs=1, minibatch_size=1, hidden_size=8, device="cpu", seed=1234)
    with pytest.raises(ProvenanceError, match="model_ppo_seed"):
        train(TrainConfig(**{**config.__dict__, "seed": 4321}), tmp_path, export_onnx=False,
              run_evaluation=False, g008_schedule=schedule, run_id=run_id)
    assert uuid.UUID(run_id).version == 4


def test_actor_critic_never_assigns_probability_to_reverse_action():
    torch = pytest.importorskip("torch")
    from training.ppo.train import _model_class

    model = _model_class(torch)(12, 12, 16)
    observation = torch.zeros((4, 12, 12, 12), dtype=torch.float32)
    for row, heading in enumerate(range(4)):
        observation[row, 0, 0, 0] = 1.0
        observation[row, 0, 0, 4 + heading] = 1.0
    logits, _ = model(observation)
    for heading in range(4):
        reverse = (heading + 2) % 4
        assert logits[heading, reverse].item() < -1e20
        assert torch.isfinite(logits[heading, [a for a in range(4) if a != reverse]]).all()


def test_onnx_export_embeds_weights_for_single_file_browser_loading(tmp_path):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    from training.ppo.train import _model_class, _save_checkpoint

    config = TrainConfig(width=5, height=5, hidden_size=8, device="cpu")
    model = _model_class(torch)(config.width, config.height, config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = _save_checkpoint(torch, model, optimizer, config, 0, torch.device("cpu"),
                                  tmp_path, config.seed, 0, export_onnx=True)
    onnx_path = checkpoint["onnx"]
    assert onnx_path.is_file()
    assert not onnx_path.with_suffix(onnx_path.suffix + ".data").exists()
    onnx.checker.check_model(onnx.load(onnx_path))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed in this environment")
class TorchTrainingSmokeTests(unittest.TestCase):
    def test_gae_preserves_time_arena_order_and_masks_each_terminal_episode(self):
        import torch
        from training.ppo.train import _gae

        rewards = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        dones = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])]
        values = [torch.tensor([0.5, 1.0]), torch.tensor([1.5, 2.0])]
        advantages, returns = _gae(torch, rewards, dones, values, gamma=1.0, lam=1.0,
                                   bootstrap=torch.tensor([10.0, 20.0]))
        self.assertTrue(torch.equal(advantages, torch.tensor([[3.5, 1.0], [1.5, 22.0]])))
        self.assertTrue(torch.equal(returns, torch.tensor([[4.0, 2.0], [3.0, 24.0]])))
        self.assertTrue(torch.equal(advantages.reshape(-1), torch.tensor([3.5, 1.0, 1.5, 22.0])))
        self.assertTrue(torch.equal(returns.reshape(-1), torch.tensor([4.0, 2.0, 3.0, 24.0])))

    def test_tiny_cpu_run_performs_optimizer_updates_and_writes_resume_checkpoint(self):
        import torch
        config = TrainConfig(width=5, height=5, arena_count=1, rollout_steps=2, total_updates=1,
                             update_epochs=1, minibatch_size=1, hidden_size=8, device="cpu", checkpoint_every=1)
        with tempfile.TemporaryDirectory() as directory:
            report = train(config, directory, export_onnx=False)
            self.assertEqual(report["device"]["optimizer_updates"], 2)
            self.assertEqual(report["device"]["cuda_optimizer_updates"], 0)
            training = report["training"]
            self.assertGreater(training["mean_rollout_seconds"], 0)
            self.assertGreater(training["mean_optimizer_seconds"], 0)
            update = training["updates"][0]
            self.assertGreater(update["rollout_seconds"], 0)
            self.assertGreater(update["optimizer_seconds"], 0)
            self.assertGreater(update["checkpoint_seconds"], 0)
            self.assertGreater(update["update_latency_seconds"], 0)
            self.assertEqual(update["optimizer_updates"], 2)
            self.assertTrue(torch.isfinite(torch.tensor(update["mean_loss"])))
            self.assertGreaterEqual(training["elapsed_seconds"], training["update_elapsed_seconds"])
            self.assertAlmostEqual(training["checkpoint_elapsed_seconds"], update["checkpoint_seconds"])
            self.assertGreaterEqual(training["elapsed_seconds"],
                                    training["update_elapsed_seconds"] + training["checkpoint_elapsed_seconds"])
            self.assertAlmostEqual(training["environment_steps_per_second"],
                                   2 / training["update_elapsed_seconds"])
            self.assertAlmostEqual(training["wall_environment_steps_per_second"],
                                   2 / training["elapsed_seconds"])
            self.assertGreater(report["software"]["cpu_count"], 0)
            import pathlib
            checkpoint = pathlib.Path(directory) / "checkpoints" / "step-1.pt"
            self.assertTrue(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
