"""Tiny-configurable, two-seat shared-policy PPO self-play trainer."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import platform
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping

from training.arena.engine import initialize_episode, step
from training.arena.heuristics import HEURISTIC_STYLES, heuristic_action
from training.ppo.protocol import digest_json, file_sha256, write_json

OBSERVATION_SCHEMA = "lc-observation/1"
ACTION_SCHEMA = "lc-action/1"
ENGINE_VERSION = "lc-engine/1"
ARCH_NAME = "mlp-hwc-flatten"
ARCH_VERSION = "1"


@dataclass
class TrainConfig:
    width: int = 12
    height: int = 12
    arena_count: int = 32
    rollout_steps: int = 128
    total_updates: int = 10
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    seed: int = 1234
    device: str = "auto"
    checkpoint_every: int = 10
    onnx_opset: int = 17
    snapshot_opponent_probability: float = 0.0
    heuristic_opponent_probability: float = 0.0

    def validate(self) -> None:
        for name in ("width", "height", "arena_count", "rollout_steps", "total_updates", "update_epochs", "minibatch_size", "hidden_size", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width < 2 or self.height < 2:
            raise ValueError("board dimensions must be at least 2")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("learning_rate must be positive, gamma in (0,1], and gae_lambda in [0,1]")
        if not 0 < self.clip_ratio < 1 or self.entropy_coef < 0 or self.value_coef < 0:
            raise ValueError("invalid PPO clipping or loss coefficient")
        if not 0.0 <= self.snapshot_opponent_probability <= 1.0:
            raise ValueError("snapshot_opponent_probability must be in [0,1]")
        if not 0.0 <= self.heuristic_opponent_probability <= 1.0:
            raise ValueError("heuristic_opponent_probability must be in [0,1]")
        if type(self.onnx_opset) is not int or self.onnx_opset < 13:
            raise ValueError("onnx_opset must be an integer >= 13")
        if self.device == "cuda" and not _torch().cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if type(self.seed) is not int or not 32 <= self.seed <= (1 << 64) - 1:
            raise ValueError("training seed must be an unsigned 64-bit master seed outside held-out seeds 0..31")


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required; install with: python -m pip install -r training/requirements.txt") from error
    return torch


def _model_class(torch):
    class ActorCritic(torch.nn.Module):
        def __init__(self, width: int, height: int, hidden: int):
            super().__init__()
            self.width, self.height = width, height
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(width * height * 12, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            )
            self.policy = torch.nn.Linear(hidden, 4)
            self.value = torch.nn.Linear(hidden, 1)

        def forward(self, observation):
            features = self.encoder(observation)
            logits = self.policy(features)
            # Reversing is not a legal Light Cycles move. The observation
            # marks our heading at the head cell (channel 0); mask the opposite
            # action before both rollout sampling and browser inference so PPO
            # never trains on an impossible action and exported ONNX agrees.
            heading_planes = observation[..., 4:8]
            head = observation[..., 0:1]
            heading = (heading_planes * head).sum(dim=(1, 2)).argmax(dim=-1)
            reverse_action = (heading + 2) % 4
            illegal = torch.nn.functional.one_hot(reverse_action, num_classes=4).to(torch.bool)
            logits = logits.masked_fill(illegal, torch.finfo(logits.dtype).min)
            return logits, self.value(features).squeeze(-1)
    return ActorCritic


def _tensor_obs(torch, obs: list, device):
    # Compact channel-last uint8 observations; convert only at the model boundary.
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


def _observation_batch(states, agent_indices):
    """Build contiguous compact observations without allocating Python pixel lists."""
    import numpy as np

    if len(states) != len(agent_indices) or not states:
        raise ValueError("states and agent_indices must be non-empty and have equal length")
    width, height = states[0].width, states[0].height
    if any(state.width != width or state.height != height for state in states):
        raise ValueError("all batched states must share board dimensions")
    observations = np.zeros((len(states), height, width, 12), dtype=np.uint8)
    for index, (state, agent_index) in enumerate(zip(states, agent_indices)):
        if agent_index not in (0, 1):
            raise ValueError("agent_indices must contain only 0 or 1")
        me, opponent = state.agents[agent_index], state.agents[1 - agent_index]
        board = observations[index]
        for x, y in me.trail:
            board[y, x, 2] = 1
        for x, y in opponent.trail:
            board[y, x, 3] = 1
        x, y = me.position
        board[y, x, 0] = 1
        board[y, x, 4 + me.heading] = 1
        x, y = opponent.position
        board[y, x, 1] = 1
        board[y, x, 8 + opponent.heading] = 1
    return observations


def _observation_cache(states):
    """Build both seat observations once; subsequent rollout ticks update only heads."""
    if not states:
        raise ValueError("states must be non-empty")
    height, width = states[0].height, states[0].width
    batch = _observation_batch([state for state in states for _ in range(2)],
                              [seat for _ in states for seat in range(2)])
    return batch.reshape(len(states), 2, height, width, 12)


def _update_observation_cache(cache, previous_states, next_states) -> None:
    """Advance cached observations in O(arenas), without rescanning every trail."""
    if len(previous_states) != len(next_states) or len(cache) != len(next_states):
        raise ValueError("observation cache and state batches must have equal length")
    for arena, (previous, current) in enumerate(zip(previous_states, next_states)):
        if current.tick <= previous.tick:
            cache[arena] = _observation_batch([current, current], [0, 1]).reshape(
                2, current.height, current.width, 12)
            continue
        for agent_index, (old_agent, new_agent) in enumerate(zip(previous.agents, current.agents)):
            if old_agent.position == new_agent.position:
                continue
            old_x, old_y = old_agent.position
            new_x, new_y = new_agent.position
            own, opponent = cache[arena, agent_index], cache[arena, 1 - agent_index]
            own[old_y, old_x, 0] = 0
            opponent[old_y, old_x, 1] = 0
            own[old_y, old_x, 4 + old_agent.heading] = 0
            opponent[old_y, old_x, 8 + old_agent.heading] = 0
            own[new_y, new_x, 2] = 1
            opponent[new_y, new_x, 3] = 1
            own[new_y, new_x, 0] = 1
            opponent[new_y, new_x, 1] = 1
            own[new_y, new_x, 4 + new_agent.heading] = 1
            opponent[new_y, new_x, 8 + new_agent.heading] = 1


def _gae(torch, rewards, dones, values, gamma, lam, bootstrap):
    # Rollout collection keeps compact per-step lists; stack them once before
    # the reverse scan so GAE remains vectorized across all arenas.
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    values = torch.stack(values)
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(len(rewards) - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < len(values) else bootstrap
        continuation = 1.0 - dones[t]
        carry = rewards[t] + gamma * next_value * continuation - values[t] + gamma * lam * continuation * carry
        advantages[t] = carry
    return advantages, advantages + values


def terminal_rewards(result: str) -> tuple[float, float]:
    """Return zero-sum raw terminal rewards for the engine result discriminator."""
    if result == "draw":
        return 0.0, 0.0
    if result == "agent_0_win":
        return 1.0, -1.0
    if result == "agent_1_win":
        return -1.0, 1.0
    raise ValueError(f"unknown terminal result: {result!r}")


def reset_arena_index(update: int, arena: int, rollout_step: int, arena_count: int, total_updates: int) -> int:
    """Deterministic index range for within-rollout terminal resets."""
    return update * arena_count + arena + (total_updates + 1) * arena_count * (rollout_step + 1)


class G008EnvironmentSeedSchedule:
    """Fail-closed adapter from a frozen expanded schedule to engine seeds.

    PPO's configured seed is deliberately not involved: every environment seed
    comes from the exact registered env-seed/v1 tuple and its derived uint64.
    Collector coordinates are (absolute update, rollout 0, arena, game 0,
    reset index); initial states use reset 0 and terminal resets use step+1.
    """

    def __init__(self, expanded: Mapping[str, Any], *, run_id: str,
                 arena_count: int, total_updates: int, rollout_steps: int = 0,
                 benchmark_role: str | None = None):
        from training.eval.provenance import ProvenanceError, validate_expanded_schedule

        if not isinstance(expanded, Mapping):
            raise ProvenanceError("G008 training requires a frozen expanded environment schedule")
        validate_expanded_schedule(expanded)
        if not isinstance(run_id, str) or expanded.get("run_id") != run_id:
            raise ProvenanceError("G008 schedule run_id does not match the training run_id")
        expected_key = expanded.get("run_key") if expanded.get("run_key", "").startswith(("pair-", "warmup-")) else f"train-{run_id}"
        if expanded.get("run_key") != expected_key:
            raise ProvenanceError("G008 schedule run_key does not bind the canonical training run_id")
        if type(arena_count) is not int or arena_count < 1 or type(total_updates) is not int or total_updates < 1:
            raise ProvenanceError("G008 schedule dimensions must be positive integers")
        self.is_benchmark = expanded["run_key"].startswith(("pair-", "warmup-"))
        if self.is_benchmark:
            from training.eval.provenance import benchmark_run_keys
            if expanded["run_key"] not in benchmark_run_keys():
                raise ProvenanceError("benchmark schedule run_key is outside the frozen roster")
            self._initialize_benchmark(expanded, arena_count, total_updates, rollout_steps, benchmark_role)
            return
        if expanded.get("suite_id") != "g008-train":
            raise ProvenanceError("G008 PPO requires the g008-train schedule suite")
        slots: dict[tuple[int, int, int, int, int], int] = {}
        for slot in expanded["slots"]:
            if slot["split"] != "train" or slot["domain_label"] not in {
                    "g008/train/initial", "g008/train/reset"}:
                raise ProvenanceError("G008 training schedule contains a non-training seed slot")
            if slot["status"] != "reserved":
                raise ProvenanceError("G008 PPO cannot consume excluded or unreserved seed slots")
            update, rollout, arena = slot["update_index"], slot["rollout_index"], slot["arena_index"]
            game, reset = slot["game_index"], slot["reset_index"]
            if update >= total_updates or rollout != 0 or arena >= arena_count or game != 0:
                raise ProvenanceError("G008 schedule slot is outside the configured PPO collector coordinates")
            initial = slot["domain_label"] == "g008/train/initial"
            if (initial and reset != 0) or (not initial and reset == 0):
                raise ProvenanceError("G008 initial/reset domain disagrees with reset_index")
            key = (update, rollout, arena, game, reset)
            if key in slots:
                raise ProvenanceError("G008 schedule contains duplicate collector coordinates")
            slots[key] = slot["derived_seed_uint64"]
        expected_initials = {(update, 0, arena, 0, 0)
                             for update in range(total_updates) for arena in range(arena_count)}
        actual_initials = {key for key in slots if key[-1] == 0}
        if actual_initials != expected_initials:
            raise ProvenanceError("G008 schedule does not bind every configured initial arena slot")
        self._slots = slots
        self.run_key = expanded["run_key"]

    def _initialize_benchmark(self, expanded, arena_count, total_updates, rollout_steps, backend_role):
        from training.eval.provenance import ProvenanceError

        if expanded.get("suite_id") != "g008-benchmark":
            raise ProvenanceError("G008 benchmark PPO requires the g008-benchmark schedule suite")
        run_key = expanded["run_key"]
        measured = run_key.startswith("pair-")
        expected_updates = 3 if measured else 1
        expected_steps = 32
        if (total_updates, rollout_steps, arena_count) != (expected_updates, expected_steps, 128):
            raise ProvenanceError("benchmark schedule dimensions differ from the frozen principal config")
        if measured and backend_role not in {"incumbent", "candidate"}:
            raise ProvenanceError("measured benchmark schedule requires explicit incumbent/candidate backend role")
        if not measured and backend_role is not None:
            raise ProvenanceError("warmup schedule is shared and does not accept a paired backend role")
        backend_by_slot = {}
        for group in expanded["collision_scan"]["duplicate_groups"]:
            for member in group["members"]:
                backend_by_slot[member["slot_id"]] = member["backend_id"]
        slots = {}
        for slot in expanded["slots"]:
            backend = backend_by_slot.get(slot["slot_id"], "shared")
            if measured and backend != backend_role:
                continue
            if not measured and backend != "shared":
                raise ProvenanceError("warmup schedule contains a paired benchmark slot")
            if slot["split"] != "benchmark" or slot["status"] != "reserved":
                raise ProvenanceError("benchmark schedule contains a non-benchmark or unreserved slot")
            is_initial = slot["domain_label"].endswith("/initial")
            if (is_initial and slot["reset_index"] != 0) or (not is_initial and slot["reset_index"] == 0):
                raise ProvenanceError("benchmark initial/reset domain disagrees with reset_index")
            key = (slot["update_index"], slot["rollout_index"], slot["arena_index"],
                   slot["game_index"], slot["reset_index"])
            if key in slots:
                raise ProvenanceError("benchmark backend role leaves duplicate collector coordinates")
            slots[key] = slot["derived_seed_uint64"]
        initial_keys = {key for key in slots if key[4] == 0}
        expected_initials = {(0, key_rollout, arena, 0, 0) for arena in range(128)
                             for key_rollout in {key[1] for key in slots if key[4] == 0}}
        if initial_keys != expected_initials or len(initial_keys) != 128:
            raise ProvenanceError("benchmark schedule must preallocate exactly one initial tuple per arena/root")
        for arena in range(128):
            for update in range(expected_updates):
                for ordinal in range(1, expected_steps + 1):
                    game_index = update * expected_steps + ordinal
                    key = (update, next(iter({item[1] for item in initial_keys})), arena,
                           game_index, ordinal)
                    if key not in slots:
                        raise ProvenanceError("benchmark schedule is missing a max-reset upper-bound slot")
        if len(slots) != 128 * (1 + expected_updates * expected_steps):
            raise ProvenanceError("benchmark schedule has unexpected or out-of-range collector slots")
        self._slots = slots
        self.run_key = run_key
        self._benchmark_rollout_index = next(iter({key[1] for key in initial_keys}))

    def seed_for(self, *, update: int, arena: int, reset_index: int) -> int:
        from training.eval.provenance import ProvenanceError

        key = (update, 0, arena, 0, reset_index)
        try:
            return self._slots[key]
        except KeyError as error:
            raise ProvenanceError(f"G008 schedule has no reserved seed for collector slot {key}") from error

    def benchmark_initial_seed(self, arena: int) -> int:
        from training.eval.provenance import ProvenanceError
        try:
            return self._slots[(0, self._benchmark_rollout_index, arena, 0, 0)]
        except (AttributeError, KeyError) as error:
            raise ProvenanceError("benchmark schedule has no initial seed for arena") from error

    def benchmark_reset_seed(self, *, update: int, arena: int, reset_ordinal: int) -> int:
        from training.eval.provenance import ProvenanceError
        key = (update, self._benchmark_rollout_index, arena,
               update * 32 + reset_ordinal, reset_ordinal)
        try:
            return self._slots[key]
        except (AttributeError, KeyError) as error:
            raise ProvenanceError(f"benchmark schedule has no reset seed for collector slot {key}") from error


def updates_remaining(total_updates: int, completed_updates: int) -> int:
    """Interpret total_updates as a global target; reject already-complete runs."""
    if completed_updates >= total_updates:
        raise ValueError(f"checkpoint already reached target total_updates={total_updates} (step={completed_updates})")
    return total_updates - completed_updates


def sample_snapshot_opponents(arena_count: int, probability: float,
                              rng: Any = random) -> list[bool]:
    """Sample one frozen step-0 opponent choice per arena using supplied RNG."""
    if probability == 0:
        return [False] * arena_count
    return [rng.random() < probability for _ in range(arena_count)]


def make_checkpoint_metadata(checkpoint_id: str, step_num: int, config: TrainConfig,
                             onnx_path: Path, pt_path: Path) -> dict[str, Any]:
    """Build the schema-bound metadata for paired browser/native checkpoint files."""
    return {"format": "lc-checkpoint/1", "checkpoint_id": checkpoint_id, "step": step_num,
            "model_architecture": {"name": ARCH_NAME, "version": ARCH_VERSION,
                                   "config": {"width": config.width, "height": config.height,
                                              "hidden_size": config.hidden_size}},
            "observation_schema": OBSERVATION_SCHEMA, "action_schema": ACTION_SCHEMA,
            "onnx": {"file": f"checkpoints/{onnx_path.name}", "sha256": file_sha256(onnx_path),
                     "opset": config.onnx_opset, "runtime": "onnxruntime-web"},
            "pytorch": {"file": f"checkpoints/{pt_path.name}", "sha256": file_sha256(pt_path),
                        "format": "state_dict+optimizer"},
            "artifacts": [{"format": "onnx", "file": f"checkpoints/{onnx_path.name}", "sha256": file_sha256(onnx_path)},
                          {"format": "pytorch", "file": f"checkpoints/{pt_path.name}", "sha256": file_sha256(pt_path)}]}


def make_run_manifest(run_id: str, created_utc: str, config: TrainConfig, training_steps: int,
                      checkpoint_refs: list[dict], evaluation: dict, software: dict,
                      device: dict, benchmark_records: list[dict], files: list[dict]) -> dict[str, Any]:
    """Compose the lc-run/1 wire document; digest is attached by write_json."""
    training_config = asdict(config)
    training_config["self_play"] = {
        "opponent_sampling": "latest" if config.snapshot_opponent_probability == 0 else "latest-step-0-mixture",
        "snapshot_checkpoint_id": "step-0" if config.snapshot_opponent_probability > 0 else None,
        "snapshot_opponent_probability": config.snapshot_opponent_probability,
        "sampling_unit": "arena-per-rollout",
        "sampling_rng": "checkpointed-python-random-state",
    }
    return {"format": "lc-run/1", "run_id": run_id, "created_utc": created_utc,
            "engine_version": ENGINE_VERSION, "observation_schema": OBSERVATION_SCHEMA,
            "action_schema": ACTION_SCHEMA, "reward_schema": "lc-reward/1", "replay_schema": "lc-replay/1",
            "master_seed": str(config.seed), "training_config": training_config,
            "training_steps": training_steps, "checkpoints": checkpoint_refs,
            "evaluation": {"protocol": "lc-eval/1", "held_out_seeds": [str(i) for i in range(32)],
                           "games_per_matchup": 64, "opponent_ids": [m["opponent_id"] for m in evaluation["matchups"]],
                           "metrics_file": "evaluation/report.json",
                           "replay_files": [g["replay_file"] for m in evaluation["matchups"] for g in m["games"]]},
            "software": software, "device": device, "benchmark_records": benchmark_records, "files": files}


def _collect(torch, model, snapshot_model, config: TrainConfig, master_seed: int, device, update: int,
             environment_schedule: G008EnvironmentSeedSchedule | None = None,
             environment_states=None):
    if environment_schedule is None:
        states = [initialize_episode(config.width, config.height, master_seed,
                                     update * config.arena_count + i)[0]
                  for i in range(config.arena_count)]
    elif environment_schedule.is_benchmark and environment_states is not None:
        states = environment_states
    elif environment_schedule.is_benchmark:
        states = [initialize_episode(config.width, config.height,
                                     environment_schedule.benchmark_initial_seed(i), 0)[0]
                  for i in range(config.arena_count)]
    else:
        states = [initialize_episode(config.width, config.height,
                                     environment_schedule.seed_for(update=update, arena=i, reset_index=0), 0)[0]
                  for i in range(config.arena_count)]
    observation_cache = _observation_cache(states)
    occupied_cells = [{cell for agent in state.agents for cell in agent.trail} for state in states]
    # Snapshot choice is made once per arena/rollout using the checkpointed Python
    # RNG stream; episode resets retain that opponent for bounded deterministic provenance.
    snapshot_opponents = (sample_snapshot_opponents(config.arena_count, config.snapshot_opponent_probability)
                          if snapshot_model is not None else [False] * config.arena_count)
    heuristic_styles = [None] * config.arena_count
    if config.heuristic_opponent_probability:
        for arena in range(config.arena_count):
            if not snapshot_opponents[arena] and random.random() < config.heuristic_opponent_probability:
                heuristic_styles[arena] = random.choice(HEURISTIC_STYLES)
    learner_seats = [(update + arena) % 2 for arena in range(config.arena_count)]
    obs_steps, action_steps, logp_steps, value_steps, reward_steps, done_steps = [], [], [], [], [], []
    completed = 0
    benchmark_reset_counts = [0] * config.arena_count
    model.eval()
    for step_no in range(config.rollout_steps):
        observations = _tensor_obs(torch, observation_cache.reshape(-1, config.height, config.width, 12), device)
        with torch.no_grad():
            logits, values = model(observations)
            distribution = torch.distributions.Categorical(logits=logits)
            sampled_actions = distribution.sample()
            sampled_log_probs = distribution.log_prob(sampled_actions)
        joint_actions = sampled_actions.cpu().tolist()
        learner_indices = [arena * 2 + learner_seats[arena] for arena in range(config.arena_count)]
        # On snapshot-vs-current arenas, replace only the opponent action; old
        # snapshot behavior is never incorrectly included in the PPO loss batch.
        if snapshot_model is not None and any(snapshot_opponents):
            opponent_indices = [arena * 2 + (1 - learner_seats[arena]) for arena in range(config.arena_count)
                                if snapshot_opponents[arena]]
            snapshot_obs = observations[opponent_indices]
            with torch.no_grad():
                snapshot_logits, _ = snapshot_model(snapshot_obs)
                snapshot_actions = torch.distributions.Categorical(logits=snapshot_logits).sample().cpu().tolist()
            for index, action in zip(opponent_indices, snapshot_actions):
                joint_actions[index] = action
        for arena, style in enumerate(heuristic_styles):
            if style is not None:
                opponent_seat = 1 - learner_seats[arena]
                joint_actions[arena * 2 + opponent_seat] = heuristic_action(states[arena], opponent_seat, style)
        actions = sampled_actions[learner_indices]
        log_probs = sampled_log_probs[learner_indices]
        learner_values = values[learner_indices]
        learner_obs = observations[learner_indices]
        rewards = [0.001 if states[arena].agents[learner_seats[arena]].alive else 0.0
                   for arena in range(config.arena_count)]
        dones = [0.0] * config.arena_count
        next_states = []
        for arena, state in enumerate(states):
            trans = step(state, joint_actions[arena * 2:arena * 2 + 2],
                         occupied_cells=occupied_cells[arena])
            next_states.append(trans.state)
            for old_agent, new_agent in zip(state.agents, trans.state.agents):
                if old_agent.position != new_agent.position:
                    occupied_cells[arena].add(new_agent.position)
            if trans.state.result is not None:
                completed += 1
                rewards[arena] = terminal_rewards(trans.state.result)[learner_seats[arena]]
                dones[arena] = 1.0
        obs_steps.append(learner_obs)
        action_steps.append(actions)
        logp_steps.append(log_probs)
        value_steps.append(learner_values)
        reward_steps.append(torch.tensor(rewards, dtype=torch.float32, device=device))
        done_steps.append(torch.tensor(dones, dtype=torch.float32, device=device))
        # Reinitialize completed environments from their deterministic, non-evaluation stream.
        previous_states, states = states, []
        for arena, next_state in enumerate(next_states):
            if next_state.result is None:
                states.append(next_state)
                continue
            if environment_schedule is not None and environment_schedule.is_benchmark:
                benchmark_reset_counts[arena] += 1
                reset_state = initialize_episode(
                    config.width, config.height,
                    environment_schedule.benchmark_reset_seed(
                        update=update, arena=arena,
                        reset_ordinal=benchmark_reset_counts[arena]), 0)[0]
            elif environment_schedule is not None:
                reset_state = initialize_episode(
                    config.width, config.height,
                    environment_schedule.seed_for(update=update, arena=arena,
                                                  reset_index=step_no + 1), 0)[0]
            else:
                reset_state = initialize_episode(
                    config.width, config.height, master_seed,
                    reset_arena_index(update, arena, step_no, config.arena_count,
                                      config.total_updates))[0]
            states.append(reset_state)
            occupied_cells[arena] = {cell for agent in reset_state.agents for cell in agent.trail}
        _update_observation_cache(observation_cache, previous_states, states)
    final_batch = observation_cache[range(config.arena_count), learner_seats]
    final_obs = _tensor_obs(torch, final_batch, device)
    with torch.no_grad():
        _, bootstrap = model(final_obs)
    advantages, returns = _gae(torch, reward_steps, done_steps, value_steps, config.gamma, config.gae_lambda, bootstrap)
    snapshot_games = sum(snapshot_opponents)
    heuristic_games = sum(style is not None for style in heuristic_styles)
    heuristic_style_counts = {style: sum(sampled_style == style for sampled_style in heuristic_styles)
                              for style in HEURISTIC_STYLES}
    latest_games = config.arena_count - snapshot_games - heuristic_games
    return (torch.cat(obs_steps), torch.cat(action_steps), torch.cat(logp_steps),
            advantages.reshape(-1), returns.reshape(-1), completed,
            snapshot_games, latest_games, heuristic_games, heuristic_style_counts, states)


def _save_checkpoint(torch, model, optimizer, config, step_num: int, device, output: Path,
                     master_seed: int, update_count: int, export_onnx: bool) -> dict[str, Any]:
    checkpoint_id = f"step-{step_num}"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    pt_path = checkpoints / f"{checkpoint_id}.pt"
    torch.save({"format": "lc-checkpoint/1", "checkpoint_id": checkpoint_id, "step": step_num,
                "model_architecture": {"name": ARCH_NAME, "version": ARCH_VERSION,
                                       "config": {"width": config.width, "height": config.height, "hidden_size": config.hidden_size}},
                "observation_schema": OBSERVATION_SCHEMA, "action_schema": ACTION_SCHEMA,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "train_config": asdict(config), "master_seed": str(master_seed), "optimizer_updates": update_count,
                "training_update": step_num, "python_random_state": random.getstate(), "torch_random_state": torch.get_rng_state(),
                "cuda_random_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, pt_path)
    onnx_path = checkpoints / f"{checkpoint_id}.onnx"
    if export_onnx:
        if not _onnx_available():
            raise RuntimeError("ONNX export requested but onnx is unavailable; install training/requirements.txt")
        model.eval()
        sample = torch.zeros((1, config.height, config.width, 12), dtype=torch.float32, device=device)
        torch.onnx.export(model, sample, str(onnx_path), input_names=["observation"], output_names=["logits", "value"],
                          opset_version=config.onnx_opset,
                          dynamic_axes={"observation": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
                          external_data=False)
        metadata = make_checkpoint_metadata(checkpoint_id, step_num, config, onnx_path, pt_path)
        write_json(checkpoints / f"{checkpoint_id}.metadata.json", metadata, "metadata_sha256")
    return {"checkpoint_id": checkpoint_id, "step": step_num,
            "pt": pt_path, "onnx": onnx_path if export_onnx else None}


def _onnx_available() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def train(config: TrainConfig, output: str | Path, export_onnx: bool = True,
          resume: str | Path | None = None, run_evaluation: bool = True,
          g008_schedule: Mapping[str, Any] | None = None,
          run_id: str | None = None,
          g008_backend_role: str | None = None) -> dict[str, Any]:
    config.validate()
    environment_schedule = None
    if g008_schedule is not None:
        from training.eval.provenance import ProvenanceError, _read_seed_plan

        plan, _ = _read_seed_plan()
        if config.seed != plan["model_ppo_seed"]:
            raise ProvenanceError("G008 TrainConfig.seed must equal the frozen model_ppo_seed")
        if not isinstance(run_id, str) or not re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", run_id):
            raise ProvenanceError("G008 requires a lowercase canonical RFC 4122 run_id")
        environment_schedule = G008EnvironmentSeedSchedule(
            g008_schedule, run_id=run_id, arena_count=config.arena_count,
            total_updates=config.total_updates, rollout_steps=config.rollout_steps,
            benchmark_role=g008_backend_role)
    elif run_id is not None:
        raise ValueError("run_id is only accepted with a frozen G008 schedule")
    if config.heuristic_opponent_probability and (
            environment_schedule is None or environment_schedule.is_benchmark):
        raise ValueError("G008 heuristic opponents require a frozen g008-train schedule")
    g008_training_run = environment_schedule is not None and not environment_schedule.is_benchmark
    if g008_training_run and run_evaluation:
        raise ValueError("G008 training must not invoke the G007 open-space evaluator; use split-aware validation")
    if export_onnx and not run_evaluation and not g008_training_run:
        raise ValueError("ONNX manifest export requires the fixed evaluation report")
    torch = _torch()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = _model_class(torch)(config.width, config.height, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_update = 0
    optimizer_updates = 0
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved.get("format") != "lc-checkpoint/1" or saved.get("observation_schema") != OBSERVATION_SCHEMA or saved.get("action_schema") != ACTION_SCHEMA:
            raise ValueError("resume checkpoint has an unsupported contract")
        if saved.get("master_seed") != str(config.seed):
            raise ValueError("resume checkpoint master seed does not match training config")
        architecture = saved["model_architecture"]["config"]
        if (architecture["width"], architecture["height"], architecture["hidden_size"]) != (config.width, config.height, config.hidden_size):
            raise ValueError("resume checkpoint architecture does not match training config")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_update = int(saved["training_update"])
        optimizer_updates = int(saved["optimizer_updates"])
        random.setstate(saved["python_random_state"])
        torch.set_rng_state(saved["torch_random_state"].cpu())
        if device.type == "cuda" and saved.get("cuda_random_states"):
            torch.cuda.set_rng_state_all(saved["cuda_random_states"])
    model.to(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Step 0 is retained as the immutable self-play evaluation reference.
    step_zero_pt = output / "checkpoints" / "step-0.pt"
    if start_update > 0 and not step_zero_pt.exists():
        raise ValueError("resume output is missing immutable step-0 checkpoint; resume from the original run directory")
    if start_update == 0:
        initial = _save_checkpoint(torch, model, optimizer, config, 0, device, output,
                                   config.seed, optimizer_updates, export_onnx)
    else:
        initial = {"checkpoint_id": "step-0", "step": 0, "pt": step_zero_pt,
                   "onnx": output / "checkpoints" / "step-0.onnx" if export_onnx else None}
        if export_onnx and not initial["onnx"].exists():
            raise ValueError("resume output is missing immutable step-0 ONNX export")
    checkpoints = [initial]
    snapshot_model = None
    if config.snapshot_opponent_probability > 0:
        initial_weights = torch.load(initial["pt"], map_location=device, weights_only=False)
        snapshot_config = initial_weights["model_architecture"]["config"]
        snapshot_model = _model_class(torch)(snapshot_config["width"], snapshot_config["height"],
                                             snapshot_config["hidden_size"]).to(device)
        snapshot_model.load_state_dict(initial_weights["model_state_dict"])
        snapshot_model.eval()
        for parameter in snapshot_model.parameters():
            parameter.requires_grad_(False)
    logs, cuda_update_count = [], 0
    updates = updates_remaining(config.total_updates, start_update)
    carried_environment_states = None
    started = time.perf_counter()
    for local_update in range(updates):
        update_started = time.perf_counter()
        absolute_update = start_update + local_update
        rollout_started = time.perf_counter()
        obs, actions, old_logp, advantages, returns, completed, snapshot_games, latest_games, heuristic_games, heuristic_style_counts, final_environment_states = _collect(
            torch, model, snapshot_model, config, config.seed, device, absolute_update,
            environment_schedule, carried_environment_states)
        if environment_schedule is not None and environment_schedule.is_benchmark:
            carried_environment_states = final_environment_states
        rollout_seconds = time.perf_counter() - rollout_started
        optimizer_started = time.perf_counter()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        loss_count = 0
        update_cuda_count = 0
        for _ in range(config.update_epochs):
            permutation = torch.randperm(len(actions), device=device)
            for indices in permutation.split(config.minibatch_size):
                logits, values = model(obs[indices])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[indices])
                ratio = (logp - old_logp[indices]).exp()
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (returns[indices] - values).square().mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer_updates += 1
                if device.type == "cuda":
                    cuda_update_count += 1
                    update_cuda_count += 1
                # Keep minibatch metrics on-device; copying each scalar here
                # would synchronize CUDA once per optimizer step.
                loss_sum.add_(loss.detach())
                loss_count += 1
        mean_loss = float((loss_sum / max(1, loss_count)).item())
        optimizer_seconds = time.perf_counter() - optimizer_started
        logs.append({"update": absolute_update + 1, "episodes_completed": completed,
                     "optimizer_updates": loss_count, "mean_loss": mean_loss,
                     "device": str(device), "cuda_updates": update_cuda_count,
                     "rollout_seconds": rollout_seconds, "optimizer_seconds": optimizer_seconds,
                     "checkpoint_seconds": 0.0,
                     "opponent_sampling": "latest-step-0-mixture" if snapshot_model is not None else "latest",
                     "snapshot_opponent_arenas": snapshot_games, "latest_opponent_arenas": latest_games,
                     "heuristic_opponent_arenas": heuristic_games,
                     "heuristic_opponent_style_counts": heuristic_style_counts,
                     "opponent_checkpoint_id": "step-0" if snapshot_games else None,
                     "update_latency_seconds": time.perf_counter() - update_started})
        step_num = absolute_update + 1
        if (local_update + 1) % config.checkpoint_every == 0 or local_update + 1 == updates:
            checkpoint_started = time.perf_counter()
            checkpoints.append(_save_checkpoint(torch, model, optimizer, config, step_num, device, output,
                                                config.seed, optimizer_updates, export_onnx))
            logs[-1]["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
    elapsed = time.perf_counter() - started
    rollout_total = sum(log["rollout_seconds"] for log in logs)
    optimizer_total = sum(log["optimizer_seconds"] for log in logs)
    update_elapsed = sum(log["update_latency_seconds"] for log in logs)
    checkpoint_elapsed = sum(log["checkpoint_seconds"] for log in logs)
    measured_steps = config.arena_count * config.rollout_steps * updates
    software = {"python": platform.python_version(), "cpu_count": os.cpu_count(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda or "unavailable", "onnx_exporter": "torch.onnx.export",
                "onnx_opset": config.onnx_opset, "kaggle_cli": "not-used"}
    run_id = run_id or re.sub(r"[^A-Za-z0-9.:+-]", "-", output.name or "light-cycles-run")
    report = {"run_id": run_id, "device": {"device": str(device),
              "cuda_available": bool(torch.cuda.is_available()), "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "optimizer_updates": sum(log["optimizer_updates"] for log in logs), "cuda_optimizer_updates": cuda_update_count},
              "training": {"config": asdict(config), "start_update": start_update, "training_steps": (start_update + updates) * config.rollout_steps * config.arena_count,
                           "elapsed_seconds": elapsed, "update_elapsed_seconds": update_elapsed,
                           "checkpoint_elapsed_seconds": checkpoint_elapsed,
                           "environment_steps_per_second": measured_steps / max(update_elapsed, 1e-9),
                           "wall_environment_steps_per_second": measured_steps / max(elapsed, 1e-9),
                           "mean_rollout_seconds": rollout_total / max(1, len(logs)),
                           "mean_optimizer_seconds": optimizer_total / max(1, len(logs)),
                           "rollout_to_optimizer_ratio": rollout_total / max(optimizer_total, 1e-9),
                           "mean_update_latency_seconds": sum(log["update_latency_seconds"] for log in logs) / max(1, len(logs)),
                           "updates": logs},
              "software": software, "initial_checkpoint": initial["pt"].relative_to(output).as_posix()}
    (output / "training-metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    evaluation = None
    if run_evaluation:
        from training.eval.lc_eval import evaluate
        target_policy, target_id = load_policy(checkpoints[-1]["pt"], str(device))
        step_zero_policy, _ = load_policy(initial["pt"], str(device))
        evaluation = evaluate(target_policy, step_zero_policy, run_id, target_id, output / "evaluation",
                              config.width, config.height)
        report["evaluation"] = {"protocol": "lc-eval/1", "matchups": {
            matchup["opponent_id"]: matchup["summary"] for matchup in evaluation["matchups"]}}
        (output / "training-metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if export_onnx and not g008_training_run:
        checkpoint_refs, files = [], []
        for item in checkpoints:
            for fmt, path in (("pytorch", item["pt"]), ("onnx", item["onnx"])):
                relative = path.relative_to(output).as_posix()
                checksum = file_sha256(path)
                checkpoint_refs.append({"checkpoint_id": item["checkpoint_id"], "step": item["step"],
                                        "file": relative, "sha256": checksum, "format": fmt,
                                        "observation_schema": OBSERVATION_SCHEMA, "action_schema": ACTION_SCHEMA})
                files.append({"path": relative, "sha256": checksum})
            metadata_path = item["pt"].with_suffix(".metadata.json")
            files.append({"path": metadata_path.relative_to(output).as_posix(), "sha256": file_sha256(metadata_path)})
        evaluation_path = output / "evaluation" / "report.json"
        files.append({"path": evaluation_path.relative_to(output).as_posix(), "sha256": file_sha256(evaluation_path)})
        for matchup in evaluation["matchups"]:
            for game in matchup["games"]:
                replay_path = output / game["replay_file"]
                files.append({"path": replay_path.relative_to(output).as_posix(), "sha256": file_sha256(replay_path)})
        files.append({"path": "training-metrics.json", "sha256": file_sha256(output / "training-metrics.json")})
        manifest = make_run_manifest(run_id, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), config,
                                     report["training"]["training_steps"], checkpoint_refs, evaluation, software,
                                     report["device"], [{"environment_steps_per_second": report["training"]["environment_steps_per_second"],
                                                        "wall_environment_steps_per_second": report["training"]["wall_environment_steps_per_second"],
                                                        "mean_update_latency_seconds": report["training"]["mean_update_latency_seconds"],
                                                        "update_elapsed_seconds": report["training"]["update_elapsed_seconds"],
                                                        "checkpoint_elapsed_seconds": report["training"]["checkpoint_elapsed_seconds"],
                                                        "mean_rollout_seconds": report["training"]["mean_rollout_seconds"],
                                                        "mean_optimizer_seconds": report["training"]["mean_optimizer_seconds"],
                                                        "rollout_to_optimizer_ratio": report["training"]["rollout_to_optimizer_ratio"],
                                                        "device": str(device), "optimizer_updates": report["device"]["optimizer_updates"]}], files)
        write_json(output / "manifest.json", manifest, "manifest_sha256")
    return report


def load_policy(checkpoint: str | Path, device: str = "cpu"):
    """Load a native checkpoint and return (deterministic_policy, id)."""
    torch = _torch()
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get("format") != "lc-checkpoint/1" or saved.get("observation_schema") != OBSERVATION_SCHEMA:
        raise ValueError("unsupported checkpoint format/schema")
    cfg = saved["model_architecture"]["config"]
    model = _model_class(torch)(cfg["width"], cfg["height"], cfg["hidden_size"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    def policy(observation):
        with torch.no_grad():
            logits, _ = model(_tensor_obs(torch, [observation], device))
            return int(torch.argmax(logits[0]).item())
    policy.width, policy.height = cfg["width"], cfg["height"]
    return policy, saved["checkpoint_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--config", type=Path, help="JSON object overriding TrainConfig defaults")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-onnx", action="store_true", help="save resumable native checkpoints only")
    parser.add_argument("--g008-schedule", type=Path,
                        help="frozen expanded G008 train schedule JSON (requires --run-id)")
    parser.add_argument("--run-id", help="canonical UUID bound to --g008-schedule")
    args = parser.parse_args(argv)
    values = json.loads(args.config.read_text()) if args.config else {}
    schedule = json.loads(args.g008_schedule.read_text()) if args.g008_schedule else None
    report = train(TrainConfig(**values), args.output, not args.no_onnx, args.resume,
                   g008_schedule=schedule, run_id=args.run_id)
    print(json.dumps({"device": report["device"], "training_steps": report["training"]["training_steps"],
                      "cuda_optimizer_updates": report["device"]["cuda_optimizer_updates"]}, indent=2))


if __name__ == "__main__":
    main()
