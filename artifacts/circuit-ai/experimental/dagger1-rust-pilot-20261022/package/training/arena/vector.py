"""Dependency-free N-arena headless runner with scalar-equivalent stepping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .engine import State, Transition, initialize_episode, step


@dataclass
class VectorArena:
    states: list[State]
    stream_seeds: list[tuple]

    @classmethod
    def create(cls, count: int, width: int, height: int, master_seed: int,
               start_index: int = 0) -> "VectorArena":
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        states, seeds = [], []
        for arena_index in range(start_index, start_index + count):
            state, arena_seeds = initialize_episode(width, height, master_seed, arena_index)
            states.append(state)
            seeds.append(arena_seeds)
        return cls(states, seeds)

    def step(self, joint_actions: Sequence[Sequence[object]]) -> list[Transition]:
        if len(joint_actions) != len(self.states):
            raise ValueError("one joint action pair is required per arena")
        transitions = [step(state, actions) for state, actions in zip(self.states, joint_actions)]
        # Commit only after every arena's transition has been computed.
        self.states = [transition.state for transition in transitions]
        return transitions
