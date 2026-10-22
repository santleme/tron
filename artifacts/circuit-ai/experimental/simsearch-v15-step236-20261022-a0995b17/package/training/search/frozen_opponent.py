"""Minimal deterministic inference for the protocol-permitted frozen opponent."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from training.arena.engine import observe
from .tournament import FROZEN_OPPONENTS


@lru_cache(maxsize=1)
def _load():
    import torch

    torch.set_num_threads(1)
    opponent_id, expected_sha = next(iter(FROZEN_OPPONENTS.items()))
    path = Path(__file__).resolve().parent / "assets" / f"{opponent_id}.pt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha:
        raise RuntimeError("staged v15 step-236 checkpoint SHA-256 mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("format") != "lc-checkpoint/1"
            or saved.get("checkpoint_id") != "step-236"
            or saved.get("step") != 236
            or saved.get("observation_schema") != "lc-observation/1"
            or saved.get("action_schema") != "lc-action/1"):
        raise RuntimeError("staged v15 step-236 checkpoint metadata is invalid")
    architecture = saved.get("model_architecture", {})
    config = architecture.get("config", {})
    if (architecture.get("name") != "mlp-hwc-flatten"
            or architecture.get("version") != "1"
            or config != {"height": 12, "hidden_size": 128, "width": 12}):
        raise RuntimeError("staged v15 step-236 architecture is not the permitted model")

    class ActorCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(12 * 12 * 12, 128), torch.nn.Tanh(),
                torch.nn.Linear(128, 128), torch.nn.Tanh(),
            )
            self.policy = torch.nn.Linear(128, 4)
            self.value = torch.nn.Linear(128, 1)

        def forward(self, observation):
            features = self.encoder(observation)
            logits = self.policy(features)
            head = observation[..., 0]
            heading_planes = observation[..., 4:8]
            heading = (heading_planes * head.unsqueeze(-1)).sum(dim=(1, 2)).argmax(dim=-1)
            reverse = torch.nn.functional.one_hot((heading + 2) % 4, num_classes=4).to(torch.bool)
            return logits.masked_fill(reverse, torch.finfo(logits.dtype).min)

    model = ActorCritic()
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return torch, model


def choose_action(state, player: int, opponent_id: str) -> int:
    """Return viewer-compatible deterministic argmax on the canonical observation."""
    if opponent_id not in FROZEN_OPPONENTS:
        raise ValueError("unapproved frozen opponent")
    if state.width != 12 or state.height != 12:
        raise ValueError("v15-step236 requires its canonical 12x12 board")
    torch, model = _load()
    with torch.no_grad():
        scores = model(torch.as_tensor([observe(state, player)], dtype=torch.float32))[0]
        return int(torch.argmax(scores).item())
