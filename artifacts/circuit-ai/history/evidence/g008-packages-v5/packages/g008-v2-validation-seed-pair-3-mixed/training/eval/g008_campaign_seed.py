"""Small seed protocol required by preregistered G008 train/validation runs.

This module contains no final evaluation roots or opponent identity. Nonfinal
Kaggle packages include it instead of the full G008 seed plan and provenance
registry implementation.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from training.ppo.protocol import digest_json


FROZEN_SEED_PLAN_SHA256 = "c6b16733a045e94e7310183e8be8a62cf4327b5d2bdc3eed40bb9d552ce17593"
SEED_CODEC_PREFIX = b"CIRCUIT-AI/G008/ENV-SEED/v1\0"
SEED_TUPLE_FIELDS = ("split", "domain_label", "env_root_seed", "run_key",
                     "update_index", "rollout_index", "arena_index", "game_index", "reset_index")
_SPLITS = {
    "train": ((2_000_000, 2_999_999), ("g008/train/initial", "g008/train/reset")),
    "validation": ((1_000_000, 1_000_255),
                   ("g008/eval/validation/initial", "g008/eval/validation/reset")),
}
_UINT64_MAX = (1 << 64) - 1


class CampaignSeedError(ValueError):
    """A campaign-safe seed tuple or schedule differs from its frozen protocol."""


def seed_plan_sha256() -> str:
    """Return the frozen full-plan commitment without packaging its holdout data."""
    return FROZEN_SEED_PLAN_SHA256


def _encode_string(value: str) -> bytes:
    if not isinstance(value, str):
        raise CampaignSeedError("seed codec string fields must be strings")
    encoded = value.encode("utf-8")
    if len(encoded) > 0xFFFFFFFF:
        raise CampaignSeedError("seed codec string exceeds uint32 framing")
    return len(encoded).to_bytes(4, "big") + encoded


def _uint64(value: Any, field: str) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _UINT64_MAX:
        raise CampaignSeedError(f"{field} must be a required uint64")
    return value.to_bytes(8, "big")


def _validate_tuple(seed_tuple: Mapping[str, Any]) -> None:
    if not isinstance(seed_tuple, Mapping) or set(seed_tuple) != set(SEED_TUPLE_FIELDS):
        raise CampaignSeedError("seed tuple must contain exactly the nine codec fields")
    split = seed_tuple["split"]
    if split not in _SPLITS:
        raise CampaignSeedError("campaign-safe seed tuple uses an unavailable split")
    (lower, upper), domains = _SPLITS[split]
    if seed_tuple["domain_label"] not in domains:
        raise CampaignSeedError("seed tuple domain label differs from the campaign-safe mapping")
    root = seed_tuple["env_root_seed"]
    if type(root) is not int or not lower <= root <= upper:
        raise CampaignSeedError("environment root is outside the campaign-safe partition")
    run_key = seed_tuple["run_key"]
    if split == "train":
        if not isinstance(run_key, str) or not re.fullmatch(
                r"train-[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", run_key):
            raise CampaignSeedError("train run_key must bind a canonical lowercase UUID")
    elif run_key != "validation-suite-v1":
        raise CampaignSeedError("validation run_key differs from the frozen roster")
    for name in SEED_TUPLE_FIELDS[4:]:
        _uint64(seed_tuple[name], name)


def encode_seed_tuple(seed_tuple: Mapping[str, Any]) -> bytes:
    """Encode only training or nonfinal validation tuples using env-seed/v1."""
    _validate_tuple(seed_tuple)
    result = bytearray(SEED_CODEC_PREFIX)
    result.extend(_encode_string(seed_tuple["split"]))
    result.extend(_encode_string(seed_tuple["domain_label"]))
    result.extend(_uint64(seed_tuple["env_root_seed"], "env_root_seed"))
    result.extend(_encode_string(seed_tuple["run_key"]))
    for name in SEED_TUPLE_FIELDS[4:]:
        result.extend(_uint64(seed_tuple[name], name))
    return bytes(result)


def derive_environment_seed(seed_tuple: Mapping[str, Any]) -> tuple[str, int, bytes]:
    payload = encode_seed_tuple(seed_tuple)
    digest = hashlib.sha256(payload).digest()
    return digest.hex(), int.from_bytes(digest[:8], "big"), payload


def encode_frozen_train_seed_tuple(seed_tuple: Mapping[str, Any]) -> bytes:
    if seed_tuple.get("split") != "train":
        raise CampaignSeedError("compact campaign seed tuple must use the train split")
    return encode_seed_tuple(seed_tuple)


def campaign_schedule_descriptor(*, run_uuid: str, root_seed: int, arena_count: int,
                                 total_updates: int, rollout_steps: int,
                                 seed_plan_sha: str) -> dict[str, Any]:
    if seed_plan_sha != FROZEN_SEED_PLAN_SHA256:
        raise CampaignSeedError("campaign schedule differs from the frozen seed-plan commitment")
    if type(root_seed) is not int or not _SPLITS["train"][0][0] <= root_seed <= _SPLITS["train"][0][1]:
        raise CampaignSeedError("campaign root seed is outside the frozen train partition")
    if not isinstance(run_uuid, str) or not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", run_uuid):
        raise CampaignSeedError("campaign run UUID must be canonical lowercase RFC 4122")
    for name, value in (("arena_count", arena_count), ("total_updates", total_updates),
                        ("rollout_steps", rollout_steps)):
        if type(value) is not int or value < 1:
            raise CampaignSeedError(f"campaign {name} must be a positive integer")
    descriptor = {
        "format": "g008-campaign-environment-seed-schedule/1",
        "seed_plan_sha256": FROZEN_SEED_PLAN_SHA256,
        "codec": "env-seed/v1",
        "run_uuid": run_uuid,
        "run_key": f"train-{run_uuid}",
        "root_seed": root_seed,
        "arena_count": arena_count,
        "total_updates": total_updates,
        "rollout_steps": rollout_steps,
    }
    return {**descriptor, "schedule_descriptor_sha256": digest_json(descriptor, "")}
