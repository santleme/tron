"""Supervised fixed-depth Policy-Value distillation for search experiments.

Input datasets use the compact arrays written by ``training.search.distill``.
Training is intentionally separate from the official PPO pipeline.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Any

from training.ppo.protocol import digest_json, file_sha256

OBSERVATION_SCHEMA = "lc-observation/1"
ACTION_SCHEMA = "lc-action/1"
ARCH_NAME = "mlp-hwc-flatten"
ARCH_VERSION = "2"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    hidden_size: int = 128
    learning_rate: float = 3e-4
    value_weight: float = 1.0
    validation_fraction: float = 0.1
    seed: int = 20261022
    device: str = "auto"

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "hidden_size", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a {'non-negative' if name == 'seed' else 'positive'} integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.value_weight) or self.value_weight < 0:
            raise ValueError("value_weight must be non-negative and finite")
        if not math.isfinite(self.validation_fraction) or not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load and validate the distill generator's compressed NPZ contract."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - Kaggle and training envs provide numpy
        raise RuntimeError("numpy is required to load Policy-Value datasets") from exc

    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"observations", "policies", "values", "legal_masks", "seeds"}
        if not required.issubset(archive.files):
            raise ValueError(f"dataset is missing arrays: {sorted(required - set(archive.files))}")
        data = {name: archive[name] for name in required}

    observations, policies = data["observations"], data["policies"]
    values, masks = data["values"], data["legal_masks"]
    if observations.dtype != np.uint8 or observations.ndim != 4 or observations.shape[-1] != 12:
        raise ValueError("observations must be uint8 [N,H,W,12]")
    count, height, width, _ = observations.shape
    if count < 2 or height < 2 or width < 2:
        raise ValueError("dataset needs at least two examples on boards at least 2x2")
    if policies.dtype != np.float32 or policies.shape != (count, 4):
        raise ValueError("policies must be float32 [N,4]")
    if values.dtype != np.float32 or values.shape != (count,):
        raise ValueError("values must be float32 [N]")
    if masks.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)) or masks.shape != (count, 4):
        raise ValueError("legal_masks must be bool [N,4] (uint8 0/1 is also accepted)")
    seeds = data["seeds"]
    if seeds.dtype != np.int64 or seeds.shape != (count,) or (seeds < 0).any():
        raise ValueError("seeds must be non-negative int64 values with shape [N]")
    legal = masks.astype(bool, copy=False)
    if not legal.any(axis=1).all():
        raise ValueError("every training example must have at least one legal action")
    if not np.isfinite(policies).all() or (policies < 0).any():
        raise ValueError("policies must contain finite non-negative probabilities")
    if not np.isfinite(values).all() or (np.abs(values) > 1).any():
        raise ValueError("values must be finite and in [-1,1]")
    if (policies[~legal] > 1e-7).any():
        raise ValueError("policy targets assign probability to an illegal action")
    if not np.allclose(policies.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("each policy target must sum to one")
    data["legal_masks"] = legal
    return data


def _grouped_seed_split(seeds, validation_fraction: float, seed: int):
    """Split examples by episode root so adjacent states never cross the boundary."""
    import numpy as np

    unique = np.unique(seeds)
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    validation_seed_count = 0
    if validation_fraction > 0 and len(unique) >= 2:
        validation_seed_count = min(len(unique) - 1,
                                    max(1, int(round(len(unique) * validation_fraction))))
    shuffled = np.random.default_rng(seed).permutation(unique)
    validation_seeds = shuffled[:validation_seed_count]
    validation_mask = np.isin(seeds, validation_seeds)
    validation_indices = np.flatnonzero(validation_mask)
    training_indices = np.flatnonzero(~validation_mask)
    if not len(training_indices):
        raise ValueError("seed-group split produced an empty training set")
    return training_indices, validation_indices


def _model_class(torch):
    """Create the viewer-compatible channel-last flattened MLP model."""
    class PolicyValue(torch.nn.Module):
        def __init__(self, width: int, height: int, hidden_size: int):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(width * height * 12, hidden_size), torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, hidden_size), torch.nn.Tanh(),
            )
            self.policy = torch.nn.Linear(hidden_size, 4)
            self.value = torch.nn.Linear(hidden_size, 1)

        def forward(self, observation):
            features = self.encoder(observation)
            logits = self.policy(features)
            # Match the viewer-compatible action mask v2 used by the PPO MLP:
            # reverse is illegal, and walls/trails are excluded when any safe
            # forward action exists. The engine resolves simultaneous head
            # collisions after both riders choose.
            head_plane = observation[..., 0]
            head = head_plane.unsqueeze(-1)
            heading_planes = observation[..., 4:8]
            heading = (heading_planes * head).sum(dim=(1, 2)).argmax(dim=-1)
            reverse = torch.nn.functional.one_hot((heading + 2) % 4, num_classes=4).to(torch.bool)
            height, width = observation.shape[1:3]
            x = (head_plane * torch.arange(width, device=observation.device,
                                            dtype=observation.dtype).view(1, 1, width)).sum((1, 2)).long()
            y = (head_plane * torch.arange(height, device=observation.device,
                                            dtype=observation.dtype).view(1, height, 1)).sum((1, 2)).long()
            occupied = (observation[..., 2] + observation[..., 3]).reshape(-1, height * width) > 0
            offsets = torch.tensor((-width, 1, width, -1), device=observation.device)
            destination = y[:, None] * width + x[:, None] + offsets[None, :]
            in_bounds = torch.stack((y > 0, x < width - 1, y < height - 1, x > 0), dim=1)
            destination_occupied = torch.gather(
                occupied, 1, destination.clamp(0, height * width - 1))
            unsafe = ~in_bounds | destination_occupied
            has_safe_move = (~unsafe & ~reverse).any(dim=1, keepdim=True)
            illegal = reverse | (unsafe & has_safe_move)
            logits = logits.masked_fill(illegal, torch.finfo(logits.dtype).min)
            return logits, self.value(features).squeeze(-1)
    return PolicyValue


def _masked_policy_value_loss(torch, logits, predicted_values, target_policies,
                              target_values, legal_masks, value_weight: float):
    masked_logits = logits.masked_fill(~legal_masks, torch.finfo(logits.dtype).min)
    policy_loss = -(target_policies * torch.log_softmax(masked_logits, dim=-1)).sum(dim=-1).mean()
    value_loss = torch.nn.functional.mse_loss(predicted_values, target_values)
    return policy_loss + value_weight * value_loss, policy_loss, value_loss


def _checkpoint_id(seed: int, run_id: str | None = None) -> str:
    candidate = f"{run_id}-policy-value" if run_id else f"search-pv-seed-{seed}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", candidate):
        raise ValueError("run_id must produce a checkpoint ID with only letters, digits, '.', '_', ':', or '-'")
    return candidate


def train(dataset_path: str | Path, output_dir: str | Path,
          config: TrainConfig = TrainConfig(), *, export_onnx: bool = True,
          run_id: str | None = None, dataset_source: str | None = None) -> dict[str, Any]:
    config.validate()
    data = load_dataset(dataset_path)
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - tested in Kaggle runtime
        raise RuntimeError("Policy-Value training requires numpy and PyTorch") from exc
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if config.device == "cuda" or
                          (config.device == "auto" and torch.cuda.is_available()) else "cpu")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    observations = data["observations"]
    count, height, width, _ = observations.shape
    training_indices, validation_indices = _grouped_seed_split(
        data["seeds"], config.validation_fraction, config.seed)
    validation_count = len(validation_indices)
    model_type = _model_class(torch)
    model = model_type(width, height, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    tensors = {
        "observations": torch.as_tensor(observations, dtype=torch.float32, device=device),
        "policies": torch.as_tensor(data["policies"], dtype=torch.float32, device=device),
        "values": torch.as_tensor(data["values"], dtype=torch.float32, device=device),
        "legal_masks": torch.as_tensor(data["legal_masks"], dtype=torch.bool, device=device),
    }
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        shuffled = torch.randperm(len(training_indices), generator=generator).numpy()
        epoch_losses: list[float] = []
        for start in range(0, len(training_indices), config.batch_size):
            indices = training_indices[shuffled[start:start + config.batch_size]]
            batch = torch.as_tensor(indices, dtype=torch.long, device=device)
            logits, predicted_values = model(tensors["observations"][batch])
            loss, _, _ = _masked_policy_value_loss(
                torch, logits, predicted_values, tensors["policies"][batch],
                tensors["values"][batch], tensors["legal_masks"][batch], config.value_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "train_loss": float(np.mean(epoch_losses))})

    model.eval()
    with torch.no_grad():
        if validation_count:
            batch = torch.as_tensor(validation_indices, dtype=torch.long, device=device)
            logits, predicted_values = model(tensors["observations"][batch])
            val_loss, val_policy_loss, val_value_loss = _masked_policy_value_loss(
                torch, logits, predicted_values, tensors["policies"][batch],
                tensors["values"][batch], tensors["legal_masks"][batch], config.value_weight)
            validation = {"loss": float(val_loss.cpu()), "policy_loss": float(val_policy_loss.cpu()),
                          "value_mse": float(val_value_loss.cpu()), "examples": int(validation_count)}
        else:
            validation = {"loss": None, "policy_loss": None, "value_mse": None, "examples": 0}

    output = Path(output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_id = _checkpoint_id(config.seed, run_id)
    pt_path = checkpoints / f"{checkpoint_id}.pt"
    onnx_path = checkpoints / f"{checkpoint_id}.onnx"
    torch.save({"format": "lc-checkpoint/1", "checkpoint_id": checkpoint_id,
                "step": config.epochs,
                "observation_schema": OBSERVATION_SCHEMA, "action_schema": ACTION_SCHEMA,
                "model_architecture": {"name": ARCH_NAME, "version": ARCH_VERSION,
                                       "config": {"width": width, "height": height,
                                                  "hidden_size": config.hidden_size}},
                "training_config": asdict(config), "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, pt_path)

    if export_onnx:
        try:
            import onnx  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ONNX export requires the training/requirements-kaggle.txt dependencies") from exc
        sample = torch.zeros((1, height, width, 12), dtype=torch.float32, device=device)
        torch.onnx.export(model, sample, str(onnx_path), input_names=["observation"],
                          output_names=["logits", "value"], opset_version=17,
                          dynamic_axes={"observation": {0: "batch"}, "logits": {0: "batch"},
                                        "value": {0: "batch"}}, dynamo=False)
        onnx_metadata = {"file": f"checkpoints/{onnx_path.name}", "sha256": file_sha256(onnx_path),
                         "opset": 17, "runtime": "onnxruntime-web"}
        pytorch_metadata = {"file": f"checkpoints/{pt_path.name}", "sha256": file_sha256(pt_path),
                            "format": "state_dict+optimizer"}
        metadata: dict[str, Any] = {
            "format": "lc-checkpoint/1", "checkpoint_id": checkpoint_id, "step": config.epochs,
            "model_architecture": {"name": ARCH_NAME, "version": ARCH_VERSION,
                                   "config": {"width": width, "height": height,
                                              "hidden_size": config.hidden_size}},
            "observation_schema": OBSERVATION_SCHEMA, "action_schema": ACTION_SCHEMA,
            "onnx": onnx_metadata, "pytorch": pytorch_metadata,
            "artifacts": [{"format": "onnx", "file": onnx_metadata["file"],
                           "sha256": onnx_metadata["sha256"]},
                          {"format": "pytorch", "file": pytorch_metadata["file"],
                           "sha256": pytorch_metadata["sha256"]}],
        }
        metadata["metadata_sha256"] = digest_json(metadata, "metadata_sha256")
        metadata_path = checkpoints / f"{checkpoint_id}.metadata.json"
        metadata_path.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
                                 encoding="utf-8")
    else:
        metadata_path = None

    dataset_path = Path(dataset_path)
    input_manifest = dataset_path.parent / "manifest.json"
    input_manifest_hash = file_sha256(input_manifest) if input_manifest.is_file() else None
    dataset_identity = {"source": dataset_source or dataset_path.name,
                        "sha256": file_sha256(dataset_path),
                        "manifest_sha256": input_manifest_hash}
    report = {"format": "lc-search-policy-value-training/1", "run_id": run_id or checkpoint_id,
              "checkpoint_id": checkpoint_id, "dataset": dataset_identity,
              "dataset_path": str(dataset_path),
              "examples": count, "training_examples": int(len(training_indices)),
              "training_seed_count": int(len(np.unique(data["seeds"][training_indices]))),
              "validation_seed_count": int(len(np.unique(data["seeds"][validation_indices]))),
              "validation": validation, "config": asdict(config), "device": str(device),
              "history": history, "checkpoint": str(pt_path),
              "onnx": str(onnx_path) if export_onnx else None,
              "metadata": str(metadata_path) if metadata_path else None}
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "training-report.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_files = [pt_path]
    if export_onnx:
        output_files.extend((onnx_path, metadata_path))
    output_files.append(report_path)
    run_manifest: dict[str, Any] = {
        "format": "lc-search-policy-value-run/1", "run_id": run_id or checkpoint_id,
        "dataset": dataset_identity,
        "files": [{"path": path.relative_to(output).as_posix(), "sha256": file_sha256(path)}
                  for path in output_files],
    }
    run_manifest["manifest_sha256"] = digest_json(run_manifest, "manifest_sha256")
    (output / "manifest.json").write_text(
        json.dumps(run_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    report["manifest"] = str(output / "manifest.json")
    return report


def train_from_npz(dataset_path: str | Path, config: TrainConfig | dict[str, Any],
                   output_dir: str | Path, *, run_id: str,
                   export_onnx: bool = True,
                   dataset_source: str | None = None) -> dict[str, Any]:
    """Stable programmatic entry point for the Kaggle experimental runner."""
    if isinstance(config, dict):
        config = TrainConfig(**config)
    if not isinstance(config, TrainConfig):
        raise TypeError("config must be TrainConfig or a mapping of TrainConfig fields")
    return train(dataset_path, output_dir, config, export_onnx=export_onnx,
                 run_id=run_id, dataset_source=dataset_source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="compressed distillation examples.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--dataset-source")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-onnx", action="store_true")
    args = parser.parse_args(argv)
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                         hidden_size=args.hidden_size, learning_rate=args.learning_rate,
                         value_weight=args.value_weight,
                         validation_fraction=args.validation_fraction, seed=args.seed,
                         device=args.device)
    print(json.dumps(train(args.dataset, args.output, config, export_onnx=not args.no_onnx,
                           dataset_source=args.dataset_source),
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI in Kaggle
    raise SystemExit(main())
