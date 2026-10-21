"""User-invoked single-device PPO throughput sweep over a bounded config list."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from statistics import median
from pathlib import Path
from typing import Any

from training.ppo.train import TrainConfig, train

MAX_CONFIGS = 5
MAX_BENCHMARK_UPDATES = 10
MEASURED_REPEATS = 3
WARMUP_UPDATES = 1
MAX_JOBS = MAX_CONFIGS * (1 + MEASURED_REPEATS)
ALLOWED_OVERRIDES = {"arena_count", "rollout_steps", "minibatch_size"}


def build_candidates(document: Any, base: TrainConfig) -> list[TrainConfig]:
    """Validate a small list of PPO throughput candidates without loading Torch."""
    if not isinstance(document, list) or not document or len(document) > MAX_CONFIGS:
        raise ValueError(f"configs must be a non-empty list of at most {MAX_CONFIGS} objects")
    candidates = []
    for index, values in enumerate(document):
        if not isinstance(values, dict):
            raise ValueError(f"config {index} must be an object")
        unknown = set(values) - ALLOWED_OVERRIDES
        if unknown:
            raise ValueError(f"config {index} contains unsupported overrides: {sorted(unknown)}")
        config = replace(base, **values)
        config.validate()
        candidates.append(config)
    return candidates


def summarize_repeats(index: int, trial_id: str, config: TrainConfig,
                       device: dict, repeats: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize exactly three warmed measurements with medians and raw evidence."""
    if len(repeats) != MEASURED_REPEATS:
        raise ValueError(f"each trial needs exactly {MEASURED_REPEATS} measured repeats")
    median_rollout_seconds = median(r["rollout_seconds"] for r in repeats)
    median_optimizer_seconds = median(r["optimizer_seconds"] for r in repeats)
    return {"index": index, "trial_id": trial_id, "config": config.__dict__.copy(), "device": device,
            "median_environment_steps_per_second": median(r["environment_steps_per_second"] for r in repeats),
            "median_wall_environment_steps_per_second": median(r["wall_environment_steps_per_second"] for r in repeats),
            "median_update_latency_seconds": median(r["mean_update_latency_seconds"] for r in repeats),
            "median_rollout_seconds": median_rollout_seconds,
            "median_optimizer_seconds": median_optimizer_seconds,
            "median_rollout_to_optimizer_ratio": median_rollout_seconds / max(median_optimizer_seconds, 1e-9),
            "optimizer_updates": sum(r["optimizer_updates"] for r in repeats),
            "cuda_optimizer_updates": sum(r["cuda_optimizer_updates"] for r in repeats),
            "measured_repeats": repeats}


def select_throughput_winner(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick highest median environment steps/s; median latency then index break ties."""
    if not trials:
        raise ValueError("at least one completed trial is required")
    for trial in trials:
        for field in ("median_environment_steps_per_second", "median_update_latency_seconds", "optimizer_updates", "cuda_optimizer_updates"):
            if field not in trial:
                raise ValueError(f"trial missing metric: {field}")
    return min(trials, key=lambda item: (-item["median_environment_steps_per_second"],
                                        item["median_update_latency_seconds"], item["index"]))


def make_benchmark_report(trials: list[dict[str, Any]], device_mode: str) -> dict[str, Any]:
    winner = select_throughput_winner(trials)
    rationale = (f"Selected trial {winner['index']} ({winner['trial_id']}) with the highest observed "
                 f"median update-only environment steps/s ({winner['median_environment_steps_per_second']:.3f}); "
                 f"median wall-inclusive throughput was {winner['median_wall_environment_steps_per_second']:.3f} steps/s. "
                 "Checkpoint I/O is excluded from the selection metric. Median PPO update "
                 f"latency ({winner['median_update_latency_seconds']:.6f}s) is the tie-breaker. Its median rollout/optimizer "
                 f"time is {winner['median_rollout_seconds']:.6f}s/{winner['median_optimizer_seconds']:.6f}s "
                 f"(ratio {winner['median_rollout_to_optimizer_ratio']:.3f}); it performed "
                 f"{winner['cuda_optimizer_updates']} CUDA optimizer updates out of {winner['optimizer_updates']}. "
                 "Selection compares measured throughput on the selected single device; GPU occupancy is not used.")
    return {"format": "lc-throughput-benchmark/1", "device_mode": device_mode,
            "execution_model": "single-device; no DDP or multi-GPU comparison",
            "warmup": {"updates_per_candidate": WARMUP_UPDATES, "discarded": True},
            "measured_repeats_per_candidate": MEASURED_REPEATS,
            "max_training_jobs": MAX_JOBS,
            "selection_metric": "median_environment_steps_per_second (update-only; excludes checkpoint I/O)", "selection_rationale": rationale,
            "selected_trial": winner["index"], "trials": trials}


def run_benchmark(configs: list[TrainConfig], output: str | Path) -> dict[str, Any]:
    """Execute each tiny fixed-update configuration and write a JSON sweep report."""
    if not configs or len(configs) > MAX_CONFIGS:
        raise ValueError(f"configs must contain between 1 and {MAX_CONFIGS} candidates")
    update_counts = {config.total_updates for config in configs}
    if len(update_counts) != 1 or not 1 <= next(iter(update_counts)) <= MAX_BENCHMARK_UPDATES:
        raise ValueError(f"all trials must use the same fixed total_updates in [1,{MAX_BENCHMARK_UPDATES}]")
    devices = {config.device for config in configs}
    if len(devices) != 1:
        raise ValueError("all candidates must use the same selected device")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("benchmark output directory must be new or empty; choose a fresh path")
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    for index, config in enumerate(configs):
        trial_id = f"trial-{index:02d}"
        trial_root = output / trial_id
        warmup_config = replace(config, total_updates=WARMUP_UPDATES)
        warmup = train(warmup_config, trial_root / "warmup", export_onnx=False, run_evaluation=False)
        warm_device = warmup["device"]
        measured = []
        for repeat in range(MEASURED_REPEATS):
            metrics = train(config, trial_root / f"repeat-{repeat:02d}", export_onnx=False, run_evaluation=False)
            if metrics["device"].get("device") != warm_device.get("device") or metrics["device"].get("cuda_name") != warm_device.get("cuda_name"):
                raise RuntimeError("selected device changed between warmup and measured repeats")
            measured.append({"repeat": repeat, "environment_steps_per_second": metrics["training"]["environment_steps_per_second"],
                             "wall_environment_steps_per_second": metrics["training"]["wall_environment_steps_per_second"],
                             "mean_update_latency_seconds": metrics["training"]["mean_update_latency_seconds"],
                             "rollout_seconds": metrics["training"]["mean_rollout_seconds"],
                             "optimizer_seconds": metrics["training"]["mean_optimizer_seconds"],
                             "optimizer_updates": metrics["device"]["optimizer_updates"],
                             "cuda_optimizer_updates": metrics["device"]["cuda_optimizer_updates"]})
        trials.append(summarize_repeats(index, trial_id, config, warm_device, measured))
    report = make_benchmark_report(trials, configs[0].device)
    (output / "benchmark-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True, help="JSON list of arena_count/rollout_steps/minibatch_size overrides")
    parser.add_argument("--output", type=Path, required=True, help="new or empty output directory; bounded to 20 short actual PPO jobs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--updates", type=int, default=3, help="fixed global PPO updates per candidate")
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    config_items = json.loads(args.configs.read_text(encoding="utf-8"))
    base = TrainConfig(width=args.width, height=args.height, total_updates=args.updates,
                       device=args.device, seed=args.seed)
    candidates = build_candidates(config_items, base)
    report = run_benchmark(candidates, args.output)
    print(json.dumps({"selected_trial": report["selected_trial"],
                      "selection_rationale": report["selection_rationale"]}, indent=2))


if __name__ == "__main__":
    main()
