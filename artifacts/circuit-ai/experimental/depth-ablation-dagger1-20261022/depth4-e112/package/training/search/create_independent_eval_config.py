"""Create a fresh, hash-pinned, nonofficial frozen-candidate assessment config."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
import uuid


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def create(repo: Path, output: Path, *, workers: int = 4) -> Path:
    candidate_path = repo / "artifacts/circuit-ai/experimental/fixed-depth-candidate-20261022/candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_source = repo / candidate["artifacts"]["rust_source"]["path"]
    candidate_wasm = repo / candidate["artifacts"]["wasm"]["path"]
    source_sha = hashlib.sha256(candidate_source.read_bytes()).hexdigest()
    wasm_sha = hashlib.sha256(candidate_wasm.read_bytes()).hexdigest()
    if (candidate["status"] != "experimental-frozen-unsubmitted"
            or source_sha != candidate["artifacts"]["rust_source"]["sha256"]
            or wasm_sha != candidate["artifacts"]["wasm"]["sha256"]):
        raise ValueError("Frozen candidate manifest no longer matches its Rust/WASM artifacts.")

    run_id = f"ind-open-space-{uuid.uuid4().hex[:20]}"
    rng = secrets.SystemRandom()
    protected = ((0, 31), (1_000_000, 1_000_255), (2_000_000, 2_999_999),
                 (3_000_000, 3_999_999), (4_000_000, 4_000_255),
                 (5_000_000, 5_000_271))
    seeds: set[int] = set()
    while len(seeds) < 256:
        seed = rng.randrange(6_000_000, 1 << 53)
        if all(not low <= seed <= high for low, high in protected):
            seeds.add(seed)
    ordered_seeds = sorted(seeds)
    plan = {"format": "circuit-ai-independent-open-space-seed-plan/1",
            "run_id": run_id, "seeds": ordered_seeds,
            "committed_before_execution": True}
    seed_plan_sha = hashlib.sha256(canonical(plan)).hexdigest()
    opponent_files = {name: hashlib.sha256((repo / "training/eval" / name).read_bytes()).hexdigest()
                      for name in ("lc_eval.py", "g008_final_opponent.py")}
    opponent_sha = hashlib.sha256(canonical(opponent_files)).hexdigest()
    config = {
        "run_id": run_id,
        "independent_open_space_evaluation": {
            "seeds": ordered_seeds,
            "seed_plan_sha256": seed_plan_sha,
            "width": 12, "height": 12,
            "max_depth": 3, "edge_budget": 56,
            "workers": workers,
            "candidate_id": candidate["candidate_id"],
            "candidate_wasm_sha256": wasm_sha,
            "candidate_source_sha256": source_sha,
            "opponent_source_sha256": opponent_sha,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "seed-plan.json").write_text(
        json.dumps({**plan, "seed_plan_sha256": seed_plan_sha}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    (output / "run-config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n",
                                             encoding="utf-8")
    return output / "run-config.json"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(create(Path(__file__).resolve().parents[2], args.output, workers=args.workers))
