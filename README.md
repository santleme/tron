# Tron

Light Cycles game engine and AI training project. It combines a deterministic
Python arena, a shared-policy PPO trainer, reproducible evaluation and seed
provenance, and a TypeScript replay viewer.

## Project map

- `training/arena/` contains the Python game rules, vector runner, and training
  opponents.
- `training/ppo/` contains PPO self-play, checkpointing, and bounded throughput
  benchmarks.
- `training/eval/` contains fixed evaluation and G008 schedule/provenance
  validation.
- `schemas/` and `shared/fixtures/` define cross-language artifact contracts.
- `apps/viewer/` contains the browser replay and checkpoint viewer.
- `artifacts/circuit-ai/` contains recorded runs and benchmark evidence.

## Run checks

Python 3.10, 3.12, and 3.13 are the CI-supported versions. CI installs the
hash-locked CPU test environment; Kaggle continues to use its supplied CUDA
PyTorch image.

```bash
python -m pip install uv==0.11.7
uv pip sync --system --require-hashes --torch-backend cpu training/requirements-ci.lock
python -m pytest -q
```

Viewer checks use Node.js 22 and 24 in CI:

```bash
cd apps/viewer
npm ci
npm test
npm run typecheck
npm run build
```

See [`training/README.md`](training/README.md) for training, benchmark, and
evaluation commands. Throughput numbers depend on the recorded hardware and
benchmark protocol. A local CPU microbenchmark is not a Kaggle/T4 result.
