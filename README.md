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

Python 3.10+:

```bash
python -m pip install -r training/requirements.txt
python -m pytest -q
```

Viewer checks (Node.js with the version supported by the lockfile):

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
