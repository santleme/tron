# CIRCUIT AI architecture

## System map

- `training/arena/` implements the deterministic Light Cycles simulator and game contracts.
- `training/ppo/` contains the policy, PPO trainer, benchmark and training schedule code.
- `training/eval/` validates run bundles, replay evidence, split manifests, campaign registrations, and the append-only experiment registry.
- `tools/kaggle/` prepares deterministic self-contained Kaggle kernels, checks their package identity, and gates submission, status, download, reconciliation, and campaign operations.
- `schemas/` defines the cross-language run, replay, checkpoint, and split contracts.
- `apps/viewer/` is the browser viewer for replay inspection and exported model playback.
- `artifacts/circuit-ai/` stores retained runs, evaluation reports, frozen candidates, the current registry, and the exported historical archive.

## Experiment authority

The current registry is `artifacts/circuit-ai/registry/g008-experiment-registry.jsonl`. Its hash chain records registrations, workload attempts, result reconciliation, and promotion state. The seed protocol is `training/configs/g008-split-seed-plan.json`. Current G008 status and operation gates are summarized in [G008 status](G008_STATUS.md) and [experiment policy](EXPERIMENT_POLICY.md).

Historical plans and evidence that originated in ignored runtime storage are kept under `artifacts/circuit-ai/history/`. The migration manifest maps their former paths to neutral paths and records SHA-256 hashes. The historical archive is read-only; current status belongs in tracked docs and the active registry.

The paired G008 throughput benchmark loads its candidate trainer from
`training/ppo/g008_candidate/train.py`, a byte-preserved snapshot bound to the
recorded source commit and SHA-256. Keep that snapshot separate from the active
trainer so later code changes cannot alter a benchmark candidate in place. The
reserved G007 validation opponent is tracked at
`training/eval/assets/g007-step-300.pt`; validation packages include it and bind
it by SHA-256.

## Kaggle package provenance

`python3 -m tools.kaggle kaggle prepare` builds a self-contained kernel locally and does not contact Kaggle. Each new package includes `source-manifest.json` with exact staged `training/` and `schemas/` paths and hashes, run configuration, Kaggle requirements, kernel metadata, revision marker, package-builder hash, runner-template hash, Git commit, and dirty-tree status. The package tree hash binds the full submitted directory, including the generated runner and source manifest. Kaggle runtime evidence reports the source-manifest hash; the registry's package hash identifies the complete submitted bytes. Older records without this readable manifest remain valid historical records; missing provenance must not be inferred.

The Kaggle image supplies PyTorch/CUDA. Heavy training, target-scale simulation, replay sweeps, and hardware performance claims run only on Kaggle under registered and authorized workloads. Local execution is for deterministic unit/integration checks and package validation.

## Artifact categories

See [experiment policy](EXPERIMENT_POLICY.md) for retention and promotion rules. In brief: raw result bytes, checkpoints, frozen rosters, packages, replays, reports, and registry events are immutable; manifests bind them by hash; exploratory artifacts cannot enter promotion; final evaluation uses a fresh protected roster.
