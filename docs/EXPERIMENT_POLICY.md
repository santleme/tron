# Experiment and artifact policy

## Eligibility

- **Exploratory:** local reasoning, unregistered smoke checks, diagnostics, and training/validation experiments. These may guide the next candidate but cannot support a promotion claim.
- **Registered:** an operation bound to a campaign registration, frozen package/source/configuration hashes, seed plan, split, resource limits, and explicit workload authorization. A submitted workload is not complete evidence until its output is reconciled to the registry.
- **Promotion eligible:** an independently executed evaluation against a protected, unused final roster with at least 512 games and at least 60% observed wins against the frozen `open-space` policy. The registry and final report must pass their validators. No checkpoint is promoted on training metrics, validation performance, an incomplete run, or a reused holdout.

The split seed plan determines allowed data use. Training may use train data, candidate selection may use validation, and final data is reserved for one frozen promotion evaluation. Do not adapt a candidate after viewing final outcomes and then reuse that final roster.

## Canonical locations and retention

- Current append-only event authority: `artifacts/circuit-ai/registry/g008-experiment-registry.jsonl`.
- Frozen split and campaign protocols: `training/configs/` and the corresponding schema files under `schemas/`.
- Accepted run bundles, checkpoint metadata, and replay manifests: `artifacts/circuit-ai/runs/`.
- Evaluation reports and diagnostics: `artifacts/circuit-ai/evaluations/` and `artifacts/circuit-ai/diagnostics/`.
- Frozen source packages and candidates: `artifacts/circuit-ai/` with package-tree hashes in their registry registration.
- Historical project records exported from ignored runtime storage: `artifacts/circuit-ai/history/`; see `MIGRATION_MANIFEST.json`.

Never overwrite or prune a checkpoint, replay, roster, package, report, or registry event that contributes to a result or decision. Keep rejected and failed outputs with their status. Temporary local staging may be deleted only after its validated destination and hashes exist. If storage pressure requires archival, copy bytes to a durable destination, verify every manifest hash, update a path map, and only then remove redundant copies.

## Provenance

For new packages, retain `source-manifest.json`, package tree SHA-256, config and seed-plan hashes, trainer/engine fingerprints, Kaggle kernel version, runtime image and Python/PyTorch/CUDA versions, GPU model, run parameters, output manifest, and replay hashes. The readable source manifest records staged code/schema/config/metadata paths and hashes, package-builder and runner-template hashes, source revision, Git commit, and dirty-tree indicator. A dirty checkout is still identifiable by its exact manifest and package hash; the commit alone is never treated as proof of the staged bytes.

Historical events remain immutable. Missing source or runtime links are labeled legacy/incomplete in a current index; do not invent hashes or rewrite historical registry records.

## Retention and migration manifest

`artifacts/circuit-ai/history/MIGRATION_MANIFEST.json` maps each exported source path to its neutral destination with size and SHA-256. To verify the archive from the repository root, run:

```bash
python3 tools/verify_history_manifest.py
```

The manifest intentionally excludes its own hash. It covers the scientific evidence, historical plans, specifications, interview notes, context, and archived project goals. Transient locks, local workflow state, HUD configuration, metrics scratch, and assistant-session logs were runtime data, not model or experiment evidence.
