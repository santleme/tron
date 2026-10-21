# CIRCUIT AI consolidation plan

**Status:** validation and push in progress
**Scope:** architecture, experiment records, reproducibility, CI, and preparation for the next G008 cycle.
**Execution boundary:** no Kaggle workload starts without a new reviewed registration, workload-specific authorization, quota reservation, and live preflight.

## Current verified state

- `training/configs/g008-split-seed-plan.json` is the canonical frozen G008 seed plan (SHA-256 `c6b16733a045e94e7310183e8be8a62cf4327b5d2bdc3eed40bb9d552ce17593`).
- `artifacts/circuit-ai/registry/g008-experiment-registry.jsonl` is the append-only event authority. Its 177 events end at event 176, hash `76838932dbabd5906eafd0f5e89d5f3ff9b788fb6b282c39fed293b13cb2ada2`.
- Campaign `g008-competitive-20261021-v2` consumed its registered train, validation, and final workloads. The 512-game independent final result was 18 wins, 24 draws, and 470 losses against `open-space` (3.5156% observed wins); promotion failed.
- No workload remains authorized in that campaign. The final-only opponent roster was consumed and must not be reused. Old cycle budgets or account quota are not authorization.
- The evidence and historical plans formerly stored under ignored `.omx` paths are being exported byte-for-byte to `artifacts/circuit-ai/history/`; `MIGRATION_MANIFEST.json` records every migrated path, size, and SHA-256. The two existing untracked evidence trees under `artifacts/circuit-ai` are user data and are retained.

## Authority and preservation rules

1. The append-only registry is authoritative for registered experiment lifecycle and recorded results. Never rewrite or reorder its events.
2. A hash-bound campaign plan plus its explicit workload authorization is authoritative for whether an operation may start. A result or quota snapshot never authorizes another run.
3. The frozen seed plan and final roster contracts govern split use. A final holdout that has been inspected is consumed; a future promotion decision needs a newly protected final roster and independent authorization.
4. `docs/G008_STATUS.md` is the current human-readable summary. It summarizes, but does not override, the registry or frozen protocols.
5. Files under `artifacts/circuit-ai/history/` are immutable historical inputs. New interpretation belongs in current docs or sidecar indexes; do not backfill missing old provenance or rewrite prior conclusions.
6. Checksums in `artifacts/circuit-ai/history/MIGRATION_MANIFEST.json` establish the export identity. Validate them before restoration or after transfer. Locks, runtime state, and assistant-session logs are transient operational data and are not scientific records.

## Work sequence

1. Finish the byte-preserving history export, verify every manifest entry, and remove local OMX/skill integration while keeping the project-specific Kaggle rules in `AGENTS.md`.
2. Publish architecture, experiment eligibility, artifact retention, and G008 status documents; correct the contradictory Kaggle history statements without editing result reports.
3. Add source manifests to newly prepared Kaggle packages. Include exact staged source paths and hashes, caller revision, Git commit, dirty-tree indicator, and propagate the manifest hash into runtime evidence. Keep historical provenance gaps labeled as legacy.
4. Align dependency locks and CI with the documented supported matrix; keep training and hardware benchmarks on Kaggle.
5. Run deterministic tests, CI checks, package preparation, registry chain validation, history hash validation, and independent review. Fix regressions and repeat targeted checks.
6. Prepare the next competitive G008 proposal using existing evidence. Do not reuse the consumed final roster or create a workload until a fresh review and operation-specific authorization exist.
7. Commit and push validated changes, preserving user work and historical files.

## Completion criteria

- No project-local OMX runtime, skill tree, or code dependency remains; project Kaggle constraints remain intact.
- Historical evidence and plans are hash-verified at the neutral archive destination; current registry validation succeeds at the recorded head.
- Current docs accurately state the failed v2 result, distinguish exploratory from promotion-eligible evidence, and identify the next authorization gate.
- New prepared packages carry verifiable source manifests and remain deterministic for the same source state.
- CI and local deterministic checks pass, with no local training or target-scale workload.
- A model checkpoint is promoted only after an independent evaluation of at least 512 games reports at least 60% observed wins on a fresh protected `open-space` final roster.

## Remaining scientific objective

The network has not met the target. The next authorized research cycle should diagnose the 2026-10-21 result using existing training and replay evidence, select changes on train/validation partitions only, then evaluate one frozen candidate against a fresh final roster. Until new workload authority exists, only local code, documentation, and lightweight validation are in scope.
