# G008 status

**Last reconciled:** 2026-10-21
**Authority:** `artifacts/circuit-ai/registry/g008-experiment-registry.jsonl` (177 events; terminal event 176 hash `76838932dbabd5906eafd0f5e89d5f3ff9b788fb6b282c39fed293b13cb2ada2`).

## Latest campaign

`g008-competitive-20261021-v2` completed its registered training and validation workloads and one independent 512-game final evaluation against `open-space`:

| Wins | Draws | Losses | Observed wins | Promotion |
|---:|---:|---:|---:|---|
| 18 | 24 | 470 | 3.5156% | Failed |

The result is far below the 60% target. The checkpoint remains a failed experimental result and is not promoted. The final-only roster was consumed and cannot be reused for a new promotion claim.

## Workload gate

The campaign's registered workloads are consumed. No authorization remains for another training, simulation, benchmark, replay sweep, or evaluation run. An account quota snapshot does not supply project authorization. The next cycle needs a new reviewed campaign, a workload-specific authorization, a fresh final roster, and live budget/preflight checks before any Kaggle operation.

## Next research direction

Use the existing train/validation outputs, candidate checkpoints, diagnostics, and replay evidence to identify the dominant failure modes before selecting a training change. Do not repeat completed analyses when the retained evidence answers the question. Selection must use train/validation partitions; only a frozen candidate may be evaluated on the fresh final roster. No candidate is promotion-eligible until an independent final evaluation covers at least 512 games and observes at least 60% wins.

## Historical records

Earlier plans and campaign reviews are preserved under `artifacts/circuit-ai/history/plans/`, `context/`, `interviews/`, and `archive/`. Treat them as historical decisions, not as current authorization. `artifacts/circuit-ai/history/MIGRATION_MANIFEST.json` provides byte hashes and source-path mapping.
