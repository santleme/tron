# G008 Kaggle Recovery — Planner Rebind Evidence

- **Session:** `01a15941-194a-7c70-b7ef-ecbf54f67dba`
- **Bound approved plan:** `.omx/plans/g008-kaggle-recovery-plan.md`
- **Approved plan SHA-256:** `0358c3c00ffa438a3bd033939687827774da5f04ba54746c075eb4680b384d72`
- **Decision:** retain the exact approved plan. The prior Planner record `d718` is stale and grants no authority to alter or supersede this plan. The current plan hash matches the final Architect and Critic approvals.

## RALPLAN-DR principles

1. Preserve the approved plan byte-for-byte unless a material constraint has changed and is evidenced.
2. Keep benchmark and training claims grounded in their separately bounded Kaggle workloads, recorded artifacts, and measured resource use.
3. Fail closed at lifecycle gates: no registration or submission before contract, package, payload, budget, and preflight checks pass.

## Top decision drivers

1. **Approval integrity:** the current exact plan has matching final Architect/Critic approval; reopening without changed evidence weakens that binding.
2. **Execution evidence:** completion depends on lifecycle tests and frozen package/payload identity, not on planning consensus alone.
3. **Resource accounting:** both workload reservations and final reserve must reconcile against the approved budget before preflight.

## Options considered

- **A — Preserve the exact approved plan (recommended):** continue its existing gates without changing scope or workload. Maintains approval integrity and avoids resetting review; requires completing all outstanding checks.
- **B — Reopen only for a materially changed constraint:** revise and re-approve only if new evidence changes feasibility, safety, or the objective. Preserves change control but costs review time and requires renewed approvals.
- **C — Shrink workloads only if accounting fails:** reduce benchmark/training reservations and their evidence scope, then revalidate. Can restore feasibility but weakens statistical coverage or training opportunity; it is not justified while accounting passes.

## Remaining acceptance gates

- Pass lifecycle contract tests for the paired benchmark and training registration payloads.
- Rebuild and verify exact benchmark/incumbent/candidate package hashes, frozen run configuration, and complete registration payloads.
- Reconcile historical usage, current quota, workload reservations, output-byte caps, and protected final reserve against the approved limits.
- Complete preflight against the exact Kaggle kernel/version and frozen artifacts. Obtain the required runtime-only confirmation that no outside writer is changing the target before the first push.
- Only after all preceding gates pass, follow the approved registration/submission lifecycle and collect terminal Kaggle output evidence.

## Current boundary and stop condition

No new registration has been appended and no submission has been made. This evidence artifact does not authorize either action. Keep the plan unchanged; proceed under option A while all gates pass. If any gate fails, stop before registration/submission and resolve the evidenced failure; consider B or C only under their stated conditions.
