# G008 Kaggle Recovery — Critic Rebind Review

- **Session:** `01a15941-194a-7c70-b7ef-ecbf54f67dba`
- **Reviewed plan:** `.omx/plans/g008-kaggle-recovery-plan.md`
- **Plan SHA-256:** `0358c3c00ffa438a3bd033939687827774da5f04ba54746c075eb4680b384d72`
- **Ordered prior reviews:**
  1. Planner: `.omx/plans/g008-kaggle-recovery-planner-rebind-20261020.md` — `a79355fe04e3c06192ec56e127169725dfdd65ef57e502f5883722a8a00547bd`
  2. Architect: `.omx/plans/g008-kaggle-recovery-architect-rebind-20261020.md` — `503baf1841f4abf452e12c84ac4eee1e150e2a9dd778ed633b50d36a2314939f`
- **Verdict:** **APPROVE (planning only).**

## Review

The exact plan hash matches the requested bytes, and both ordered prior review artifacts match their recorded hashes. The plan is actionable: it fixes the two workload identities and bounds, requires immutable registration before execution, defines the benchmark-selection rule, gives explicit append-only event ordering and fail-closed recovery behavior, and ties each retry to terminal reconciliation and full remaining-budget reservation. Its acceptance section supplies concrete positive and rejection tests for registration scope, timeout propagation, push assignment, status identity, crash recovery, lock ancestry, artifact linkage, resource accounting, and package immutability.

The selected option is consistent with the user's explicit request to run both workloads and remove the fixed attempt ceiling. The alternatives are credible and conditional: reopen only for a material constraint change, or reduce frozen workload bounds only if reconciled accounting requires it. The plan preserves the ordinary cap for other campaigns, cumulative GPU/quota/output/time limits, the final-evaluation reserve, and the distinction between the benchmark threshold and overall G008 approval.

The main failure paths are addressed with testable stops: ambiguous push assignment remains unresolved and fully reserved; latest-status polling requires exact-kernel metadata bracketing plus exclusive-writer controls; retries cannot proceed with a running or unknown prior attempt; and budget or quota shortfall blocks another full reservation. The no-outside-writer confirmation remains an actual-preflight gate, as intended.

**Backend identity caveat:** `backend_sha256` is acceptable only as the digest of the declared, canonical execution-backend contract. It does not attest to Kaggle's actual container or installed runtime. The exact contract object and independently recomputed digest must travel together in amendment `reservation_metadata` and bind the registered variants. Actual Kaggle image, Python, PyTorch, CUDA, and accelerator details remain separate runtime evidence collected from each remote artifact. This caveat does not change plan scope or authorize a registry write.

## Stop boundary

This approval binds the exact plan for the current planning review only. It does not itself authorize implementation, registry mutation, package registration, or Kaggle submission. Those remain behind the plan's durable user-authorized handoff, lifecycle/package validation, reconciled budgets, and fresh actual-preflight gates.
