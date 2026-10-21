# G008 Kaggle recovery — Planner review, fresh Ralplan session

**Recommendation: READY FOR ARCHITECT REVIEW; the latest recorded Architect review approves the plan for planning only, so the next sequential review is Critic. Execution remains blocked by the explicit Cycle-5 predecessor gate.**

## RALPLAN-DR

### Outcome

Run the requested paired Kaggle benchmark and one bounded training workload through exactly two recovery registrations, within cumulative compute and resource caps. Preserve G008's existing provenance, validation, holdout, and final-evaluation requirements; report workload evidence separately from G008 approval.

### Findings

- **Scope and retries:** The amendment permits exactly two new recovery workload registrations: one benchmark and one training run. It removes the submission/attempt count ceiling only for these registrations; cumulative GPU, quota, batch, wall-time, output, and reserve caps still apply. No third workload or registration is allowed.
- **Cycle-5 predecessor:** The existing `/3` seed-plan candidate hash `0173875c2f13fc3ffe5dbae76c2b6d6e46bc8398da7f8a748988a51b345d4b24` is explicitly superseded because lifecycle clauses conflict. The plan requires correcting the seed-plan, test spec, and affected draft; proving benchmark attempts/artifact pre-lock and training post-lock; computing a new exact hash; then obtaining separate sequential Cycle-5 Architect and Critic approvals plus consumer rejection tests before any seed consumer, implementation, package, registration, or Kaggle work. The corrected hash is intentionally a future evidence value; using the old hash would falsely bind uncorrected bytes.
- **Kaggle version/status evidence:** Preflight records latest known `current_version_number`, its status, and status for all prior attempt-linked versions; the plan does not assume a complete historical-version inventory or API. After push, require an exact positive version in the supported response, corroborate against `kernels list` JSON `current_version_number`, and poll only that exact version. Missing, invalid, conflicting, or ambiguous evidence fails closed with no set-difference inference or retry. Null is limited to explicit pre-assignment rejection with unchanged current version; response/error evidence is sanitized.
- **Variants and failures:** Benchmark and training registrations predeclare exactly two immutable implementation hashes: incumbent and candidate. No post-registration code/package fix or packaging-only exception is allowed. Candidate defect/parity failure means use the unchanged valid incumbent or stop; any code change needs a separate reviewed cycle and new pre-lock paired comparison/artifact/lock.
- **History and event order:** Preserve legacy registry events 0–12 byte-for-byte; append the amendment at 13. Reservation-only validation/final metadata lives inside that amendment, not a new registration or event. Complete benchmark attempts pre-lock, append its artifact immediately before the lock, then run training post-lock bound to the exact lock hash.

### Alternatives considered

| Option | Decision | Reason |
|---|---|---|
| Correct Cycle-5 lifecycle/hash before any recovery implementation | Selected | Resolves the contradictory predecessor and supplies an auditable exact-hash approval gate. |
| Carry forward old `/3` or infer a missing Kaggle version | Rejected | Neither is backed by valid exact-byte/version evidence. |
| Add another registration or repair a registered package in place | Rejected | Violates the two-registration boundary and immutable variant selection. |

## Acceptance, risks, and handoff

- Current recovery plan satisfies the requested planning conditions, and `.omx/plans/g008-kaggle-recovery-architect-review-fresh-iteration-1.md` records **APPROVE (planning only)**. Critic should review the exact current plan bytes next.
- Execution prerequisites remain: corrected Cycle-5 seed-plan bytes/hash, sequential Architect→Critic approvals, applicable consumer tests, and local verification that supported Kaggle CLI/API responses provide the required exact version and corroborating JSON current version. If version evidence cannot be obtained, stop and reconcile explicitly.
- Later deterministic verification must cover version parsing/cross-check/failures, sanitized rejection, prior-attempt status, exactly two variant hashes and fallback/stop, immutable 0–12 prefix, amendment/reservation metadata, benchmark-artifact→lock adjacency, and post-lock training ancestry.
- No code, package, registry, quota, training, or Kaggle work was performed in this Planner review.
