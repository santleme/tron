# G008 Kaggle recovery — Planner review, fresh iteration 1

**Recommendation: READY FOR ARCHITECT RE-REVIEW.** The recovery plan now closes all four fresh Architect planning blockers. This is readiness for review of the recovery plan only. Execution remains explicitly blocked until the Cycle-5 predecessor artifacts are corrected, approved, and tested.

## RALPLAN-DR

### Outcome

Complete the requested paired Kaggle benchmark and one bounded train-only Kaggle workload under exactly two recovery registrations. Preserve the immutable legacy registry prefix, seed/provenance/holdout contracts, and protected final-evaluation reserve. Separate benchmark/training evidence from the overall G008 approval decision.

### Principles and decisions

1. **Cycle-5 lifecycle authority:** The existing `/3` candidate at SHA-256 `0173875c2f13fc3ffe5dbae76c2b6d6e46bc8398da7f8a748988a51b345d4b24` is explicitly superseded because clauses conflict on whether benchmark execution is pre-lock or post-lock. Correct the seed-plan source, test spec, and affected draft to establish benchmark registration and terminal benchmark attempts pre-lock, artifact immediately before lock, and training post-lock. Add deterministic event-order tests, compute the corrected seed-plan hash, and bind separate sequential Cycle-5 Architect then Critic approvals to those exact bytes/hash. Applicable seed-consumer rejection tests must pass before any G008 consumer, implementation, package, or registration work.
2. **New hash is intentionally TBD:** The corrected seed-plan file does not yet exist as an approved artifact. Inventing or carrying forward a hash before that correction would assert authority over bytes that have not been written and reviewed. The plan therefore uses the named binding `approved_corrected_cycle5_seed_plan_sha256`; fill it only from the exact corrected file after both sequential approvals. The previous `/3` hash must never satisfy the gate.
3. **Version/status lifecycle:** Preflight captures the latest known `current_version_number` and status plus statuses for every prior registered attempt version. It does not claim a complete historical-version inventory or depend on a history API. After push, accept only an exact positive `version_number` from the supported CLI/API response, cross-check it against `kernels list` JSON `current_version_number`, and poll only that version. Missing, invalid, or conflicting assignment remains unresolved; do not infer by set difference or retry before explicit reconciliation. Null is allowed only for confirmed rejection before assignment with unchanged current version. Persist sanitized evidence only.
4. **Immutable implementation options:** Exactly two implementation hashes are predeclared in the benchmark and training registrations: unchanged incumbent and candidate. The candidate is selected only under the frozen parity/gain rule. After registration, no code fix, package rebuild, or packaging equivalence path is allowed. Candidate defect or parity failure selects a valid unchanged incumbent, otherwise stop. Any code change requires a separate reviewed cycle, fresh pre-lock paired comparison, and new artifact/lock.
5. **History and resource boundary:** Keep indices 0–12 unchanged, append the recovery amendment at index 13, include reservation-only metadata inside `recovery_amendment`, and use exactly two recovery registrations. Benchmark attempts/artifact remain pre-lock, artifact and lock adjacent, and train attempts post-lock with exact lock hash. Cumulative compute, quota, wall-time, output, and final reserve bounds remain in force; no fixed retry count is added.

### Alternatives considered

| Option | Decision | Rationale |
|---|---|---|
| Correct Cycle-5 lifecycle/hash and obtain fresh sequential approval before consuming it | Selected | Resolves contradictory authority and makes the predecessor gate auditable. |
| Treat old `/3` as approved or infer which lifecycle clause controls | Rejected | Prior approval did not bind those bytes, and the bytes are internally inconsistent. |
| Infer a missing version from a complete before/after inventory | Rejected | No full-history API is available; inference can target an unrelated or ambiguous version. |
| Repair a registered package in place or add a third variant | Rejected | Breaks immutable benchmark-to-training selection and exceeds the two-registration scope. |

## Evidence and acceptance

- Reviewed exact current contents of `.omx/plans/g008-kaggle-recovery-plan.md` and its prior fresh Planner review. The plan now states the corrected Cycle-5 lifecycle, supersedes old `/3`, gates all downstream G008 implementation/package/registration/submission on new exact-hash Architect→Critic approval and consumer tests, and leaves the corrected hash as a future evidence value.
- The plan now distinguishes latest-known kernel version and prior attempt-linked statuses from a complete historical inventory, requires supported exact push-response version plus `kernels list` corroboration, polls only the resolved exact version, and fails closed on missing/invalid/conflicting evidence without set-difference inference or retry. It restricts null version to explicit pre-assignment rejection with unchanged current version and requires sanitized response evidence.
- The plan fixes exactly two registered variants and removes post-registration code/package fixes and packaging-only exceptions. Candidate failure means unchanged valid incumbent or stop; future code changes require a separate reviewed paired cycle.
- The plan preserves immutable registry indices 0–12, amendment index 13, embedded reservation metadata, benchmark pre-lock/artifact→lock adjacency, and training post-lock validation. Deterministic tests are explicitly required for these event-order properties and the version-assignment negative cases.

## Risks, verification, stop, and handoff

- **Remaining execution predecessor:** Cycle-5 correction and hash are not evidence yet; the current plan correctly blocks execution until new sequential approval and consumer tests pass. This is not a blocker to Architect re-reviewing this recovery-plan revision.
- **Adapter capability:** Before execution, executor must confirm the supported Kaggle push response exposes the exact positive version and that `kernels list` JSON exposes a corroborating `current_version_number`. If not, stop for explicit reconciliation/API support; do not guess.
- **Validation:** Architect re-review the exact recovery plan bytes; Critic review follows sequentially. Later implementation validation must include deterministic Cycle-5 ordering, version parsing/cross-check/negative cases, error sanitization, prior-attempt status gates, exact two-variant enforcement, fallback/stop behavior, and registry chronology. Kaggle run evidence is required for throughput/training claims; local work is static/deterministic only.
- **Stop condition:** Stop on unapproved corrected seed hash, failed consumer binding test, unresolved version/status, changed registered bytes, invalid incumbent fallback, resource/reserve breach, or registry ambiguity.
- **Handoff:** Send the updated recovery plan to Architect for re-review. Keep implementation, package creation, registration, and Kaggle submission blocked until the Cycle-5 predecessor gate and this plan's sequential approvals are complete. No code or Kaggle work was performed for this Planner review.
