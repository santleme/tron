# G008 Kaggle recovery — Planner review, fresh iteration 0 (revised)

**Recommendation: READY FOR ARCHITECT RE-REVIEW, WITH EXECUTION BLOCKED BY THE CYCLE-5 PREDECESSOR GATE.** The recovery plan now incorporates the fresh Architect findings. Re-review may assess the corrected recovery plan as a plan; no implementation, package, registration, or Kaggle work may start until Cycle-5 lifecycle correction, exact-hash approval, and consumer tests are complete.

## RALPLAN-DR

### Outcome

Complete the authorized paired Kaggle benchmark and one bounded train-only Kaggle workload under the two-registration recovery amendment. Preserve the original G008 gates and two GPU-hour final reserve; report benchmark and training evidence separately from G008 approval.

### Principles

1. The current Cycle-5 `/3` seed-plan candidate (SHA-256 `0173875c2f13fc3ffe5dbae76c2b6d6e46bc8398da7f8a748988a51b345d4b24`) is superseded because its benchmark/lock lifecycle clauses conflict. Correct the seed plan, test specification, and affected draft; sequentially approve the new exact bytes/hash with Architect then Critic; pass applicable consumer tests before any G008 consumer, implementation, registration, or package work.
2. Run target-scale benchmarking and training on Kaggle; local checks remain deterministic/static.
3. Keep the no-count-ceiling exception to exactly two immutable recovery registrations, bounded by cumulative compute, quota, wall-time, batch, and output limits.
4. Bind training to one of exactly two predeclared immutable variants selected by paired benchmark evidence; retries preserve exact registered package bytes.
5. Treat Kaggle quota and status data as evidence, not a guarantee of future acceptance.

### Decision drivers and alternatives

1. Preserve seed-plan, source, package, parity, holdout, and append-only registry provenance while completing the user's requested work.
2. Avoid concurrent or ambiguous Kaggle executions while allowing retries within project and platform limits.
3. Complete benchmark and training without consuming the protected final-evaluation reserve.

| Option | Decision | Reason |
|---|---|---|
| Correct Cycle-5 lifecycle first; then use two named workload registrations and reservation-only metadata inside `recovery_amendment` | Selected | Preserves benchmark-before-lock/train-after-lock ordering and the two-registration boundary. |
| Proceed against current `/3` or infer its intent | Rejected | Its lifecycle is contradictory, and prior approval reviewed earlier bytes. |
| Add reservation events/workloads or post-registration code variants | Rejected | Expands the approved recovery scope and weakens immutable benchmark-to-training identity. |
| Gate training on achieving 10,000 transitions/s | Rejected | The threshold governs overall G008 approval, not the separately requested bounded train workload. |

## Readiness findings

- **Cycle-5 predecessor:** The plan now states that `/3` is superseded, requires a coherent corrected lifecycle and deterministic ordering tests, then new sequential Architect→Critic approvals bound to the exact corrected seed-plan bytes/hash. `recovery_amendment.reservation_metadata` refers to `approved_corrected_cycle5_seed_plan_sha256`; the old hash is prohibited. Execution remains blocked until this gate and applicable `/1`, `/2`, wrong-hash/version, and stale-binding consumer tests pass.
- **Version/status evidence:** Preflight captures the latest known `current_version_number` and status plus statuses for all prior registered attempts. After push, the supported push response must provide an exact positive `version_number`, cross-checked against `kernels list` JSON `current_version_number`; only that version is polled. Missing, non-positive, or conflicting accepted-version evidence remains unresolved. The plan forbids inferring versions by set difference or retrying before explicit reconciliation. `null` is reserved for explicit server rejection before assignment with unchanged current version. Error evidence must be sanitized.
- **Immutable variants/failure path:** Exactly two predeclared hashes are allowed: unchanged incumbent and candidate. No post-registration code/package fix or packaging-equivalence exception is allowed. Candidate defect/parity failure selects the valid unchanged incumbent or stops; a code change requires a separate reviewed cycle with a new pre-lock paired comparison and new artifact/lock.
- **History/event order:** Preserve registry prefix indices 0–12 byte-for-byte and append the amendment at index 13. Benchmark attempts and artifact precede the lock; artifact and lock remain adjacent; training attempts follow lock and bind its exact hash. Reservation-only metadata remains embedded in the amendment, with no third workload/event/registration.

## Risks and verification

- The local Kaggle adapter's current push/status behavior must be checked by the executor against supported CLI/API output; the plan intentionally stops if exact accepted-version evidence cannot be obtained. No public numeric push cap was found in the platform-policy research, but quota is not a project-budget measure or guarantee of acceptance.
- Deterministic tests should cover corrected Cycle-5 event ordering; exact supported push-version parsing and list cross-check; absent/non-positive/conflicting version rejection; explicit rejection with unchanged version; sanitization; prior attempt terminal preflight; no inferred version/retry; exact two variant hashes; candidate failure fallback/stop; no post-registration fix; immutable event prefix; and benchmark-artifact→lock→training chronology.
- Local verification is limited to deterministic tests, static checks, package-byte review, and event validation. Training and target-scale throughput claims require Kaggle evidence.
- Stop if corrected Cycle-5 bytes/hash are not approved, any status/version/lock evidence is ambiguous, project/account resource constraints breach, or the incumbent is not valid when candidate fails.

## Handoff

Return the revised recovery plan to Architect for re-review. Critic follows sequentially on the exact approved plan bytes. Separately complete the corrected Cycle-5 Architect→Critic predecessor and consumer tests before implementation. After all approval gates, hand off to execution for deterministic implementation/tests, package review, and later the explicit Kaggle `--confirm` lifecycle. No implementation or Kaggle submission was performed for this Planner revision.
