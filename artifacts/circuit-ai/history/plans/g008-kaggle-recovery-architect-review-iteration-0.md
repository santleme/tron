# G008 Kaggle recovery — Architect review, iteration 0

**Verdict: ITERATE**

The recovery plan closes the Planner's three gaps (train-only scope, explicit preflight freshness, and sizing before registration) and the core architecture is coherent. Four details must be settled before Critic review:

1. **Cycle-5 authority is still pending.** The base plan says Cycle-5 is pending sequential Architect/Critic review and becomes sole authority only after approval (`.omx/plans/g008-competitive-intelligence-plan.md`, lines 1–4); the test spec also remains pending. The recovery plan must not treat the `/3` seed-plan file or this recovery review as upstream approval. Make Cycle-5 approval and exact frozen seed-plan hash a hard predecessor gate before seed consumers, workload implementation, or package creation, or complete Cycle-5 in its own proper sequential review.
2. **Pre-submit exact-version status is impossible for a new attempt.** Kaggle creates the version only after accepting a push. Before submission, query the latest known kernel version and verify terminal status for every prior registered attempt/no concurrent run. After submission, record the push response's new version and poll that exact version. Keep unknown/running status fail-closed. Quota and documentation evidence do not guarantee acceptance.
3. **Post-lock source changes can invalidate candidate selection.** Candidate training is selected using the pre-lock paired benchmark, but recovery currently allows source/package changes in later attempts. After lock, retry only identical source/package bytes. A packaging-only change is allowed only with deterministic equivalence evidence proving the benchmarked backend/trainer bytes are unchanged. Any candidate/backend/trainer code change must occur before lock, consume remaining benchmark budget, and produce a new paired comparison before artifact/lock; otherwise use the unchanged measured incumbent.
4. **Reservation metadata placement is unspecified.** Define where frozen validation/final reservation metadata lives so it does not require an unmodeled event type or become a third recovery workload registration.

## Registry evidence

The current validator rejects unknown event types (`training/eval/registry.py`, `_validate_data`) and knows only legacy registration/lock/artifact lifecycle events. The append-only registry has exactly 13 events, indices 0–12, head `0208abba4954ffd08b01d5069c674e8bc3f5d1817c95ddb0b0bb4eedd41f3fd7`; legacy artifact index 10 is followed by historical registrations 11 and 12, with no lock. Implementation must preserve this prefix and add backward-compatible validation for the recovery amendment, two named registrations, preflight/attempt events, and new artifact→lock adjacency. Legacy artifact cannot be retroactively upgraded.

## Steelman antithesis

Keep the no-count ceiling as requested: cumulative compute, quota, output, wall time, and immutable workload identity are the real protections; a retry counter does not ensure affordability and can obstruct recovery from infrastructure failures.

## Tradeoff and synthesis

Unlimited version attempts improve recoverability while each push consumes platform capacity and may hit undocumented throttling. Official quota/status/docs are evidence, never a guarantee. Retain `submission_attempt_limit: null`, fail closed before each push on cumulative budgets and every known-version terminal status, record each accepted/rejected version and exact post-push status, and continue only under those bounds.

## Verification recommendations

- Add negative tests for missing/unapproved Cycle-5 hash and incomplete seed plan before any consumer/package path.
- Test pre-submit latest-known terminal status separately from post-push exact-version polling, including unknown/running/conflicting responses.
- Test post-lock retries preserve source/package hashes; permit only packaging-only changes with proven trainer/backend equivalence; reject code changes that would detach training from benchmark selection.
- Test that frozen validation/final reservations are bound to a known existing record or exact registration payload and do not create a third recovery workload or unknown event.
- Test legacy prefix verification and the distinct legacy-artifact versus recovery-artifact/lock rules.

No code, tests, Kaggle quota/status calls, or submissions were performed during this review.
