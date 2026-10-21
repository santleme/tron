# G008 Kaggle recovery — Architect review, iteration 5

**Verdict: ITERATE**

The scoped no-count-ceiling amendment is clear, compute caps remain intact, and event/lock/artifact chronology is coherent. Make the recovery validator executable by specifying the evidence required at each transition:

1. A `recovery_amendment` event must precede and identify exactly the benchmark and training registrations/run UUIDs; each registration binds its hash. Reject a third registration or attempts outside those registrations.
2. Before every Kaggle attempt, append a `workload_preflight` tied to registration, attempt ID and package. It records current Kaggle platform submission policy, quota, live kernel status/no concurrent execution, cumulative GPU seconds, aggregate/per-run output bytes, planned batch and remaining final reserve.
3. The terminal `workload_attempt` links that preflight and any prior terminal attempt. The validator recomputes cumulative GPU/output totals and rejects absent/invalid preflight, unknown/running state, quota/budget/output overflow, or invalid event order.
4. Add negative tests for event order, third/out-of-scope registration, missing/invalid preflight and cumulative limit overflow. Do not add an internal attempt-count ceiling; Kaggle platform restrictions remain a live preflight.

This supplies auditability without reintroducing a fixed attempt barrier. No implementation, tests, training or Kaggle submission occurred.
