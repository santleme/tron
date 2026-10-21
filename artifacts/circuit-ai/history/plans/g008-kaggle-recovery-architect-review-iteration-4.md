# G008 Kaggle recovery — Architect review, iteration 4

**Verdict: ITERATE**

The retry and event chronology are coherent. One policy-precedence gap remains: the main recovery plan overrides the six-submission limit in the base plan, but `.omx/plans/g008-competitive-intelligence-test-spec.md` also normatively requires `<=6 submissions` and requires registrations to bind applicable submission caps.

Explicitly supersede those test-spec clauses only for the two preregistered recovery workloads and their `workload_attempt` events. Keep all cumulative GPU/quota/batch/wall/output caps and make Kaggle platform submission rules a runtime preflight. Update validators/tests so the scoped exception does not alter other campaigns.

The user removed the fixed submission/retry count. No code, tests, training or Kaggle submission were performed.
