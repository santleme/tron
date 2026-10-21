# G008 Kaggle recovery — Architect review, iteration 2

**Verdict: ITERATE**

The retry amendment is materially improved and cumulative GPU/quota/runtime/output limits remain enforceable. Resolve two points before Critic review:

1. The plan says candidate parity failure makes that candidate ineligible, but also permits retrying an implementation failure. Define the sequence: a parity-failing attempt is rejected; a corrected candidate may re-enter comparison only after deterministic parity validation, fresh source/package hashes and a `retry_attempt` record under the same immutable workload and remaining budget. Candidate selection still requires the frozen paired gain criteria. Otherwise train the incumbent.
2. Explicitly authorize `retry_attempt` as a post-lock evidence event, distinct from a workload registration. Extend the registry validator so post-lock training retries directly pin the existing lock head and cannot change workload identity, config, seed/schedule, roots, opponents or bounds.

The user removed the fixed attempt ceiling. This review did not authorize implementation, tests, training or Kaggle submissions.
