# G008 Kaggle recovery — Architect review, iteration 1

**Verdict: ITERATE**

The plan preserves the original cap overrun, compute budget, reserve, 10k/s gate, paired snapshots/hash/parity contract and Cycle-5 migration ordering. It needs two reconciliations before Critic review:

1. Remove all fixed-attempt language. The reviewed plan now allows retries but retains phrases implying no retries or a final submission. Normalize all wording to no fixed submission/retry count ceiling, with cumulative workload GPU caps and per-batch, per-kernel, quota and output limits. Every attempt is append-only, and ambiguous live Kaggle status is resolved before retry.
2. Reconcile package changes with immutable registration/lock rules. Define whether retries use the exact package or provide an append-only `retry_attempt`/`recovery_package` mechanism. Any recovery event must bind the same parent registration, run UUID, config, seed/schedule, opponent/root roster, bounds and lock ancestry; record failed attempt/status/reason, patch and new source/package hashes, deterministic validation evidence, remaining cumulative budget and result. Pre-lock benchmark retries stay pre-lock; post-lock training retries pin `lock_head_hash`. No semantic/workload changes.

The user directed removal of the fixed submission/retry ceiling. This review is planning evidence only; it did not authorize code changes, tests, training or Kaggle submissions.
