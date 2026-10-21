# G008 Kaggle recovery — Planner review, iteration 0

**Recommendation: ITERATE before Critic.** The current draft resolves the five prior Architect rounds, including the iteration-5 validator requirements, but two execution ambiguities remain that could make the approved two-workload boundary impossible to enforce.

## RALPLAN-DR

### Outcome

Complete one paired Kaggle rollout benchmark and one bounded Kaggle training job, preserving the original G008 gates and final-evaluation reserve. A missed 10,000 joint-transitions/s throughput gate is reported independently and does not cancel the requested training.

### Principles

1. Run training and target-scale throughput only on Kaggle; use local execution only for deterministic/static checks.
2. Keep every recovery attempt under the same immutable workload registration and append-only provenance chain.
3. Remove this project's fixed attempt-count ceiling only for the two specifically authorized recovery workloads; preserve live Kaggle/account restrictions and cumulative resource limits.
4. Preserve seed, parity, holdout, validation and final-evaluation contracts; report benchmark/training results separately from G008 approval.

### Decision drivers

1. Produce target-scale paired evidence for the rollout optimization.
2. Complete the requested training while retaining the two GPU-hour final reserve.
3. Keep the historical submission overrun and scoped recovery exception auditable.

### Viable options

1. **Paired benchmark (1 GPU-hour cumulative), then training (2 GPU-hours cumulative): selected.** Meets both requested outcomes; based on 2.34 G008 hours used, preserves 0.66 hours of the six-hour experiment allowance plus the two-hour final reserve, subject to fresh reconciliation.
2. **Lower the registered training allocation before any workload starts.** Preserves both stages if the initial project-usage audit leaves less headroom, at the cost of fewer fixed training updates and a potentially incomplete checkpoint.
3. **Benchmark only or require 10k/s before training: rejected.** Costs less but fails the explicit training request; the throughput threshold is a G008 approval gate rather than a precondition for this authorized bounded training run.

## Evidence against prior Architect findings

- Iterations 1–2: count-free retries and corrected-package handling are now specified; same registration/run UUID and frozen semantic fields, terminal prior attempt, unique attempt ID, patch hash and deterministic validation evidence are required.
- Iteration 3: artifact/lock chronology is specified: benchmark attempts finish pre-lock; artifact binds the immediately preceding prefix and successful attempt; `registration_closed` follows the artifact; a later attestation binds unchanged artifact bytes and original anchors to the lock.
- Iteration 4: the exception explicitly supersedes the six-submission clauses in both the base plan and test spec only for the exact two recovery registrations; other campaigns retain the cap.
- Iteration 5: `recovery_amendment`, per-attempt `workload_preflight`, linked terminal `workload_attempt`, cumulative GPU/output recomputation, and negative-test categories are all present.

## Required clarifications

1. **Training workload boundary:** the plan allows approved train/imitation seeds/opponents, while Cycle-5 requires every concrete train and imitation workload to be separately preregistered. The recovery amendment permits exactly one benchmark and one training registration. State explicitly that this recovery training is train-only and performs/consumes no separate `imitation_demo` workload, or revise the amended scope before review so every required workload is individually registered. Do not leave implementation to infer whether imitation is in scope.
2. **Preflight freshness and evidence source:** `current`/`stale` Kaggle policy and quota evidence is not operationally defined. Name the authoritative command/API or captured source, required fields/version/timestamp, maximum age, and the fail-closed behavior if platform submission policy cannot be verified. Likewise specify the accepted live status evidence and how it proves there is no running attempt. This makes the planned stale/missing-evidence validator tests deterministic.
3. **Budget fallback timing:** the options table permits reducing training bounds, but both registrations are appended before benchmark execution and are immutable. Clarify that any reduced training allocation is selected before amendment/registration; after the benchmark starts, a frozen registration may only be stopped if the remaining budget cannot satisfy its preflight, not edited or re-registered.

## Handoff

Planner recommends that the leader fold the three clarifications into the recovery plan/test obligations, then send the exact revised bytes to the role-specific Architect. After Architect completion, launch Critic sequentially. No implementation or Kaggle action is part of this review.
