# G008 Architect Review — Cycle 4

**Verdict: APPROVE**

The cycle-4 revision closes the resume-equivalence gap without changing the principal G007 configuration or weakening any approved acceptance gate. No remaining ambiguity blocks the next planning handoff.

## Reviewed scope

- Cycle-4 revision: `.omx/plans/g008-competitive-intelligence-ralplan-draft-cycle-4.md`, especially lines 19–21, 31, 37–49, and 54–70.
- Approved plan: `.omx/plans/g008-competitive-intelligence-plan.md`.
- Requirements: `.omx/specs/g008-competitive-intelligence.md`.

## Approval rationale

The revised checkpoint contract requires actual save **and restore** of Python, NumPy (or an explicitly tested no-NumPy mode), Torch CPU, and every CUDA RNG state, with retained bytes and content identities. It also captures the environment protocol and schedule cursor, live environment/vector state, PPO sampler/rollout position, model, optimizer/scheduler, counters, and deterministic backend state. Resume restores and validates the complete state before the next observation or action. Process-restart tests compare resumed execution with uninterrupted execution, including subsequent seeds, observations, actions, transitions/rewards/dones, RNG bytes, and post-update model/optimizer state. This resolves the prior save-only/resume ambiguity.

The cycle-4 revision also preserves the approved constraints:

- Benchmark outputs and resulting weights/state remain ephemeral and performance-only; downstream consumers must reject benchmark ancestry.
- The principal config hash remains unchanged; G007 measurements remain historical, non-paired evidence.
- The throughput gate remains five paired unprofiled repeats with three PPO updates each, median at least 10,000 joint transitions/s; an optimization requires at least 10% median paired improvement and at least four of five pairs individually at least 10% faster, with exact parity. Fixed-fixture results remain diagnostic only.
- Environment seed roots and the complete derivation schedule are frozen and collision-checked before execution.
- Exact `open-space` remains final-evaluation-only.
- Original evaluation, statistical, ONNX, Kaggle budget, no-paid-services, and no-unjustified-DDP gates remain in force.

## Decision

**APPROVE** the cycle-4 planning revision and handoff. This is a planning review only; it does not authorize source edits, benchmark execution, Kaggle submissions, or training.
