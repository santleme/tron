# G008 Competitive Intelligence — Critic Review (Cycle 4 Recheck)

**Verdict: ITERATE (planning-only)**

## Justification

The canonical plan and G008 test spec now resolve the previous cycle's concrete issues: Cycle-4 serialization is byte-specified and its golden vector independently reproduces (133 bytes; SHA-256 `e1d2cd7a82ca19bc5277a87aa4819271802e14c23fb9d071f92b6b2e3d539076`; first uint64 `16272294329707010492`); the G007/MVP test spec explicitly declares itself legacy-only; the five measured roots and I→C/C→I order, exact warmup assignments, reset bounds, benchmark-only PPO artifact rule, holdout statistics, CUDA throughput gate, ONNX browser gate, budgets and no-DDP/no-UI scope are testable and synchronized across the two execution briefs. The current seed-plan JSON is still v1, but the reviewed documents correctly make an in-place v2+ migration and rejection of v1 a hard gate before any G008 consumer implementation or remote run. That is an intentional execution prerequisite, not a reason by itself to reject this plan.

One reproducibility ambiguity remains: the plan/test spec requires a complete expanded seed/reset schedule and says **all** G008 manifests carry the same schedule digest, while also allowing per-run train/imitation UUID `run_key`s, variable train episodes/resets, and distinct validation/final/benchmark suites. It does not define whether that digest covers only a static schedule specification or each instantiated run's actual tuples. A single digest cannot both identify different run-specific expanded tuples and remain identical across all manifests; treating it as a static template digest would not prove the claimed full schedule was collision-scanned. Executors would have to choose an interpretation.

## Summary
- **Clarity:** Strong apart from the global-vs-instantiated schedule-digest contract.
- **Verifiability:** Gates, golden codec, pair ordering, reset upper bounds and statistical procedures are concretely testable; schedule-digest semantics are not.
- **Completeness:** Add an explicit digest scope/binding rule before execution handoff.
- **Big Picture:** User-approved throughput, holdout, learning, budget, ONNX, no-DDP and internal-pool boundaries are preserved.
- **Principle/Option Consistency (consensus):** Pass; preserve the Python oracle, profile first, allow bounded NumPy/Numba only, exclude Rust/PyO3 for this cycle.
- **Alternatives Depth (consensus):** Pass; options and tradeoffs are substantive and the selected bounded ladder is justified by the rollout bottleneck.
- **Risk/Verification Rigor (consensus):** Partial; v1 migration is correctly fail-closed, but schedule identity is ambiguous across run-specific realizations.
- **Deliberate Additions:** Pass; RNG/resume equivalence and benchmark containment directly support reproducibility and do not expand product scope.

## Required fix before APPROVE

1. Define the digest model in the canonical plan, test spec, and both briefs. For example, distinguish a common immutable `schedule_spec_digest` (partition/codec/tuple-generation rules) from a per-run `expanded_schedule_digest` over the exact ordered tuple list and collision-scan result; specify which artifacts bind each digest and which digests must be shared. Alternatively, define one finite, fully instantiated schedule and remove incompatible run-specific variability. Add tests proving a changed run key/root/tuple changes the instantiated digest, identical manifests preserve it, and no consumer treats a template digest as evidence of collision-scanning actual tuples.

## Stop condition

Do not start G008 consumer implementation, benchmark, training, or Kaggle work until the schedule-digest scope is explicit and the existing seed-plan file has been migrated in place to a hash-frozen v2+ authority with the specified schedule and v1 fail-closed tests. This is a plan review only; no source implementation, benchmark, or Kaggle run was performed.
