# G008 Competitive Intelligence — Critic Review (Cycle 4 Realized Recheck)

**Verdict: ITERATE (planning-only)**

## Justification

The revised three-digest model resolves the earlier common-vs-per-run schedule ambiguity: `schedule_spec_digest` is common, each pre-run `expanded_schedule_digest` binds an exact frozen upper-bound roster and collision-scan record, and the post-run `realized_schedule_digest` binds statuses/consumed tuples/outcomes to its parent expansion. The plan and both briefs specify the expanded payload fields, stable slot ordering, deterministic collision-group ordering, paired-tuple reuse, ancestry/aggregate bindings, and fail-closed consumer requirements. The G008 test spec covers repeatability, roster/run-key/order/exclusion and outcome sensitivity, collision ordering, full-roster scans, and wrong-parent/template-only rejection.

One remaining determinism gap is material to the realized digest: `episode_outcomes` is described as “ordered” and “keyed by slot IDs,” but no exact ordering rule or record cardinality/schema is defined. The JSON canonicalization sorts object keys, not arrays. If a game can emit multiple outcome records for a slot, or records are produced in different runtime order, executors must choose an ordering and may produce different realized digests for the same execution. The test spec verifies that changing outcomes changes the digest but does not verify canonical outcome ordering/permutation invariance or the one-record-per-slot invariant. This is a small, actionable plan fix, but canonical reproducibility is an explicit acceptance contract, so it should be closed before approval.

The seed-plan file at `training/configs/g008-split-seed-plan.json` is still format v1. This is correctly and repeatedly designated as a hard Stage-0 prerequisite: migrate it in place to v2+ as the single authority, preserve ranges/fingerprints/legacy G007 semantics, add required schedule/codec entries, and prove every consumer rejects v1 before any G008 consumer implementation, benchmark, training, or remote work. It is not evidence that the prerequisite is complete, and no such work is authorized by this review. All user gates and boundaries remain intact: exact open-space final-only; >=60% observed wins per final adversary in >=512 games with Wilson95 reported; throughput >=10,000 joint transitions/s under the frozen five-repeat full-PPO gate; validation-only paired promotion; ONNX browser validation; Kaggle caps/reserve and no paid services; no unjustified DDP, Rust/PyO3, or league UI.

## Required fix before APPROVE

1. In the canonical plan, G008 test spec, and both execution briefs, define the realized `episode_outcomes` record schema/cardinality and a total deterministic ordering (for example, one outcome per game keyed by its initial-slot ID, ordered by the referenced slot's position in `slots`; if multiple records per slot are valid, add a stable within-slot key). Add tests proving that logically identical outcomes serialize to the same realized digest independent of producer iteration order, while a changed outcome changes only the realized digest and preserves the parent expanded digest. Keep the exact payload key set and parent binding consistent across all documents.

## Summary
- **Clarity:** Strong for all three digest layers except exact `episode_outcomes` ordering/cardinality.
- **Verifiability:** Required roster/collision/digest tests are strong; realized outcome ordering needs a canonical test.
- **Completeness:** Seed-plan v1 migration is a clear hard gate, not yet complete; one canonical serialization detail remains.
- **Big Picture:** Scope and user-approved statistical, throughput, isolation, Kaggle, and ONNX gates are preserved.
- **Principle/Option Consistency (consensus):** Pass; retain bounded profile-first optimization and Python oracle parity.
- **Alternatives Depth (consensus):** Pass; NumPy/at-most-one-Numba ladder is bounded; DDP/Rust/UI are not presumed.
- **Risk/Verification Rigor (consensus):** Partial; schedule lineage and collision controls are explicit, but nondeterministic outcome-array serialization could invalidate digest reproducibility.
- **Deliberate Additions:** Pass; seed-plan migration and benchmark containment directly support auditability without changing product scope.

**Stop condition:** Keep all G008 consumer implementation, benchmark execution, training, Kaggle submissions, and remote work stopped until the outcome serialization rule is synchronized and tested, then complete the in-place v2+ seed-plan migration and its fail-closed consumer tests before downstream work. This review is planning-only; it makes no claim of CUDA, training, throughput, learning, or ONNX evidence.
