# G008 Critic Review — Cycle 4

**Verdict: ITERATE**

## Justification

The cycle-4 draft is substantially more precise than the preceding revisions: it keeps the principal config hash and original numerical gates, explicitly treats G007 seeds/throughput as legacy non-paired evidence, separates model RNG from environment roots, defines a five-pair × three-update full-PPO throughput gate, requires retained snapshot bytes and input/action/output digests, gives exact benchmark and warmup root rosters, and makes benchmark ancestry unusable outside its ephemeral benchmark invocation. The Architect cycle-4 approval correctly recognizes the resume-equivalence improvement.

Executors still cannot safely treat the canonical handoff as authoritative without guessing or following contradictory artifacts: the reviewed approved plan and both execution briefs remain cycle-3 text, and the seed derivation's byte encoding/conversion is delegated to a future implementer rather than frozen. The existing shared test specification also still describes the old master-seed contract. These are planning integration gaps, not objections to the user's gates.

## Summary
- **Clarity:** Cycle-4 addendum is detailed; its seed derivation is not fully deterministic until canonical encoding and conversion are frozen.
- **Verifiability:** Strong gate and resume assertions; seed/partition collision checks are testable once the exact derivation and schedule expansion are specified.
- **Completeness:** Missing synchronization of canonical execution artifacts and the legacy test contract.
- **Big Picture:** Preserves user-approved holdout, throughput, strategic, budget, CUDA, ONNX and no-DDP constraints. Benchmark artifacts are appropriately performance-only.
- **Principle/Option Consistency (consensus):** Pass on preserving G007 config, holdouts and bounded optimization ladder.
- **Alternatives Depth (consensus):** Pass; NumPy and bounded Numba path retained, Rust/DDP not presumed.
- **Risk/Verification Rigor (consensus):** Partial; exhaustive consumer checks and resume equivalence are appropriate, but seed protocol and document authority need closure before execution.
- **Deliberate Additions:** The expanded state-restore contract is justified for process-restart equivalence; no new product/UI scope.

## Required fixes before APPROVE

1. **Integrate the revision into the canonical execution handoff.** Update `.omx/plans/g008-competitive-intelligence-plan.md` Stage 0/1 and affected acceptance/tests, plus `.omx/ultragoal/brief.md` and `.omx/ultragoal/g008-brief.md`, to incorporate the cycle-4 legacy boundary, RNG adapter/resume contract, root schedule, five-pair full-update protocol, snapshots and benchmark-consumer ban. Explicitly identify the resulting canonical plan revision/hash and state that no Stage 0 implementation may use the superseded cycle-3 wording. The current approved plan still calls each pair's input `seed=1234` and fixed state/action stream, which conflicts with cycle 4's model-seed/environment-root separation.
2. **Freeze the derivation algorithm, not just a requirement to define one.** Specify the canonical tuple encoding (field order, integer width/signedness, text encoding/normalization and framing), domain labels, digest algorithm, conversion to the environment's accepted seed range, and whether uniqueness is required over the expanded finite schedule or injectivity over a larger domain. Include a golden vector and collision-scan rules. This prevents implementation-specific schedules and lets reviewers independently reproduce the lock.
3. **Resolve the existing test-contract conflict.** Add/update a G008-specific test spec or clearly mark the old `.omx/plans/light-cycles-arena-test-spec.md` seed/checkpoint paragraphs as immutable G007/MVP-only and point to the new G008 protocol. The old spec defines one unsigned 64-bit master seed and explicitly does not promise bitwise cross-GPU PPO reproducibility; cycle 4 requires separate model/environment seeds and conditional exact resume equivalence. Document the boundary so implementers do not accidentally satisfy one contract while violating the other.
4. **Keep the benchmark schedule operationally bounded and auditable.** In the canonical plan, require the expanded manifest to enumerate the exact warmup assignment for each registered implementation/backend slot and every initial/reset tuple; specify the exact run order for all five incumbent/candidate pairs (not merely “alternating”); and verify the same initial serialized optimizer/model/RNG snapshots and resulting transition traces before accepting timed results. Reject incomplete schedule expansion before any benchmark run.

## Stop condition

Do not begin source edits, Kaggle runs, benchmark or training from the cycle-3 canonical plan. Re-review the synchronized canonical plan/brief/test contract and frozen seed derivation before implementation handoff. This review is planning-only and makes no claim of execution or gate passage.
