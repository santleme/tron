# G008 Critic Review — Cycle 4 Outcomes Recheck

**Verdict: APPROVE (planning/handoff only)**

## Justification

The previous realized-digest ambiguity is closed consistently in the canonical plan, G008 test specification, and both execution briefs. `episode_outcomes` has an exact six-field schema; exactly one record is required per completed terminal match, none for incomplete matches; `initial_slot_id` is unique and references one consumed initial-game slot; count must equal completed-game count; enums, NFC UTF-8 strings, uint64 ticks, exact key/type validation, and the `invalid` → `other` rule are fixed. Canonicalization sorts by `initial_slot_id` UTF-8 bytes before canonical JSON, so producer iteration order cannot affect the realized payload/digest. Tests explicitly require permutation invariance, outcome sensitivity, and preservation of the parent expanded digest. The Architect outcomes recheck independently approved this change; agent identity binding to game provenance is a bounded consumer validation detail, not an unresolved plan choice.

The three-digest model is now unambiguous: one common immutable `schedule_spec_digest`, a per-run pre-execution `expanded_schedule_digest` over the complete frozen upper-bound roster and collision scan, and a post-run `realized_schedule_digest` bound to its exact parent. The per-run versus common digest distinction, payload keys, slot ordering, collision-group ordering, run/seed identity, parent lineage, and fail-closed rules are specified and covered by the test matrix.

All user gates and boundaries are retained: observed win rate >=60% per final opponent over >=512 games with Wilson 95% reported (no lower-bound >=60% requirement); alternating seats/reproducible seeds and separate W/L/D; validation-only checkpoint selection with paired seed-cluster bootstrap/Holm regression protection; >=10,000 completed joint transitions/s on the frozen single-T4 principal workload; self-contained ONNX validated in a real local browser match; exact `open-space` final-only holdout; Kaggle time/submission/storage/quota limits and final reserve; no unjustified DDP, Rust/PyO3, paid services, or league/tournament UI. The legacy MVP/G007 test spec explicitly retains its historical semantics and delegates all G008 seed/resume/statistical contracts to the G008 test spec and canonical plan.

The seed-plan file is still format v1. This is an explicit unmet prerequisite, not a plan ambiguity: migrate it in place to a hash-frozen v2+ single authority, preserve existing ranges/exclusions/fingerprints and G007 legacy semantics, add Cycle-4 schedule/seed bindings, and prove v1 rejection at every actual consumer before any G008 consumer execution, benchmark, training, or Kaggle work. Do not run the existing benchmark scaffold until it is adapted to and gated by that migration contract.

## Summary
- **Clarity:** Pass. Outcome schema/cardinality/order and all digest scopes are explicit and synchronized.
- **Verifiability:** Pass. Golden codec vector, collision scan, permutation/outcome mutation, consumer rejection, statistical, throughput, and browser gates have concrete tests/evidence.
- **Completeness:** Pass for planning; the v1→v2+ seed-plan migration is a mandatory Stage-0 execution gate, not an approval condition for this plan.
- **Big Picture:** Pass. Scope remains self-play/competitive training and local viewer validation, with final holdouts protected.
- **Principle/Option Consistency (consensus):** Pass; profile-first, Python oracle parity, bounded NumPy/Numba ladder, no presumed DDP.
- **Alternatives Depth (consensus):** Pass; selected and rejected options have concrete rationale and bounded experiment paths.
- **Risk/Verification Rigor (consensus):** Pass; failure gates, frozen schedules, lineage, quota limits, and holdout protections are testable.
- **Deliberate Additions:** Pass; resume-equivalence and provenance/digest layers directly support reproducibility without UI/product expansion.

## Stop condition

Ralplan may hand this approved plan to Ultragoal. This is planning approval only: it does not authorize bypassing Stage 0, claim the v1 migration is complete, or assert any new CUDA, benchmark, training, learning, Kaggle, or ONNX evidence. No implementation or remote execution was performed in this review.
