# G008 Competitive Intelligence — Architect Review (Cycle 4 Realized Digest Recheck)

**Verdict: APPROVE (planning/handoff only)**

The prior schedule-digest blocker is resolved. The plan defines (a) common immutable `schedule_spec_digest` over seed-plan/codec/generation rules; (b) pre-run per-suite/run `expanded_schedule_digest` over an exact ordered upper-bound slot roster and collision-scan result, with preregistered-only statuses; and (c) post-run `realized_schedule_digest` over the parent expansion plus same-order runtime statuses, consumed tuples, and outcomes. Outcome-only mutations preserve the expanded digest and change the realized digest; distinct run IDs/roots/keys share only the common spec digest.

Child lineage, aggregate mappings, duplicate collision ordering, exact paired-side reuse, and consumer fail-closed rules are explicit. The G008 test spec requires tests for repeatability, roster/run-key/order/exclusion sensitivity, outcome-only changes, collision group ordering, and wrong-parent/template-only rejection. Both Ultragoal briefs carry the same digest separation and Stage 0 freeze-before-consumer gate (`plan:56,60`; `test-spec:18–22,39–41`; `brief:13,18`).

The seed-plan JSON remains v1, but the plan/test spec/briefs explicitly block implementation, benchmark, and remote work until an in-place v2+ migration and tests. This is an execution gate, not a planning blocker. Minor non-blocking implementation detail: specify deterministic ordering/schema for `episode_outcomes` records sharing a slot when implementing canonical serialization.

**Antithesis:** Without separate frozen and realized digests, one hash cannot represent both the pre-run reservation and outcome-dependent consumption. **Tradeoff:** retain the immutable pre-run roster for launch reproducibility while adding a post-run digest for actual realization. **Synthesis:** the three-level digest contract preserves both audit goals without changing the G008 approval gates.

Planning-only; this review does not authorize code edits, benchmark execution, Kaggle submissions, or training.
