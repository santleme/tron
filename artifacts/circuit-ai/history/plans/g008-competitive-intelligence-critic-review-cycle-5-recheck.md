# G008 Competitive Intelligence — Critic Review (Cycle 5 Recheck)

**Verdict: ITERATE (planning only)**

The cycle-5 amendment is substantively aligned and preserves the user-approved G008 acceptance gates. The exact twelve-label registry, separate imitation initial/reset domains, UUID-bound `imitation-<run_id>` key, benchmark range/gap, unchanged golden vector, schedule/outcome digests, and all operational/holdout constraints are present in the canonical plan and execution brief. The current seed-plan JSON is still `/2` with the old benchmark upper bound 5,000,255 and lacks the two imitation labels; this is correctly identified as incomplete, not valid execution evidence. No benchmark, training, or Kaggle activity is authorized by this review.

## Required revisions

1. **Remove the migration-gate sequencing ambiguity.** The draft says no consumer implementation may begin until per-consumer rejection tests exist and pass (`cycle-5 draft:25,29`); the test spec likewise requires rejection tests at “every actual consumer” before implementation (`test-spec:16,39`). Yet those tests require the actual consumer boundaries/guards, and the execution plan asks Stage 0 to bind consumers and add those tests (`brief.md:18`; plan:51,60). Clarify the ordered exception: first freeze `/3+`, its exact hash, `schedule_spec_digest`, and complete collision-scanned expanded rosters; then implement only the migration/manifest guard and tests for enumerated existing consumers (and add mandatory guard tests with any later consumer); pass all applicable `/1`, `/2`, wrong-version/hash, missing/stale-binding tests before enabling any benchmark/training/Kaggle workload. Explicitly state this gate/guard implementation is the only allowed pre-work, not a G008 consumer feature/workload. Alternatively enumerate which existing consumers are in the Stage-0 rejection matrix and defer not-yet-existing boundaries to their implementation gates. This avoids both a circular prerequisite and an unsafe interpretation that workload code can proceed early.

2. **Make the imitation tests exhaustive in the canonical test spec.** The cycle-5 draft explicitly requires both initial and reset derivations and a distinct UUID (`draft:23,29`), but the canonical executable test row only says “test exact train vs imitation_demo domain dispatch and require `imitation-<run UUID>`” (`test-spec:41`). Spell out positive assertions for both imitation initial and reset labels using `imitation-<same lowercase canonical UUID as run_id>`, plus negative assertions for noncanonical/mismatched run IDs and UUID collision with a train run. Assert the schedule-spec digest changes/fails if either imitation label, the split mapping, or the UUID-binding rule is omitted/altered. The broader run-key mapping remains consistent in the plan (`plan:52`) but does not currently make all of these concrete assertions explicit in the canonical test contract.

## Evidence and checks

- Cycle-5 exact domain registry and split dispatch: `plan:52`; draft:13–21,27–29; test-spec:32,41.
- UUID mapping and non-aliasing: draft:23; plan:52. Test-spec:41 should state the same-run-ID binding and train/imitation UUID non-alias explicitly.
- Seed-plan migration and exact-hash rejection: plan:51,60; test-spec:16,39. Current `training/configs/g008-split-seed-plan.json` is `/2`, has only train/benchmark/validation/final labels, benchmark range `[5000000,5000255]`, and warmup roots through `5000271`; it must be rejected until reviewed successor and exact-hash pins/tests are frozen.
- New approved benchmark roots are consistent across plan, test spec and both briefs: inclusive 5,000,000–5,000,271; measured 5,000,000–5,000,004; warmup 5,000,256–5,000,271; reserved gap 5,000,005–5,000,255.
- Schedule digest and outcome semantics are synchronized in plan:56, test-spec:20–22, and both briefs: expanded schedule excludes outcomes; realized schedule binds parent expansion, canonical outcome records, and outcome-only changes do not mutate expanded digest.
- Hard gates and constraints remain: plan:13,15,45–49; test-spec:10; briefs:7,10. These retain 10k transitions/s median over the exact five full-PPO repeats on one used T4, >=60% observed wins per final holdout adversary over >=512 with Wilson95 reported (not requiring its lower bound >=60%), validation-only paired anti-regression with bootstrap/Holm, final-only exact open-space, local browser ONNX, budgets, no unproven DDP, no Rust/PyO3, and no tournament UI.
- Architect cycle-5 recheck is APPROVE (planning only), but it does not resolve the above test-order ambiguity; its documented Stage-0 prerequisites reinforce that the migration gate must be sequenced explicitly.

## Summary

- Clarity: strong on the intended codec and partitions; migration/test ordering remains ambiguous.
- Verifiability: strong overall; imitation positive/negative test assertions need to be made explicit in the canonical test spec.
- Completeness: gates, root assignments, hashes, holdouts and outcome contracts are present; revise the two items above before execution handoff.
- Big Picture: fits G008 and preserves throughput-first plus competitive-learning scope.
- Principle/Option Consistency (consensus): pass; preserves the chosen measured, parity-first path and holdout integrity.
- Alternatives Depth (consensus): pass; cycle-5 changes only the necessary seed identity/domain contract.
- Risk/Verification Rigor (consensus): iterate; gate sequencing currently admits either deadlock or accidental early implementation.
- Deliberate Additions: pre-mortem — the most likely failure is implementers either halt at a self-referential migration gate or silently weaken it; a second likely failure is deriving imitation resets under train domains or a UUID not bound to `run_id`. The revised tests should exercise initial+reset paths, UUID identity/non-alias, exact hash/version rejection at each current consumer, full reserved benchmark roster and gap, and schedule/outcome digest mutation/ordering properties before workload enablement.

**Stop condition:** Do not begin G008 consumer changes, benchmarks, training, Kaggle submission, or final evaluation until the canonical plan/test spec and briefs spell out this migration order and the revised test obligations, receive Architect approval, and then receive Critic approval.
