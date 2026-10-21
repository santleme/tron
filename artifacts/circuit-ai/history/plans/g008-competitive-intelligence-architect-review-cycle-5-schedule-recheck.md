# G008 Competitive Intelligence — Architect Review (Cycle 5 Schedule Recheck)

**Verdict: APPROVE (planning only)**

## Exact artifacts reviewed

| Artifact | SHA-256 |
|---|---|
| `.omx/plans/g008-competitive-intelligence-plan.md` | `5d97fe0441c7e176ee03e6fee1491d186d99bc04b3b30da8221178d0a7b078b0` |
| `.omx/plans/g008-competitive-intelligence-test-spec.md` | `30fcb15fdb4a1f7ea9693ccf7920514684897e70431c6d6ab6bd0186ee02f0aa` |
| `.omx/ultragoal/brief.md` | `8e001d47335d1e0ba17d730e7a78e688cf891b65fde122a3c753f667b61e4210` |
| `.omx/ultragoal/g008-brief.md` | `3ecdc84a0055be884a6c553a6716076ccfa9c153646fdacfb875dcba0a2f04fa` |
| `.omx/plans/g008-competitive-intelligence-ralplan-draft-cycle-5.md` | `373ae61d6e36402025912c5f72641a592a89f9c71b4e6d2fe5386d43da2fa487` |

## Findings

The planner revision closes the finite-roster blocker while keeping global rules distinct from contingent run instances. Phase A freezes the immutable global schedule rules/digest and complete benchmark, validation and final schedules. Validation/final identities and counts are frozen, while every concrete train/imitation run must be registered before workload with unique UUID, hashes, finite bounds, exact roots, complete upper-bound roster, collision scan and run-specific expanded digest. Completion binds a separate realized digest. Outcomes cannot authorize another run, ablation, expansion or root reassignment (`plan:51,56`; `test-spec:14-24`).

The global-vs-run-specific digest split is coherent: the common `schedule_spec_digest` binds immutable rules, while contingent run identity/bounds are bound by each pre-registered `expanded_schedule_digest`; outcome records affect the realized digest only. The updated test obligations reject unknown IDs, bounds overruns, unregistered ablations, root reassignment, post-outcome registration, omitted roster slots and digest mutations, including final opponent roster identity/count (`test-spec:14-24,34-36`).

The preceding cycle-5 changes remain intact: imitation initial/reset domains and run UUID binding, non-aliasing and digest coverage; ordered Phase A then limited Phase B migration guards; and frozen benchmark roots with unused gap. The original acceptance gates and constraints remain unchanged: exact open-space final-only, >=10,000 joint transitions/s, >=60% observed win rate for each >=512-game final opponent with Wilson95, validation-only paired anti-regression, browser ONNX, fixed Kaggle budget, no paid services, no unmeasured DDP, Rust/PyO3 or league UI.

**Antithesis:** Per-run registration allows contingent training schedules and could be abused to expand after observing results.

**Tradeoff:** Freezing all future UUIDs/runs up front would be brittle; per-run preregistration supports bounded experiments but requires strict no-post-outcome registration and digest enforcement.

**Synthesis:** The revised protocol is acceptable: freeze immutable rules and holdout suites globally, register each train/imitation workload with its entire finite collision-scanned roster before it starts, forbid outcome-driven follow-on registrations, and bind outcomes only in the realized digest.

No implementation, seed-plan update, benchmark, training or Kaggle activity was performed. This is planning-only approval; sequential Critic approval and durable Ralplan handoff remain required.
