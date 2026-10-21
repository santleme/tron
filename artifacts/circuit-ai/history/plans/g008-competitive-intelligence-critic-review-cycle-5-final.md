# G008 Competitive Intelligence — Critic Review (Cycle 5 Final)

**Verdict: ITERATE (planning only)**

## Exact artifacts reviewed

| Artifact | SHA-256 | Expected |
|---|---|---|
| `.omx/plans/g008-competitive-intelligence-plan.md` | `a63c5b250aae881d2be8b9757820eb12b4bb64c974a2a92f33ce73149724c667` | match |
| `.omx/plans/g008-competitive-intelligence-test-spec.md` | `f903374d71ff31d830ecdf2ca0902cd7bf3949cadba3f4c95e88462b06244233` | match |
| `.omx/ultragoal/brief.md` | `053a3888b73d2710782a00a12c76c4f1fd03810696db1b232850b1f1eeeb3fc6` | match |
| `.omx/ultragoal/g008-brief.md` | `10ef3af7948c9b6f4cc67210b519d1dee487afebb5ffbf7b2761da54bcde0d72` | match |
| `.omx/plans/g008-competitive-intelligence-ralplan-draft-cycle-5.md` | `740748b53b6330cccdf25a284333c42819ab2ad6318b582278153ad8196b44da` | match |

Also read the G008 requirements/context, the prior cycle-5 Critic ITERATE, and the sequential Architect APPROVE. Inspected current seed-plan `/2` and referenced current code boundaries (`provenance.validate_manifest`, `g008_manifest.load_manifest`/`rows_for_usage`, benchmark helpers). The seed plan is correctly treated as incomplete and not execution-ready.

## Findings

The prior two Critic blockers are resolved in the canonical artifacts:

- The migration gate is now ordered and non-circular: Phase A freezes reviewed `/3+`, exact plan hash, `schedule_spec_digest`, and collision-scanned rosters; Phase B permits only guard/loader/schema/validator work and rejection tests at extant boundaries; later consumers add their own guard tests; no workload implementation/execution begins until the applicable gate passes (`plan:51`, `test-spec:14–20`, both briefs).
- Imitation dispatch is explicit for both `initial` and `reset`; its `imitation-<run_id>` key is bound to the same lowercase canonical UUID, distinct from training UUIDs. Canonical tests include positive dispatch, malformed/mismatched UUID, cross-domain and collision negatives, and digest sensitivity to omitted/altered labels, mapping, or binding (`plan:52`, `test-spec:34`, draft:12–29).
- The benchmark range/gap is consistent and exact: measured roots `5000000..5000004`, warmups `5000256..5000271`, unused gap `5000005..5000255` (`plan:51,54`, `test-spec:18,36`, briefs).
- User-approved gates remain intact: one-used-T4 exact frozen workload, five-repeat median >=10k joint transitions/s; >=60% observed wins/opponent over >=512 with W/L/D and Wilson95; validation-only paired bootstrap/Holm anti-regression; exact `open-space` final-only; real-browser self-contained ONNX; budget/no-paid-service constraints; no unmeasured DDP, Rust/PyO3, or league UI. The final plan explicitly defines win rate as wins/games and draws are not counted as wins (`plan:110`).

### Required revision — finite training/imitation schedule bounds are not specified

The immutable schedule contract requires a **complete finite upper-bound roster for every suite** before workload: the `schedule_spec_digest` binds finite root/episode/reset bounds, and the test contract requires materializing and collision-scanning the complete upper-bound roster (`plan:56`, `test-spec:16–20,34`). However, the plan fixes the complete measured/warmup benchmark schedule and evaluation game counts but does not freeze finite counts or allocation rules for `train` and `imitation_demo`: Stage 2 calls for ablations, heuristic curriculum and pool self-play, with a general GPU-hour cap, but no maximum run count/UUID roster, updates/rollouts/arenas per run, episode/reset upper bounds, or deterministic mapping of roots to those runs (`plan:72–79`). A wall-clock/GPU-hour cap does not define the exact seed tuple roster. An executor would have to invent these values, or Phase A could not freeze the complete schedule/digest that the plan makes a precondition to implementation/workload.

Please make one of these choices explicit before approval:

1. Freeze finite per-run and total upper bounds for training and imitation (including max run UUIDs/ablation runs, updates, rollout/arena counts, maximum resets/episodes and root allocation/unused-root behavior), and include them in `schedule_spec_digest` and tests; **or**
2. Narrow the pre-work freeze to immutable schedule rules and require each concrete training/imitation run to pre-register its finite exact roster, UUID, collision scan and `expanded_schedule_digest` before that run's workload starts, while explicitly stating how per-run bounds are chosen and preventing post-outcome expansion/reassignment.

Also ensure validation/final roster upper bounds bind the frozen number and identities of adversaries, not only per-opponent games, if not already guaranteed by the seed-plan schema. Preserve the fixed 512-game gate and final holdout rule.

## Deliberate review additions

- **Pre-mortem:** Most likely failure is schedule generation proceeding with an undocumented training batch/ablation size, causing different seed rosters or allowing a later run to consume roots not included in the reviewed collision scan. The resulting evidence may be reproducible per run but would not satisfy the plan's claim that the complete roster was frozen before workload.
- **Representative execution simulation:** (1) Phase A cannot enumerate all train/imitation slot tuples without knowing run count and update/rollout limits. (2) Stage 2's “equal compute” ablations under only a total 6-GPU-hour budget can produce different numbers of consumed tuples per arm; no predeclared maximum currently ties those outcomes to a finite roster. (3) A newly assigned run UUID after roster freeze changes run identity/slot IDs and expanded digest, but no run-ID preregistration rule is stated.
- **Required tests after revision:** exact roster-size/upper-bound assertions for every training/imitation run; deterministic UUID-to-run registration and rejection of unregistered IDs; fail-closed rejection at the pre-work gate for an omitted/overrun run, new ablation, root reassignment, or slot beyond frozen max; full roster collision scan including unused reserved tuples; schedule digest changes on any bound/root/UUID/order change; ensure final opponent roster size/identity is frozen and disjoint. Keep previously added imitation initial/reset and UUID/digest regression tests.

## Summary

- Clarity: strong for codec, migration ordering, benchmark allocation, holdouts and acceptance gates; incomplete finite training/imitation schedule bounds.
- Verifiability: strong executable test obligations overall; cannot prove full-roster freeze until training/imitation upper bounds and registration are fixed.
- Completeness: one material gap blocks implementation without guessing.
- Big Picture: the selected parity-first optimization and validation-only strategic promotion path remain aligned with G008.
- Principle/Option Consistency (consensus): pass; cycle-5 changes preserve staged profiling, correctness oracle and holdout isolation.
- Alternatives Depth (consensus): pass; amendment is limited to the necessary seed-authority resolution.
- Risk/Verification Rigor (consensus): iterate; full roster/digest requirement cannot be verified for unbounded/unspecified training run schedules.
- Deliberate Additions: pre-mortem and representative schedule-generation/test scenarios above.

**Stop condition:** Planning only. Do not implement seed-plan successor, benchmark/workload, training, or Kaggle activity until the schedule-bound decision is made, synchronized into the canonical plan/spec/briefs, and approved sequentially by Architect then Critic.
