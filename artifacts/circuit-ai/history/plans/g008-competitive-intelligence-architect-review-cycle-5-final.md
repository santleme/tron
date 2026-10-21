# G008 Competitive Intelligence — Architect Review (Cycle 5 Final)

**Verdict: APPROVE (planning only)**

## Exact artifacts reviewed

| Artifact | SHA-256 |
|---|---|
| `.omx/plans/g008-competitive-intelligence-plan.md` | `a63c5b250aae881d2be8b9757820eb12b4bb64c974a2a92f33ce73149724c667` |
| `.omx/plans/g008-competitive-intelligence-test-spec.md` | `f903374d71ff31d830ecdf2ca0902cd7bf3949cadba3f4c95e88462b06244233` |
| `.omx/ultragoal/brief.md` | `053a3888b73d2710782a00a12c76c4f1fd03810696db1b232850b1f1eeeb3fc6` |
| `.omx/ultragoal/g008-brief.md` | `10ef3af7948c9b6f4cc67210b519d1dee487afebb5ffbf7b2761da54bcde0d72` |
| `.omx/plans/g008-competitive-intelligence-ralplan-draft-cycle-5.md` | `740748b53b6330cccdf25a284333c42819ab2ad6318b582278153ad8196b44da` |

## Findings

The cycle-5 amendment resolves the implementation-discovered seed-domain alias and benchmark-root ambiguity without weakening the earlier G008 acceptance gates.

- Migration sequencing is non-circular: Phase A freezes the reviewed `/3+` authority, exact hash, schedule specification digest and collision-scanned expanded rosters. Phase B permits only the minimal migration guard/loader/schema/validator and negative tests at enumerated existing boundaries. Workload implementation/execution begins only after applicable guard tests pass; later consumers must add their own fail-closed guards at their implementation gate (`plan:51,60`; `test-spec:16-20,43`).
- Imitation uses independent `g008/imitation_demo/initial` and `/reset` labels with split-specific dispatch and no aliasing fallback. Its run key is bound to the same canonical UUID as `run_id`, disjoint from training UUIDs. Positive and negative tests, including schedule-spec digest mutation coverage, are explicit (`plan:52`; `test-spec:34,36,45`; draft:12-29).
- Benchmark roots are consistently widened to 5,000,000–5,000,271 inclusive. Measured roots, warmup roots and the permanently unused 5,000,005–5,000,255 gap are unambiguous (`plan:51,54`; `test-spec:18,36`; briefs:13). Train/imitation roots remain preserved and disjoint.
- Previously approved hard gates remain conjunctive: five-repeat median >=10,000 joint transitions/s on the exact frozen single-used-T4 workload; >=60% observed wins against each final holdout adversary in >=512 games with Wilson 95% reported; validation-only paired regression control; self-contained ONNX in a real local browser match; exact `open-space` final-only; budget and no-paid-service constraints; no unproven DDP, Rust/PyO3, or league UI (`plan:13-15,45-49`; `test-spec:8-10`; briefs:7-10; draft:31-33).

**Antithesis:** Repeating seed/codec authority across the plan, test spec, briefs and amendment draft can cause future drift.

**Tradeoff:** Redundant summaries improve independent execution and auditability but impose synchronization cost.

**Synthesis:** Keep the synchronized planning summaries; make the reviewed `/3+` exact plan hash, common schedule-spec digest and expanded-roster verification the single machine-enforced execution authority before any workload.

No code, seed-plan, benchmark, training or Kaggle work was performed. This planning approval does not itself authorize execution; Critic review and the durable Ralplan handoff remain required.
