# G008 Competitive Intelligence — Architect Review (Cycle 4 Final)

**Verdict: ITERATE**

## Summary

Cycle 4 resolves the critic's major design ambiguities: the G007 principal config remains unchanged, model/PPO seed 1234 is separated from environment roots, the seed codec has a reproducible golden vector, the five measured-pair order and all warmup roots are specified, and benchmark-derived learning state is ephemeral. The hard gates, open-space holdout, validation promotion rule, resource limits, and no-DDP/no-UI boundaries are preserved across the canonical plan, G008 test specification, and execution briefs.

One pre-execution source-of-truth gap remains: the repository's existing `training/configs/g008-split-seed-plan.json` is still format v1 and contains neither the benchmark partition nor the new seed codec/model-seed contract. The plan freezes a new manifest in Stage 0, but does not explicitly require upgrading/replacing that concrete seed-plan artifact before consumers are implemented. Because the current file defines partition membership and fingerprints while acknowledging that it is not a complete game-level manifest, relying on implicit implementation-time interpretation risks a consumer honoring v1 and omitting the new benchmark namespace/derivation rules.

## Evidence reviewed

- Canonical G008 plan: `.omx/plans/g008-competitive-intelligence-plan.md`
- G008-specific test contract: `.omx/plans/g008-competitive-intelligence-test-spec.md`
- Requirements and execution briefs: `.omx/specs/g008-competitive-intelligence.md`, `.omx/ultragoal/brief.md`, `.omx/ultragoal/g008-brief.md`
- Cycle-4 review record: `.omx/plans/g008-competitive-intelligence-critic-review-cycle-4.md`
- Existing seed-plan artifact: `training/configs/g008-split-seed-plan.json`
- Legacy G007/MVP test spec: `.omx/plans/light-cycles-arena-test-spec.md`

## Findings

1. **The canonical plan/test/brief synchronization is substantially complete.** The plan marks earlier generic `seed=1234`/fixture-only wording superseded and makes itself the execution authority (`g008-competitive-intelligence-plan.md:4-16,97`). The G008 test spec explicitly declares the old test spec G007/MVP-only (`g008-competitive-intelligence-test-spec.md:1-8`); the old spec's 64-game/open-space evaluation contract therefore remains historical rather than silently defining G008. Both execution briefs repeat the G008 hard gates and cycle-4 seed boundary (`.omx/ultragoal/g008-brief.md:3-19`, `.omx/ultragoal/brief.md:3-19`).

2. **The config/seed boundary is sound and reproducible.** The principal config retains seed 1234 and SHA-256 `0ec65b7978e26ecb13b893e76ec358d48c41428db96b0a60dd810f9f24f0e484`; the actual G008 benchmark wrapper hash is `8a9436d9cdad8f2f1873965bb96bca536e9577b0b9646b148ae6ba55d81b0681`. The plan explicitly reserves 1234 for model/PPO RNG and keeps undocumented G007 seed semantics legacy-unspecified (`plan:47,50`). The canonical codec fixes prefix, field order, NFC UTF-8 framing, unsigned integer encoding, SHA-256 truncation, run-key mapping, domain labels, and collision handling (`plan:51-52`; `test-spec:14-23`). I independently checked the documented golden vector: 133-byte payload, SHA-256 `e1d2cd7a82ca19bc5277a87aa4819271802e14c23fb9d071f92b6b2e3d539076`, derived uint64 `16272294329707010492`.

3. **The benchmark schedule and ephemeral boundary are operationally specific.** Measured roots 5,000,000–5,000,004, exact I→C/C→I pair ordering, all 16 warmup root assignments, reset upper bounds, and skipped-seed handling are frozen (`plan:53-54`; `test-spec:22-23`). The hard gate is five unprofiled, full PPO repeats × three updates on 128×32 workload; the timer boundary and joint-transition numerator are explicit (`plan:61-69,100-101`). Benchmark outputs/descendants are barred from all later consumers, while only the active benchmark invocation may use its own returns for throughput-only updates (`plan:56-58,103`; `test-spec:36,43-45`). Resume requires saving and restoring RNG, model/optimizer, live environment/schedule position, and deterministic backend state with restart-equivalence tests (`plan:55,103`; `test-spec:32-33`).

4. **User-approved success and safety gates remain intact.** The plan requires exact open-space code/config/data/games final-only and includes it in the frozen final set (`plan:15,57-59,104,107`); ≥512 games per final opponent with alternating seats, W/L/D and Wilson95, observed wins ≥60% (not CI lower-bound ≥60%) (`plan:83,107`; `test-spec:8,37-40`); paired validation with common conditions, seed-cluster bootstrap, one-sided regression test and Holm veto (`plan:80-82,106`; `test-spec:38`); plus browser-validated self-contained ONNX (`plan:85,108`). Kaggle GPU-hour/submission/time/output caps, final reserve, no paid service, no unmeasured DDP, and internal-only pool/no tournament UI are explicit (`plan:45-48,84,109`; `g008-brief.md:9-10`).

5. **Blocking handoff gap — upgrade the concrete seed-plan file before implementation.** The current `training/configs/g008-split-seed-plan.json:1-38` remains `circuit-ai-g008-split-seed-plan/1`, partitions only train/imitation/validation/final, contains no benchmark split or codec/root schedule, and uses only a partial open-space fingerprint; its `seed_semantics` section describes roots but not the G008 `model_rng_seed=1234` separation. The plan mentions freezing manifests and fully expanding the schedule (`plan:43,50-58`), but nowhere explicitly names/requires revising this checked-in seed-plan artifact (the cycle-4 draft had proposed this explicitly). Add a mandatory Stage-0 deliverable/test: publish a new versioned seed-plan manifest (or an equally authoritative successor linked from the old file), bind its hash into run/evidence manifests, encode benchmark/warmup plus existing data partitions and model-vs-environment seed roles, preserve all current roots/fingerprints and G007 legacy semantics, and reject v1 for G008 execution. The implementation must not run until that artifact and expanded collision-checked schedule exist.

## Root cause

The review record now describes a deterministic protocol, but a pre-existing checked-in partition authority still advertises an older incomplete protocol. Since validators and consumers need one frozen partition authority—not merely prose—they could satisfy the new text while continuing to read the old v1 seed plan. This is a planning/handoff inconsistency, not a defect in the specified codec or numerical gates.

## Antithesis / tradeoff / synthesis

- **Antithesis:** Stage 0 already says to freeze manifests and complete the upper-bound schedule; an executor could reasonably infer it must replace the old JSON. The test spec also requires an expanded manifest, so a separate requirement may appear redundant.
- **Tradeoff tension:** Avoid freezing redundant parallel sources of truth, but do not let an existing versioned partition file silently remain authoritative for an incompatible protocol.
- **Synthesis:** Keep the JSON as the single partition/root/fingerprint source, version it forward with the cycle-4 codec and benchmark assignment, and have Stage 0 bind its digest into every manifest. Keep prose as explanation and tests as executable enforcement.

## Recommendation to close review

Revise the canonical plan and briefs to explicitly require a versioned successor to `training/configs/g008-split-seed-plan.json` before code/test consumers or remote work; specify preservation of existing ranges and exact open-space hashes, the benchmark/warmup roots, codec version, model/environment seed split, and the frozen schedule hash. Add a negative test that rejects v1 manifests at G008 consumers. Then re-review the resulting authoritative artifacts. No implementation, benchmark, training, or Kaggle work is authorized by this planning review.
