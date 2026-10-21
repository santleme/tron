# G008 Ralplan — Architect Review, Cycle 4

**Verdict: ITERATE**  
**Reviewed plan:** `.omx/plans/g008-competitive-intelligence-ralplan-draft-cycle-4.md`

## Summary

Cycle 4 preserves the frozen G007 config hash and user gates and separates `model_rng_seed` from environment roots at the protocol level, but it is missing current trainer migration, injective root-to-initial/reset mapping, paired replayability, and benchmark-exclusion details.

## Evidence

- Cycle 4 draft lines 9–11 and 20–22 keep config seed `1234` for model/PPO RNG, but `training/ppo/train.py` has one seed (`:42`), validation calls it master seed (`:66-67`), it seeds Python/Torch/CUDA (`:313-317`), passes to `_collect` (`:371-372`) and episode initialization (`:200-202,252-255`); initial and reset index formulas differ (`:140-142,201,253-255`), `training/arena/vector.py:17-25` derives from `(master_seed,arena_index)`. Existing test contract requires 64-bit master seed/domain-separated stream derivation/replay provenance (`.omx/plans/light-cycles-arena-test-spec.md:17,22-24`); root+arena is insufficient unless update/reset identity is defined.
- Paired runs need retained fixture bytes/versioned generator, separate canonical state/action digests and transition-output digests, RNG snapshots; scalar seed alone doesn't guarantee same global RNG after alternating order.
- Benchmark scope is ambiguous: draft allows fixed state/action fixture but also 5 repeats × 3 PPO updates; approved hard timer includes init/reset/obs/inference/transfers/orchestration/transitions/PPO optimizer (`g008-plan:55-56`), so fixed-state microbench cannot substitute; explicitly identify hard-gate workload.
- Exclusion must be checked at actual loader/consumer boundaries, not only schema/split labels.

## Precise edits

1. Add runtime migration for distinct `model_rng_seed`/`env_root_seed`; boundaries include config/TrainConfig/RNG init/`_collect`/resets/checkpoint-resume/benchmark/manifests, with versioned G007 legacy interpretation; tests seed roles independently.
2. Define injective root-to-episode derivation including domain labels and update/rollout/arena/reset identity; account for reset formula and disjointness.
3. Pair identities one incumbent/candidate per `paired_repeat_id`; retain canonical fixture bytes/versioned generator; separate state/action hashes, record transition/result digests and model/optimizer/Python/Torch/CUDA RNG snapshot identities; verify before timing.
4. Distinguish simulator fixture diagnostic vs full PPO update-only hard gate; freeze model/optimizer/RNG snapshots and exact replay inputs; G007 historical non-paired.
5. Add executable fail-closed use checks at trainer/imitation/reward-design/validation/promotion/final loaders, with no relabel bypass.

## Gates

Gates remain 10k median, >=60% observed wins each final opponent with >=512 and Wilson95, anti-regression, ONNX/browser, exact open-space final-only.

## Antithesis, tradeoff, and synthesis

- **Antithesis:** Preserving config byte-for-byte is necessary for comparability; manifest-only new semantics might seem sufficient.
- **Tradeoff:** Preserving hash vs operationally distinct/auditable RNG.
- **Synthesis:** Keep G007 config/hash unchanged, version G008 runtime/schema interface with seed roles outside artifact and explicit G007 legacy interpretation.

## References

- `.omx/specs/g008-competitive-intelligence.md:10-24`
- G008 plan `:42-61,89-98`
- Cycle 4 draft `:9-23`
- `.omx/plans/light-cycles-arena-test-spec.md:17,22-24`
- `training/ppo/train.py`: `:42`, `:66-67`, `:140-142`, `:200-202`, `:252-255`, `:313-317`, `:371-372`
- `training/arena/vector.py:17-25`
- G008 plan `:55-56`

