# G008 Ralplan — Architect Re-review, Cycle 4 Iteration 2

**Verdict: ITERATE**  
**Reviewed plan:** `.omx/plans/g008-competitive-intelligence-ralplan-draft-cycle-4.md`  
**Review scope:** Re-review of the revised cycle-4 draft against the remaining architectural findings. No implementation performed.

## Summary

The revision resolves most prior concerns: it marks G007's historical benchmark as non-paired, separates environment-root RNG from model RNG, makes the full PPO workload a hard gate, and excludes forbidden data at consumer boundaries. The following items remain necessary before approval:

1. **Checkpoint-resume reproducibility:** Persist and validate the environment-root protocol/version and NumPy RNG state across checkpoint save/resume. Tests must show resumed collection reproduces the uninterrupted run, rather than merely restoring model/PPO state.
2. **Replay fixture evidence:** Retain serialized snapshot bytes (or a versioned generator capable of reproducing those exact bytes), not hashes alone. Hashes identify data but cannot reconstruct the paired fixture. Validate the bytes and their identity before timing.
3. **Exact benchmark schedule and collision proof:** Specify the measured and warmup root schedule exactly, including the full derived episode tuple (domain/update/rollout/arena/reset identity), and run a collision/disjointness scan over all derived identities. A root range alone is not a reproducible schedule or proof of non-overlap.
4. **PPO repeat and intervention gate:** State explicitly that each measured repeat executes three PPO updates. Preserve the approved paired-improvement requirement: median improvement at least 10%, with at least four of five paired repeats individually improving at least 10%.
5. **Benchmark return boundary:** Prohibit benchmark-result access by downstream consumers that could use it for training, imitation, validation, promotion, or final selection. Permit only the benchmark's own ephemeral update path where required to perform that update; do not turn this narrow allowance into general access for consumers.

## References

- `.omx/plans/g008-competitive-intelligence-architect-review-cycle-4.md` — preceding review and evidence for RNG migration, paired fixtures, hard-gate workload, and consumer-boundary exclusions.
- `.omx/plans/g008-competitive-intelligence-plan.md:55-56` — approved hard-gate timer boundary.
- `.omx/plans/light-cycles-arena-test-spec.md:17,22-24` — seed derivation and replay provenance contract.
- `training/ppo/train.py:42,66-67,140-142,200-202,252-255,313-317,371-372` — training RNG, episode initialization/reset, and collection paths reviewed previously.
- `training/arena/vector.py:17-25` — existing environment seed derivation reviewed previously.

## Stop condition / handoff

Architect re-review is complete but does not approve the current draft. Resolve the five remaining items and request another review. This is plan-review evidence only; it does not claim implementation or passing benchmark/learning gates.
