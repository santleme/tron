# G008 Ralplan — Architect Review, Cycle 3

**Verdict: APPROVE**  
**Reviewed plan:** `.omx/plans/g008-competitive-intelligence-ralplan-draft.md`  
**Baseline config SHA-256:** `0ec65b7978e26ecb13b893e76ec358d48c41428db96b0a60dd810f9f24f0e484`  
**Review scope:** cycle-2 findings and the full G008 requirements, interview/context and prior MVP contracts. No implementation performed.

## Findings

The revised plan resolves the cycle-2 blockers:

1. **Principal workload and runtime are fixed.** The plan pins the approval workload to 128 arenas × 32 rollout, 12×12 board, minibatch 1,024 and three PPO updates per measured repeat, with the complete config and canonical hash above. It pins the reference setup to Kaggle single-T4 execution and records the exact device count used. A separate 256-arena diagnostic cannot substitute for or select over the principal workload.
2. **Hard throughput gate and optimization promotion are executable.** Approval requires the median of five unprofiled repeats to reach 10,000 joint arena transitions/s on the pinned workload. A throughput intervention additionally requires a paired median gain of at least 10% and at least four of five paired repeats each at least 10% faster. The plan fixes warmup, repeat count, seed/input, alternating order and timer boundaries.
3. **The baseline is supported by primary local evidence.** `artifacts/circuit-ai/runs/lc-circuit-ai-smoke-r4/benchmark-results.json` records trial 2 at 128 × 32 / minibatch 1,024, three repeats, median update-only throughput 1,275.171 transitions/s, wall-inclusive 1,272.767, rollout 3.194609 s and optimizer 0.017926 s. Its config hashes to the pinned SHA-256. `artifacts/circuit-ai/RESULTS.md` records Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8 and one used Tesla T4 (two visible); the plan requires checking the current Kaggle image and does not treat historical quota as current.
4. **Other architecture controls remain sound.** Exact `open-space` and final seeds/opponents are isolated by hashed split manifest and included only in final evaluation; validation promotion uses paired seed clusters, explicit bootstrap and Holm procedures; the optimization ladder and Kaggle budget/recovery limits are bounded; old G007 schemas remain compatible; the opponent pool is internal, with no league UI.

## Stop condition / handoff

Architect review is complete and approves the current plan for the sequential Critic review. This approval is architectural review evidence only; it is not a claim that G008 has been implemented or that any throughput/competitive gate has passed.
