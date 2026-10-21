# Requirements — G008 Competitive Intelligence

**Status:** Deep-interview handoff to Ralplan.  
**Date:** 2026-10-20 UTC.  
**Baseline:** G007 evidence in artifacts/circuit-ai/RESULTS.md; detailed facts in .omx/context/g008-competitive-intelligence-20261020T002306Z.md.

## Intent
Improve both useful rollout throughput and competitive play. G007 proved CUDA updates and learning against initialization, but step-300 won only 1/64 vs open-space. More PPO updates without strategy/performance evidence are insufficient.

## Approval gates
1. **Throughput:** >=10,000 environment steps/s at a documented benchmark workload. Report aggregate arena transitions/s and exact wall-time boundaries, raw rollout, inference/transfer, simulator and optimizer timings. Use comparable G007 benchmark and repeated measurements.
2. **Competitive win rate:** observed win rate >=60% against every required final-evaluation adversary after >=512 games/adversary. Report Wilson 95%; lower bound need not be >=60%.
3. **Evaluation design:** reproducible seeds, alternating target seats, W/L/D separately, adversaries absent from training. Preserve game-level results, replays and provenance.
4. **Checkpoint selection:** never select/tune checkpoints using locked final evaluation. Use separate validation set for promotion/regression decisions.
5. **Regression protection:** on validation, compare candidate/incumbent with common seeds and seats, per opponent, score W=1/D=.5/L=0; paired bootstrap 95% with multiple-comparison correction. Do not promote on statistically significant decline against any primary opponent.
6. **Browser portability:** export selected checkpoint as supported self-contained ONNX; validate real inference/fresh match in local viewer.

These are user-approved gates, not guaranteed outcomes. If any gate remains unmet after the planned sequence, report the miss and measured bottleneck; do not mark G008 approved.

## Training/evaluation isolation
- Exact open-space implementation and parameters are final-evaluation-only. Do not use its opponent games, demonstrations, labels, distillation targets or generated examples for training.
- Training/imitation may use distinct other heuristic styles.
- Use separate training-opponent, validation, and locked final unseen opponent/seed suites. Exact open-space is absent from training and checkpoint selection.
- Validation guides promotions, with paired anti-regression check. The final suite is read-only for the frozen candidate; do not tune after observing it.

## Scope
**In scope:** stage-level performance profile; measured optimization of simulation/rollouts/transfers; evaluate vectorization/Numba/Rust candidates; deterministic parity; heuristic curriculum and imitation path; internal opponent/checkpoint pool; validation promotion; statistical final evaluation/schema changes; ONNX/viewer validation; free Kaggle experiments within verified quota.

**Out of scope:** raising update count alone; unmeasured DDP/two-GPU claims; paid services or billing changes; hosted product/multiplayer; The Grid society; visual tournament/league UI; training against or imitating exact open-space.

## Codebase baseline
- Scalar deterministic engine: training/arena/engine.py; Python-loop batch wrapper training/arena/vector.py.
- PPO rollout and per-tick device synchronization: training/ppo/train.py.
- Current benchmark/eval: training/ppo/benchmark.py, training/eval/lc_eval.py; only two opponents and 64 games per matchup.
- G007 schemas/viewer contract: schemas/, apps/viewer/src/artifacts.ts.
- Full evidence and paths in the context snapshot.

## Ralplan decisions required
Specify (a) stage-level profiling and comparable benchmark; (b) low-risk CPU/GPU-transfer optimizations before heavy port; (c) Rust/Numba/vectorization only after measured end-to-end gain and differential parity; (d) heuristic curriculum and imitation with strict baseline isolation; (e) diverse internal pool and validation promotion; (f) versioned >=512-game evaluation artifacts and final selection boundary; (g) fixed regression protocol; (h) verified free Kaggle budget/stopping rules; (i) real ONNX browser validation.
