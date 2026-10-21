# G008 Competitive Intelligence — context snapshot

## Task statement
Improve CIRCUIT AI competitive performance while materially increasing useful training throughput. User proposed targets (not guaranteed): >=10,000 environment steps/s; >=60% wins against open-space; >=512 evaluation games per opponent; no statistically significant regression against previous checkpoints; validated ONNX in the local viewer.

## Desired outcome / intent
A rigorously evaluated agent that outplays the current baseline, learned through an efficient and diagnostically sound pipeline—not merely many more PPO updates. Measure the bottleneck, then improve strategy through strong heuristic opponents, imitation learning, diverse opponent self-play, and performance-based checkpoint selection.

## Stated approach and order
1. Profile/optimize Python rollout, CPU↔GPU synchronization/transfers, observations and arena stepping; compare NumPy/vectorization and Rust/Numba experimentally.
2. Establish a supervised/heuristic curriculum, including imitation learning from policies other than the exact open-space baseline.
3. Broaden self-play with a heterogeneous/adaptive opponent pool and checkpoint selection based on evaluation.
4. Evaluate >=512 games/adversary and retain paired comparisons against previous policies; validate exported ONNX in browser.

## Known facts from code and prior runs
- G001–G007 are complete; G007 includes actual Kaggle T4 CUDA evidence and viewer acceptance in artifacts/circuit-ai/RESULTS.md.
- Previous benchmark: about 1,275 env steps/s at 128 arenas × 32 rollout, rollout about 3.195 s/update vs 0.0179 s optimizer; v5 training measured 1,433 wall-inclusive steps/s, 1,200 CUDA optimizer updates, 300 PPO updates.
- Step-300 beat step-0 58/64 (90.6%) but beat open-space only 1/64 (1.6%).
- training/arena/engine.py is a scalar Python reference using immutable states/trail tuples, Python loops/sets and nested observation lists. training/arena/vector.py batches by looping scalar step(); not true vectorization. No Rust/Numba/Cython/torch.compile engine currently.
- training/ppo/train.py transfers observations to device and samples actions then synchronizes to CPU each tick; rollout has Python per-agent/per-arena loops. Actor supports latest-policy self-play plus frozen step-0 mixture; no diverse league or imitation-learning path.
- training/eval/lc_eval.py supports only open-space and step-0, 64 games each (32 seeds × seats). v1 schemas reflect this protocol.
- Current ONNX viewer path is validated; external-data sidecars are rejected. Relevant artifacts: prior PRD/test spec at .omx/plans/, requirements at .omx/specs/deep-interview-light-cycles-arena.md, training/README.md, checkpoint comparison JSON, browser evidence under artifacts/circuit-ai/reports/browser-acceptance/.
- No project root README, G008 goal or G008-specific plan exists yet. Start a new G008 goal rather than modifying historical G007 evidence.

## Constraints
- Treat 10k steps/s and 60% win-rate as proposed targets, not achieved outcomes.
- Do not assume another T4 solves this; do not implement DDP absent measured benefit.
- Do not blindly increase PPO updates. First gather stage-level profile and intervention gains.
- No paid services/cost changes. Kaggle submissions/runs are permitted by prior explicit authorization, but remain within free available quota; do not expose secrets.
- Keep local viewer/ONNX validation and deterministic arena contracts intact; version schemas if statistical protocol changes.
- Maintain personal/local product boundary; do not expand to society simulation, multiplayer service, or full tournament UI.

## User decisions confirmed
- Approval gates: aggregate throughput >=10,000 environment steps/s and observed >=60% win rate against every final-evaluation opponent with >=512 games/adversary. These are gates, not guarantees. If missed after the planned work, report the miss and measured bottleneck; do not mark G008 approved.
- Eval reports W/L/D separately, Wilson 95% intervals, reproducible seeds and alternating seats. Evaluation adversaries must not be used in training.
- Exact open-space policy is fully held out: do not train against it and do not use it as an imitation-learning demonstrator. Train/imitation data comes from other heuristic styles.
- Final evaluation set is locked and never used to select checkpoints. Use a separate validation holdout for promotion/regression checks.
- Compare candidate to incumbent on validation with common seeds/seats and per-opponent score W=1/D=0.5/L=0; reject statistically significant decline on any primary opponent using paired bootstrap 95% and multiple-comparison correction.
- G008 includes an internal opponent pool/league for training and evaluation only; no viewer tournament/league UI.

## Unknowns and decision boundaries for Ralplan
- Define train, validation and final-test opponent/seed splits; ensure final opponents are unseen in all training/selection and exact open-space remains isolated.
- Choose optimization candidates by profiling and measured end-to-end gain/parity, not preference.
- Choose heuristic styles and imitation dataset; avoid memorizing one policy.
- Define benchmark timing boundary comparable to G007 and validation bootstrap/multiple-comparison implementation details.
- Define free Kaggle runtime/experiment budget based on current quota before remote runs.

## Likely touchpoints
training/arena/engine.py, training/arena/vector.py, training/ppo/train.py, training/ppo/benchmark.py, training/eval/lc_eval.py, run/eval schemas and viewer validation, Kaggle run generator/requirements, checkpoint promotion/comparison reports, and G008 experiment docs/artifacts.

## Existing product boundary
Two-agent deterministic grid, Python canonical engine + TypeScript browser parity, PPO/Kaggle training, metrics/replays, local ONNX inference; no full tournament UI or society simulation. G008 may add an internal opponent pool without expanding product UI.

## Prompt-safe initial-context summary
Recorded; full G007 evidence remains in referenced results and bundles.
