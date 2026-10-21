# G008 Deep Interview — Competitive Intelligence

Profile: Standard brownfield; threshold <=0.20; final ambiguity score 0.11. Included an explicit pressure pass on training/evaluation contamination and statistical selection. User decisions are requirements; repo facts are separate.

## Starting intent
Move the objective from proving training to improving competitive performance. Reduce Python rollout/simulation overhead, then train against strong heuristics, add imitation learning and diverse self-play, and choose checkpoints by real performance instead of increasing PPO updates blindly. G007: 90.6% vs initialization but only 1.6% vs open-space; rollout 3.195s vs optimizer 0.018s.

## Rounds
1. Success criterion/statistical interpretation. User chose observed win rate >=60% after at least 512 games per opponent, reporting Wilson 95% without requiring lower bound >=60%. Also alternate seats, reproducible seeds, evaluate against adversaries absent from training, report W/L/D separately, and never select on final evaluation.
2. Pressure pass — exact held-out baseline. User confirmed exact open-space policy is fully held out from training, including as imitation-learning demonstrator. Train/imitation must use other heuristic styles; open-space is reserved for evaluation.
3. Regression protection. User chose candidate-vs-incumbent validation on common seeds/seats, per-adversary score W=1/D=0.5/L=0, block statistically significant drop on any primary opponent using paired bootstrap 95% plus multiple-comparison correction. This is validation, not final test.
4. Approval/stop condition. User confirmed >=10,000 env steps/s and observed >=60% against each final-evaluation opponent over >=512 games/adversary are approval gates. If missed after planned optimization/curriculum, report the miss and actual bottleneck; do not call G008 approved.
5. Non-goal/scope boundary. G008 includes an internal opponent pool/league for training/evaluation only; no viewer tournament/league UI.

## Requirements handoff
- Optimize useful end-to-end environment throughput: first profile stages, then compare CPU/vectorization options such as Numba/Rust against current scalar Python. Keep deterministic Python reference and parity tests; adopt only measured wins.
- Remove avoidable per-tick device sync/transfer and Python allocation/loops where profiling justifies.
- Create distinct non-open-space heuristic styles for adversarial curriculum and imitation data; verify exact open-space policy/parameters are absent from training games, demonstrations, labels and opponent schedule.
- Add an internal opponent/checkpoint pool, self-play sampling diversity, and performance-based promotion on a separate validation suite.
- Maintain a locked final suite of unseen adversaries including exact open-space, reproducible seeds, both seats, >=512 games/adversary, W/L/D, Wilson intervals, replay/provenance. Never use it to select/tune checkpoints.
- Final paired validation compares candidate/incumbent per primary opponent with common seeds/seats, W/D/L score 1/.5/0, paired bootstrap 95%, multiple-comparison correction; reject significant regression.
- G008 is not approved unless throughput and win-rate gates both pass. Report each gate and evidence; no unsupported claims.
- No paid services/billing changes. Verify free Kaggle quota/runtime before training; do not waste GPU on an unmeasured 3,000-update extension.
- Out of scope: society simulation, multiplayer/hosted product, viewer tournament UI/full user-facing league. Internal league is allowed as training primitive.

## Facts routed from code
See .omx/context/g008-competitive-intelligence-20261020T002306Z.md. Key: VectorArena is a Python wrapper around scalar engine, collector synchronizes sampled actions to CPU per tick, only open-space/step-0 evaluation and 64 games per matchup exist today, and no Rust/Numba/IL/diverse league exists.

## Downstream planning decisions
Ralplan must specify stage-level profiling and comparable benchmark; candidate acceleration experiments; artifact/evaluation schema versioning for >=512 games and richer opponent sets; no-train/validation/final separation; heuristic styles/dataset generation; promotion/regression tests; Kaggle budget and staged experiment gates; ONNX/browser end-to-end validation.
