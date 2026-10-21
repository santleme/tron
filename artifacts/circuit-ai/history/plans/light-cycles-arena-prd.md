# Product Requirements — Light Cycles AI Arena MVP

**Status:** Planning artifact, revision 1 for Architect re-review  
**Source of truth:** `.omx/specs/deep-interview-light-cycles-arena.md`; interview `.omx/interviews/light-cycles-arena-20261019T211321Z.md`; decisions recorded in `.omx/plans/light-cycles-arena-ralplan-draft.md` and `.omx/plans/light-cycles-arena-plan-revision-1.md`.

## 1. Product outcome
A personal, local-first Light Cycles arena that lets its owner train two neural agents through self-play, use downloaded trained checkpoints in fresh browser-local matches, and inspect training metrics/replays. Kaggle is the user-invoked training environment. This is a development/training project, not a hosted product.

## 2. User and primary workflows
**Primary user:** project owner, training on Kaggle and using the trained policy locally.

1. **Train remotely (explicit user action):** prepare reproducible run, explicitly submit a selected kernel/source revision, monitor that kernel version, download an explicitly selected output version, validate artifacts.
2. **Inspect training evidence:** open local artifact library, review evaluation metrics/replays and training curves/checkpoint identity.
3. **Play a fresh local match:** choose a downloaded checkpoint, initialize two policies in the browser (or checkpoint vs selected self-play opponent snapshot), run inference locally in the browser and step the deterministic browser engine; save/reopen replay.
4. **Benchmark throughput:** train/headless-benchmark vectorized arenas and PPO updates; select GPU configuration using measured throughput.

## 3. MVP functional requirements

### Arena and deterministic contracts
- Exactly two AI agents per match; one life each. Rectangular board, cardinal grid movement, no reverse action.
- Both policies receive the same pre-tick observation snapshot and choose before either moves.
- Resolution order per tick: (1) compute both proposed destinations; (2) an agent is collision-dead if destination is outside the board, is any cell occupied by either agent's trail at tick start (including either current head), or action is invalid/reverse; (3) if both agents propose the same destination, both die; (4) if they propose a direct head swap, both die; all causes are accumulated before state mutation; (5) append destinations only for survivors and commit both deaths simultaneously. If both die in the same tick, match is a draw; if exactly one dies, the other wins; if neither dies, continue. No collision priority is based on agent index. Already-dead agents never act again.
- Seeded episode initialization and resets use a specified PRNG/stream scheme: unsigned 64-bit master seed; derive independent episode/arena and agent/environment substream seeds with SHA-256 over canonical domain-separated UTF-8 labels plus little-endian indices; use PCG-XSH-RR 64/32 in both Python and TypeScript: multiplier `6364136223846793005`, 64-bit wraparound, odd increment `(initseq << 1) | 1`, and the reference two-step seeding/output rotation procedure; split SHA-256 digest into little-endian 64-bit `initstate`/`initseq` as specified in test vectors. Record master seed, arena index and derived stream labels/seed values. The determinism guarantee is exact game-state/action/replay equality for the same schema version and implementation contract; it does not promise bitwise equality of PPO training across GPU models/framework versions.
- Model action set order is `UP, RIGHT, DOWN, LEFT`; observation has a versioned fixed shape and orientation defined in interface fixture before implementation. Evaluation inference is deterministic (eval mode, argmax, fixed tie-break to lowest action index); no sampling/dropout.

### Learning and evaluation
- Implement PPO self-play; Kaggle PPO optimizer updates must execute on CUDA when supported GPU runtime is present. Local CPU smoke is supported.
- Vectorize headless independent arenas and compact rollout buffers. Compare one vs two GPUs only when both actually available; choose by measured useful end-to-end throughput, not occupancy.
- Fixed evaluation protocol v1: held-out master seeds are integers `0..31`, never used for training; for each matchup play both seat assignments for each seed (64 games per matchup). Compare selected target checkpoint against (a) fixed deterministic baseline `open-space`: choose legal destination maximizing count of unoccupied reachable cells in a 4-neighbor flood fill, tie-break by action order; and (b) step-0 self-play checkpoint. Match timeout is not expected because trail space is finite; if engine invariant is violated, record error rather than truncate silently. Report wins/draws/losses, win rate with Wilson 95% interval, mean/median survival ticks, collision causes and all 64 match seed/seat results. Evaluation success criterion is production of the complete reproducible report and replays—not statistically significant improvement or guaranteed learning.
- Raw outcomes remain separate from PPO shaped reward. No learning-improvement claim without evidence.

### Frozen artifact interfaces (planning contract v1)
All numbers are JSON integers unless specified; canonical JSON for digests is UTF-8, sorted object keys, no insignificant whitespace, no NaN/Infinity, arrays preserve order, strings Unicode NFC, and numeric values are integers or finite decimals rendered by the implementation's agreed canonical JSON library. Hash is SHA-256 over canonical content excluding the digest field itself. Validate digest before use.

**Run manifest (`lc-run/1`) required keys:**
- `format`: `lc-run/1`; `run_id`: stable run identifier; `created_utc`.
- `engine_version`, `observation_schema`, `action_schema`, `reward_schema`, `replay_schema` (all explicit strings/versions).
- `master_seed`, `training_config` (including arena count, rollout/update settings and self-play opponent/checkpoint sampling), `training_steps`.
- `checkpoints`: array of `{checkpoint_id, step, file, sha256, format, observation_schema, action_schema}`; include ONNX model for browser inference and native PyTorch checkpoint for resumed training.
- `evaluation`: `{protocol: "lc-eval/1", held_out_seeds: [0..31], games_per_matchup: 64, opponent_ids, metrics_file, replay_files}`.
- `software`: `{python, torch, cuda, onnx_exporter, onnx_opset, kaggle_cli}` when applicable; `device`/hardware fields where available, and benchmark records.
- `files`: list of relative artifact paths and SHA-256 digests; `manifest_sha256` canonical digest.

**Replay (`lc-replay/1`) required keys:** `format`, `run_id`, `match_id`, `engine_version`, observation/action schema versions, `master_seed`, `arena_index`, derived stream identifiers, `checkpoint_ids`/agent IDs, initial grid/positions/headings, ordered per-tick joint actions (including explicit invalid-action token if any), per-tick terminal cause(s), final result, tick count, and canonical digest. Replay stores joint actions and collision causes; viewer reconstructs frames using its engine. The cross-runtime golden corpus is mandatory, and unsupported engine/replay versions fail visibly. Do not add compression/dependencies until replay size is benchmarked.

**Checkpoint interface:** native PyTorch weights/config for training resume plus ONNX inference model for the browser. Both share `checkpoint_id`, training step, model architecture/version, observation/action schemas, and SHA-256 in manifest. Browser accepts only ONNX models whose declared opset/runtime support has been validated; browser inference output is deterministic argmax. Export/load/action parity is required before artifact promotion. Do not place secrets or Kaggle credentials in artifacts.

### Viewer and local inference
- Local browser viewer loads validated run manifests, metrics and replay files; displays replay and supports play/pause/restart/scrub.
- Browser can load ONNX checkpoint and run fresh matches locally without Python inference service or network access. MVP game uses two agent policies with selected checkpoint(s); default may pit checkpoint against its designated opponent snapshot or itself with independent seeds.
- TypeScript browser engine implements same contract as Python canonical/training engine. Python is authoritative reference; shared golden vectors and differential tests protect parity for initialization, observations, legal action ordering, action choice, collisions, terminal states, and replay digest. This is accepted complexity to satisfy browser-local inference.
- Glow/trails/particles/cinematic camera are presentation only; the game-state/replay contract is unaltered by effects.
- No login, hosted service, human-vs-AI, or full tournament requirement.

### Kaggle automation and side-effect boundary
- Kaggle CLI stays behind an isolated adapter. Wrapper commands are: `arena kaggle submit --kernel-id <owner/slug> --source-revision <git-sha-or-explicit-local-revision> --confirm`; `arena kaggle status --kernel-id <owner/slug> --version <n>`; `arena kaggle download --kernel-id <owner/slug> --version <n> --run-id <id>`.
- Only `submit` is side-effecting. It is never called from app startup, build, training tests, viewer or default command; missing `--confirm`, ambiguous/empty kernel ID, source revision mismatch or an already-running target must fail without submitting. `--dry-run` is default for tests/examples and prints a redacted command plan. Confirmed submit validates metadata target matches explicit kernel ID and invokes documented `kaggle kernels push -p <dir>`; record returned/observed kernel version/run identifier. The wrapper confirmation is the intentional action boundary.
- Status/download are read-only against explicitly selected kernel and version. Poll running status at bounded interval/deadline; terminal success allows download, terminal error/cancel or timeout returns actionable status and never silently retries submission. Download output into staging directory under artifact library; validate manifest/schema/digests/required files and run ID/version; atomically rename to final run directory only after all checks. Existing target directory fails closed (no partial overwrite); incomplete/invalid download leaves prior artifacts untouched and retains/removes staging according to a documented cleanup policy.
- Resolve credentials only at explicit user invocation using the official supported Kaggle CLI configuration; never copy secrets to repository/artifacts, log environment or secret-bearing args, or require auth for tests.

## 4. Non-goals
Human-vs-AI, full checkpoint tournament organizer, The Grid society/citizen/campaign/RPG simulation, SaaS/accounts/hosted service/multiplayer/social platform, mandatory dual GPU, remote job execution by this delivery or by tests.

## 5. Architecture direction
Python-canonical rules, PPO/training, evaluation, export and artifact production; TypeScript browser engine + ONNX Runtime Web policy inference for fresh local matches and replay viewer. Shared language-neutral fixtures pin observation/action/rules/artifact contracts. Portable ONNX and Python checkpoints are both emitted. This intentionally accepts a bounded dual-engine parity obligation rather than weakening explicit browser-local inference. Validate ONNX opset/browser runtime/device behavior before committing a dependency/version during implementation; if browser model inference proves infeasible, return to Architect/user with concrete evidence rather than silently dropping the requirement.

## 6. Acceptance criteria
1. Browser loads a downloaded ONNX checkpoint and runs a fresh seeded two-agent match with inference local to the browser; it can save/display a replay and play/pause/restart/scrub it. No network/Python inference required after artifacts are present.
2. For fixed contract version/seed/config, Python and TypeScript engine match initial state, observations, legal action order, deterministic policy action, every transition, terminal causes/result and canonical replay digest on golden corpus; exact parity guarantee is scoped to contract/schema version, not PPO cross-GPU training bits.
3. Collision cases follow the ordered rules above including both same-target and head-swap draw, reverse/out-of-range action death, existing trail/head collision, and no index-order bias.
4. Kaggle PPO optimizer performs nonzero CUDA updates when supported GPU is available and records device/update evidence. Local CPU smoke passes. Actual Kaggle GPU smoke remains external until user invokes it.
5. Manifest/replay/checkpoint adhere to v1 field contracts, canonical digest verification and model/schema compatibility; corruption/unknown schema fails closed.
6. Fixed `lc-eval/1` completes two 64-game matchups (target vs baseline and target vs step-0 checkpoint) over seeds 0..31 with both seat assignments, emits all per-game outcomes, aggregate metrics, Wilson interval, and replay artifacts. Passing means complete report/artifacts, not improvement.
7. Benchmark records arenas/s, steps/s, PPO update latency, available GPU use and hardware/software, compares eligible configs and records selected throughput rationale; no universal speedup threshold or dual-GPU requirement.
8. Kaggle wrapper side-effects only via explicit `submit` + target + source revision + `--confirm`; status/download require explicit kernel/version/run; status failures/timeouts are bounded; staging is validated then atomically promoted; tests never need credentials or contact Kaggle.
9. Personal/local scope remains; no society sim, human play, full tournament or hosted service.

## 7. Mutable external constraints
Current official research: Kaggle CLI docs describe `kernels push -p` with `kernel-metadata.json`, `kernels status`, `kernels output`; prefer an explicit output version; CLI v2/Python/runtime facts can change. Kaggle T4 x2 is an optional configuration, not guaranteed quota. Time/output limits and installed image are mutable; check before user run, checkpoint/resume, validate output sizes. PyTorch DDP adds communication and does not parallelize environment generation; small PPO/CPU-limited workload may favor one GPU. Pin/check installed CLI and torch/CUDA/ONNX versions and benchmark actual available hardware. No remote run or credentials used in planning.

Sources: [Kaggle kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [Kaggle auth](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md), [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), [Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage), [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

## 8. Stop condition
MVP is ready when browser-local checkpoint inference/fresh matches and Python/TypeScript parity pass; artifact contracts and fixed evaluation outputs validate; mock Kaggle lifecycle proves safe, bounded/atomic behavior; local CPU tests pass; and benchmark records a throughput-based choice. Actual Kaggle CUDA/throughput/remote workflow is an external validation gap until user-invoked. Return to planning if ONNX browser compatibility, cross-runtime parity, or safe Kaggle assumptions fail; do not silently downgrade browser inference.
