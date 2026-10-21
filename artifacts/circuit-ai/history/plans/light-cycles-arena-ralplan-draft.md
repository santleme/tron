# Light Cycles AI Arena — RALPLAN-DR (revision 1)

## Plan summary
**Outcome:** local personal two-agent Light Cycles arena where downloaded Kaggle PPO checkpoints run inference locally in the browser, produce fresh deterministic matches, and expose inspectable metrics/replays. Full PRD: `.omx/plans/light-cycles-arena-prd.md`; test spec: `.omx/plans/light-cycles-arena-test-spec.md`; revision log: `.omx/plans/light-cycles-arena-plan-revision-1.md`.

**Scope:** greenfield local web viewer + browser ONNX inference + TypeScript engine; Python reference engine/training/evaluation/Kaggle packaging; safe explicit Kaggle wrapper. No society sim, hosted service, human play or full tournament. No credentials or real Kaggle jobs in this delivery. Estimated complexity: HIGH due cross-runtime parity, PPO, browser model runtime, and Kaggle lifecycle.

**Evidence:** `.omx/specs/deep-interview-light-cycles-arena.md`, `.omx/interviews/light-cycles-arena-20261019T211321Z.md`, `.omx/context/a-proposta-construir-uma-arena-de-light-cycles-e-20261019T210425Z.md`; greenfield repo. Current official docs reviewed: Kaggle CLI kernel push/status/output and mutable quotas/runtime; PyTorch DDP overhead and tuning considerations. Details are current-documentation facts, not future availability guarantees.

## RALPLAN-DR

### Principles
1. **Browser-local checkpoint behavior is core:** downloaded trained model performs fresh inference/matches in browser with network disabled.
2. **One rules contract, two tested runtimes:** Python is reference for training; TypeScript is browser runtime; shared fixtures prove exact contract parity.
3. **Determinism and artifacts are product contracts:** version seed streams, rules, schemas, serialization and digest semantics before implementation.
4. **Evidence, not promises:** fixed evaluation creates reports/replays; no required learning gain or hardware-independent speed threshold.
5. **Safe and throughput-first Kaggle use:** explicit confirmed submit only; benchmark useful work and never demand both GPUs.

### Top decision drivers
1. User's explicit original requirement for trained-checkpoint inference in browser.
2. Reproducible cross-runtime action/rules and portable artifacts.
3. Local/Kaggle usability, throughput evidence and safe external side effects.

### Bounded architecture options
- **A — Python-canonical engine + local Python checkpoint runner + browser replay viewer.** Pros: one game/policy runtime, least parity risk. Cons: requires Python for fresh local matches and does not meet browser-local inference behavior. Rejected for MVP despite being simplest.
- **B — Python training/reference engine + TypeScript browser engine + ONNX Runtime Web inference.** Pros: downloaded checkpoint runs fresh matches with local browser inference and no local Python inference service/network. Cons: duplicates engine semantics and requires model export/runtime support; mitigated by versioned contracts, deterministic policy and cross-language golden/differential tests. **Selected** because it directly fulfills the original user requirement; validate ONNX runtime/opset support before implementation pins dependency/version.

### Frozen planning contracts and decisions
- Rules: same pre-tick observation; cardinal action order UP/RIGHT/DOWN/LEFT; no reverse; invalid/reverse kills. Compute both destinations, collect out-of-board/tick-start trail-or-head collisions, same-destination and direct head-swap collisions before state mutation; append only survivors. Both die same tick = draw, one dies = other wins; no agent-index priority. Full details in PRD.
- Seeds: unsigned 64-bit master; domain-separated SHA-256 stream derivation using canonical labels/little-endian indexes; PCG-XSH-RR 64/32 with fixed multiplier, wraparound, increment and reference seeding/output procedure in both runtimes. Exact same-contract engine/action/replay reproducibility only; no cross-device PPO bitwise guarantee.
- `lc-run/1`, `lc-replay/1`, checkpoint formats, canonical digest, required fields and compatibility failures are frozen at planning level in PRD/test spec. Replay uses joint action/cause records and engine-version contract; fixture/digest parity required.
- Browser receives ONNX checkpoint; local TypeScript engine runs both policies and generates fresh replays. Native PyTorch checkpoint retained for training resume. No conditional browser inference.
- Evaluation `lc-eval/1`: held-out seeds 0..31; target vs deterministic open-space flood-fill baseline and target vs step-0 checkpoint; each matchup runs both seat assignments (64 games). Report W/D/L, Wilson 95% CI, survival and causes plus per-game data/replays. Pass is complete report/artifacts, not improvement.
- Kaggle side-effect wrapper: only `arena kaggle submit --kernel-id <owner/slug> --source-revision <revision> --confirm` submits. Dry-run default in tests/examples; metadata and running target validated first. Read-only status/download require explicit kernel/version/run. Download to staging, validate hashes/contracts and atomically promote; no overwrite/partial promotion.

## Resource map (planning paths, greenfield target layout)
- `.omx/plans/light-cycles-arena-prd.md` — product workflows, deterministic rules, frozen interfaces and acceptance.
- `.omx/plans/light-cycles-arena-test-spec.md` — parity/evaluation/Kaggle test matrix and stop rules.
- `training/arena/` — Python reference rules, seeded vector runner, observations/actions/reward and parity fixture producer.
- `training/ppo/` — PPO, self-play opponent snapshots, evaluation, metrics, checkpoint/export and benchmark.
- `training/kaggle/` — kernel metadata/entrypoint and resumable output artifacts.
- `apps/viewer/` — local web viewer, TypeScript engine, ONNX browser inference, replay/metrics controls and visual effects.
- `tools/kaggle/` — explicit submit/status/download commands, CLI adapter and atomic artifact promotion.
- `schemas/` or `shared/fixtures/` — v1 JSON schema/contracts and cross-language golden vectors; pick one path consistently during scaffold.
- `tests/` — Python engine/training/artifact, TypeScript/browser, cross-runtime parity, Kaggle mocks.

## Ordered milestones

### 1. Freeze interfaces and compatibility proof
Use PRD/test spec v1: collision resolution, PCG32 seed derivation, observations/action ordering, model/checkpoint export metadata, manifest/replay canonical digest and error behavior, fixed evaluation protocol, Kaggle command safety. Before broad implementation, prove selected ONNX browser runtime supports exported policy opset and deterministic inference; pin/check versions. Materialize a small shared fixture corpus.

### 2. Implement Python reference engine and tests
Build scalar deterministic rules, independent seed streams, observations/actions/reward and terminal outcomes; then headless vectorized runner. Pass boundary/collision/seed golden tests and benchmark rollout before PPO depends on it.

### 3. Implement browser engine and parity
Port engine/observation/action semantics to TypeScript. Differential tests compare initializations, observations, legal actions, transitions, collision causes, outcomes and replay digest across Python/TS for golden corpus. Do not start visual polish until parity gate passes.

### 4. Add PPO self-play, fixed evaluation and exports
Train two agents per arena; PPO optimizer updates use CUDA in Kaggle when available. Produce native PyTorch resumable checkpoints plus ONNX browser inference model and `lc-run/1` manifest. Run fixed `lc-eval/1` held-out protocol, produce outcome reports and replays. Optimizer update >0/device evidence required; evaluation artifact completeness, not win-rate gain, is success. Add vectorization and compact buffers.

### 5. Build browser-local inference and viewer
Load a validated ONNX model/checkpoint/manifest in the browser, run fresh seeded two-agent matches with TS engine, record `lc-replay/1`, and replay it. Add metrics/run inspection, timeline controls, luminous trails, glow/particles/camera after behavior works. Verify action parity against Python policy on fixed observations and run E2E with network disabled.

### 6. Add safe Kaggle submit/watch/download tooling
Isolate documented CLI adapter. Implement explicit `submit` requiring target/source revision/`--confirm`, metadata-target validation and no-running-target guard; status/download read-only with explicit kernel/version/run, bounded polling/timeouts and actionable errors. Stage, validate hashes/contracts and atomically promote only complete outputs. Mock all subprocess/network/auth boundaries. Never run a remote job in delivery.

### 7. Integrate, benchmark and verify
Run CPU smoke/unit/E2E/mocked CLI suites; perform CUDA update test only on actual CUDA/Kaggle availability and report external validation gap honestly otherwise. Benchmark environment rollout and PPO updates separately/end-to-end, compare single vs dual only if both available, record GPU usage/versions and choose by throughput. Stop on all PRD checks with no unresolved parity/artifact/safety failures.

## Acceptance criteria
1. Browser with network disabled loads downloaded ONNX checkpoint and runs a fresh seeded two-agent match without Python inference; saves/displays a replay with playback/scrubbing.
2. Python/TypeScript golden parity matches initialization, observation, action ordering, deterministic policy action, state transitions, collision causes/result and replay digest for contract v1.
3. All collisions and simultaneous movement follow frozen rules, including same-target/head-swap simultaneous draw and no index-order privilege.
4. PPO optimizer makes nonzero CUDA updates under available supported Kaggle GPU and records device/update evidence; CPU smoke works. Actual Kaggle validation is external until user execution.
5. Manifest/replay/checkpoint v1 digest/schema/model compatibility tests pass; corrupted or unsupported artifacts fail closed.
6. Fixed evaluation emits two 64-match comparisons over seeds 0..31 with seat swap, baseline/step-0 identities, W/D/L + Wilson CI, survival/collision metrics, per-game outcomes and replays. No performance improvement is required.
7. Benchmark records arena/step throughput, PPO update timing, available GPU use and environment/software details; records selected configuration/rationale without universal speed threshold or mandatory dual-GPU.
8. Kaggle submit happens only via explicit command + target + source revision + confirm; status/download require explicit target/version/run; timeout/error handling bounded; download validates/stages/atomically promotes; no credentials/network/real run in tests.
9. MVP remains personal/local and excludes society simulation, human-vs-AI, full tournaments and hosted accounts/service.

## Risks and mitigations
- **ONNX runtime/opset/browser constraints:** validate export/load/action parity during contract gate; pin/check versions. If infeasible, stop and return evidence; do not silently downgrade browser inference.
- **Python/TS semantic drift:** one versioned contract, shared fixtures, exact differential tests before viewer/policy integration.
- **Seed/replay nondeterminism:** implement same SHA-256 stream derivation and PCG32; canonical digest fixture; scope guarantee to version contract, not GPU PPO weights.
- **PPO does not improve:** baseline + fixed held-out report/replays; product acceptance is honest report production, not learned superiority.
- **GPU starved or two-GPU slower:** vectorize first; measure rollout/update separately; T4 x2 optional and DDP has overhead/no environment sharding.
- **Kaggle drift/quotas/partial output:** pin/check CLI/runtime, bounded polling, resumable checkpoints, staging validation/atomic promotion, tests with fakes.
- **Scope creep:** defer social Grid, human and tournament modes.

## Verification and stop conditions
Use `.omx/plans/light-cycles-arena-test-spec.md` focused matrix. Sequential gates: (1) schema/rules/seed/evaluation/runtime fixture review, (2) Python rules, (3) TS parity, (4) PPO and ONNX/checkpoint exports, (5) browser fresh match offline, (6) fixed eval artifact, (7) Kaggle mocked side effects/atomic downloads, (8) throughput report. Stop when PRD acceptance has fresh evidence; external Kaggle GPU/remote tests remain explicitly outstanding until user invokes them. Loop back if model runtime compatibility, parity, schema integrity or safety fails.

## Handoff
Architect re-review this revision, then Critic review. After approval, Ultragoal may sequence source lanes: Python engine/training; browser TS engine/ONNX UI; Kaggle adapter/docs. Shared contracts/fixtures are a prerequisite and require one integrator; test/review lanes remain independent. No implementation work begins before reviews.
