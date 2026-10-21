# Test Specification — Light Cycles AI Arena MVP

**Status:** Planning artifact, revision 1 for Architect re-review  
**Source:** `.omx/specs/deep-interview-light-cycles-arena.md`, `.omx/plans/light-cycles-arena-prd.md`, `.omx/plans/light-cycles-arena-ralplan-draft.md`.  
**No source code/tests are created by this document.** Framework/commands are selected during implementation.

> **Legacy scope:** This document specifies the original MVP/G007 contracts only. It is not authoritative for G008 training, seed derivation, benchmark pairing, checkpoint resume or statistical gates. For G008 use `.omx/plans/g008-competitive-intelligence-test-spec.md` together with the Cycle-4 canonical execution plan `.omx/plans/g008-competitive-intelligence-plan.md`. Preserve this file's historical semantics for existing G007 artifacts; do not retrofit G008 rules into G007 evidence.

## Test goals
- Prove simultaneous deterministic rules and Python/TypeScript contract parity.
- Prove browser-local inference from a downloaded trained checkpoint creates fresh matches without Python inference/network.
- Prove checkpoint/manifest/replay artifacts interoperate across Kaggle and local viewer.
- Prove PPO updates, fixed evaluation report production and measured useful throughput.
- Prove Kaggle submit is explicit/confirmed and status/download are bounded, read-only, mocked in tests, validated and atomic.

## Frozen behavioral and reproducibility test contract
- Both agents observe pre-tick state and select actions before either move is committed. Action order is `UP, RIGHT, DOWN, LEFT`; reverse/invalid action kills the actor.
- Compute both destinations. Out-of-bounds or destination equal to any tick-start occupied trail/head cell kills that actor. If both destinations match or actions create direct head swap, both die. Collect collisions before mutating state; append destinations only for survivors. Simultaneous death is draw; exactly one death means other wins. No index priority.
- Seed contract: unsigned 64-bit master seed; SHA-256 domain-separated episode/arena and stream labels with little-endian indexes derives independent streams; PCG-XSH-RR 64/32 is implemented identically: multiplier `6364136223846793005`, 64-bit wraparound, odd increment `(initseq << 1) | 1`, reference two-step seeding/output rotation; first 16 SHA-256 digest bytes unpacked as little-endian `initstate` and `initseq`. Golden tests compare generated initialization, observations and action tie-breaks. Guarantee exact engine/action/replay equality under same schema/contract; not bitwise cross-GPU PPO training reproducibility.
- Evaluation inference uses eval mode, deterministic argmax, no sampling/dropout, lowest action index for ties.
- Replay digest uses canonical UTF-8 JSON: sorted object keys, no insignificant whitespace, NFC strings, finite numbers only, array order preserved, digest field excluded from digest input; SHA-256. Contract fixtures pin canonical bytes/digest. Any change requires schema/version bump.

## Frozen artifact fixtures
- `lc-run/1` manifest requires run/time IDs; engine, observation/action/reward/replay schema versions; master seed/training config/steps; checkpoint list with IDs, step, relative file, SHA-256, format and obs/action versions; `lc-eval/1` definition/seeds/match count/opponent identities/metrics/replay paths; software/device versions; artifact path/digest list and canonical manifest digest.
- Checkpoint set includes native PyTorch weights/config for training resume and ONNX model for browser inference; shared checkpoint ID/step/model/obs/action versions and SHA-256. Browser test rejects incompatible opset/runtime/schema; only validates versions actually selected by implementation.
- `lc-replay/1` includes run/match/engine/schema IDs, master/arena/derived seed streams, agent/checkpoint IDs, initial grid/positions/headings, per-tick joint actions, collision causes, terminal result/tick count and canonical digest. Python and TS replayers reconstruct exact state and digest. No compression library until size benchmark justifies it.
- Fixture tests verify required fields, canonical digest, unknown/missing/corrupt values, traversal/unsafe paths, wrong checkpoint hash, wrong run/version, and clear fail-closed messages.

## Focused test matrix

| Area | Level/fixture | Required cases | Passing evidence |
|---|---|---|---|
| Python engine rules | Unit/golden | wall/bounds; own/opponent trail/head; same target; head swap; simultaneous death/draw; exactly one death; invalid/reverse; terminal no-action; action-independent index order | Expected states/outcomes match contract for each fixture |
| TypeScript engine parity | Differential contract test | Same corpus of starting states/actions and seeded initialization in Python+TS; compare observation bytes/values, action order, every resulting cell/head/trail, cause/result and digest | Exact parity for v1 fixture set; mismatch blocks merge |
| Seed streams | Cross-language golden | SHA-256 domain labels/index encoding, PCG32 sequence, arena resets, independent agents, repeated same seed, changed arena index | Both runtimes produce exact stream outputs; no cross-arena/agent leakage |
| Observation/action/policy | Unit/cross-runtime | Shape, orientation, action legality/order, deterministic argmax/tie; representative observations through Python policy and exported ONNX/browser runtime | Same selected action for all fixed vectors; unsupported model/schema fails closed |
| Vectorized runner | Unit/integration | Independent arena seeds/state; batch step equals scalar step; reset derivation deterministic; no rendering dependency | Per-arena transitions match scalar contract and seed records |
| Manifest/checkpoint/replay | Unit/round trip | `lc-run/1`, `lc-replay/1`; schema validation; canonical digest/hash; model/checkpoint linkage; replay reconstruct; missing/corrupt/unsupported versions/files; path traversal; stage promotion atomic | Valid artifact round-trip; corrupt/incompatible fails before UI use; no partial final directory |
| PPO training | Tiny integration | Rollout tensor shapes, finite returns/advantages, optimizer parameter change, count > 0, native checkpoint save/load, ONNX export+load/action equivalence; seeded CPU smoke; CUDA smoke if available | CPU smoke passes; CUDA evidence records device and nonzero optimizer updates on Kaggle/available CUDA environment; local CPU/mock does not masquerade as CUDA validation |
| Fixed evaluation `lc-eval/1` | Integration | Seeds exactly 0..31 held out; target vs baseline open-space and target vs step-0 checkpoint; both seats for every seed = 64 matches per matchup; metrics and all per-game records; replay examples/full selected replay set; Wilson 95% interval | Complete report, seed/opponent/checkpoint identity and replay artifacts. Pass does not require target outperforming either opponent |
| Browser local inference | Browser E2E, network disabled | Load local ONNX checkpoint/manifest; start fresh deterministic game; two policies execute in browser; replay recording; reload/replay scrub; verify no Python service/network request; unsupported model error | Fresh match completes and replay round-trips; inference action vectors match fixtures; zero external network calls |
| Viewer/replay UI | Browser E2E | Play/pause/restart/scrub; match result and agent/checkpoint IDs; metrics; malformed replay; effects do not change underlying result | Known timeline states and terminal outcome match fixture; readable failure recovery |
| Kaggle submit wrapper | Unit tests with fake process | `submit` without `--confirm`; missing/ambiguous target; mismatched kernel metadata; missing/ambiguous source revision; already-running target; explicit confirmed invocation builds `kernels push -p` from selected directory; no startup/build/viewer submit | No side-effect call unless explicit command + kernel/source revision + confirmation pass; command arguments redacted; tests use no network/auth |
| Kaggle status/download | Mocked integration | Explicit kernel and version; running polling and bounded timeout; success; terminal error/cancel; missing CLI/auth; output failure/partial/corrupt/wrong run; valid staging; existing final target | Status/download only read-only; failures actionable and bounded; valid directory promoted atomically; invalid staging never overwrites existing artifact |
| Throughput benchmark | Reproducible benchmark, non-CI threshold | Warmup; scalar/vector rollout; PPO update and end-to-end timings; single/dual only if available; CPU/GPU utilization and versions | Report arenas/s, steps/s, update latency, workload/device/software and selected config rationale; no universal numeric pass target |
| Scope/security | Static/repo checklist | No credentials in source/log/artifacts; tests don't submit; no hosted accounts, society sim, human play/full tournament | Scope review and mocks pass |

## Evaluation protocol/pass meaning
- Fixed held-out seeds `0..31`, excluded from training. For each matchup use all seeds and both seat assignments (64 games): target checkpoint vs deterministic `open-space` baseline and target vs step-0 self-play checkpoint. Persist individual outcomes.
- Baseline computes four-neighbor flood-fill reachable free cells for each legal candidate move, chooses the maximum, tie-breaks by action order. Baseline ID and implementation version enter manifest.
- Metrics: W/D/L, win-rate, Wilson 95% interval, mean/median survival ticks, collision causes, per-seed/seat results. Save replay files with selected target and run identity.
- Acceptance is successful, complete and reproducible report/artifact generation, not significant win-rate or guaranteed learning. Any improvement wording must be evidence-backed.

## Kaggle side-effect safety and failure handling
- Exact wrapper contract: `arena kaggle submit --kernel-id <owner/slug> --source-revision <revision> --confirm` is the only submission command; `--dry-run` is the default in tests/examples. It validates kernel metadata ID and source revision, refuses ambiguous/running target without side effect, then invokes documented `kaggle kernels push -p <dir>` and records resulting version/run identity. No submit from startup/build/viewer/training tests.
- `arena kaggle status --kernel-id <owner/slug> --version <n>` and `arena kaggle download --kernel-id <owner/slug> --version <n> --run-id <id>` are explicit read-only operations. Status polling uses max attempts/deadline and returns distinguishable success/running/terminal-error/timeout states; download begins only on success.
- Download stages within artifact root, verifies manifest/schema/digests/expected IDs, then atomically renames into a new final directory. Existing destination or invalid/partial content fails closed; never partially overwrite. Cleanup policy for abandoned staging is documented/tested.
- Credentials resolved only on explicit invocation via currently official Kaggle CLI configuration; tests do not require auth. No secrets are printed or stored.
- Kaggle CLI command/API versions, notebook image, CUDA availability, T4 x2 quota, runtime and output limits are mutable. Pin/check installed CLI and torch/CUDA/ONNX versions; checkpoint/resume and output-size validation. Actual Kaggle GPU smoke and remote lifecycle are external evidence until user invokes them.

## Throughput protocol
Measure engine rollout and PPO optimizer update separately and together. Capture warmup, duration, number of arenas, rollout/update ratio, batch size, CPU cores, GPU model/utilization, software versions. Compare one vs two GPU only if actually available with same workload; DDP adds communication and does not parallelize environment generation. Select and document useful throughput config; do not treat occupancy as a success criterion.

## Verification order / stop rules
1. Review/freeze v1 schemas, collision ordering, seed scheme, browser inference and eval protocol before implementation split.
2. Pass Python fixtures, then TS differential engine and policy/model action parity before viewer integration.
3. Pass PPO CPU smoke, ONNX export/load, model/checkpoint hashes, replay and manifest round-trip.
4. Pass browser E2E with network disabled and fresh checkpoint inference/replay.
5. Pass fixed evaluation artifact report (not improvement threshold), then mocked Kaggle command/failure/atomic download tests and secret audit.
6. Gather throughput records and select config; mark actual Kaggle GPU/remote test as outstanding if not user-run.
7. Stop when PRD criteria have evidence; return to planning if browser runtime/model compatibility or exact cross-engine parity cannot be achieved, schemas mismatch, or Kaggle safety cannot be demonstrated. Never silently remove browser inference.

## Official references and caveats
[Kaggle kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [Kaggle auth](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md), [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), [Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage), [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html). Current CLI docs describe `kernels push -p` with metadata, status and output; explicit output version is preferred. T4 x2 is optional, not guaranteed. Current quotas/time/output/runtime versions can change. DDP may be slower for small policies/CPU-limited rollout; benchmark actual configurations.
