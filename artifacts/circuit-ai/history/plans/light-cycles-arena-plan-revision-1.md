# Plan Revision 1 — Architect ITERATE Findings Disposition

**Purpose:** record changes to `.omx/plans/light-cycles-arena-ralplan-draft.md`, `.omx/plans/light-cycles-arena-prd.md`, and `.omx/plans/light-cycles-arena-test-spec.md` before subsequent Architect review. Planning-only; no source code, Kaggle credentials, or remote jobs.

## Finding disposition

1. **Runtime boundary / original browser-local inference intent — RESOLVED.** Browser-local checkpoint inference and fresh matches are required MVP behavior, not conditional. Revised architecture is Python training/reference rules + TypeScript browser engine + ONNX Runtime Web inference; browser works with network disabled and does not need a Python inference service. Option A (Python local runner/replay viewer) is retained as rejected alternative. The additional cross-engine burden is accepted and covered by exact golden/differential tests. PRD sections 1, 2, 3, 5, 6; RALPLAN options/milestones; test matrix.

2. **Interoperable replay/manifest/checkpoint contract — RESOLVED.** Frozen planning-level `lc-run/1`, `lc-replay/1`, and paired native PyTorch + ONNX checkpoint requirements. Required identifiers/schema fields, canonical JSON constraints, SHA-256 digest scope, per-tick joint actions/collision causes, replay reconstruction and fail-closed compatibility behavior are specified in PRD and repeated in test fixtures. Compression is deferred until measured need. PRD artifact contracts; test spec artifact fixtures.

3. **Seed/determinism guarantee — RESOLVED.** Specifies unsigned 64-bit master seed, SHA-256 domain-separated episode/arena stream derivation with little-endian indices, PCG-XSH-RR 64/32 with fixed multiplier, wraparound, increment and reference seeding/output procedure shared between Python/TypeScript, deterministic evaluation inference/action tie-break, recorded seed streams, and exact reproducibility boundary limited to matching schema/runtime contract; no cross-device PPO bitwise guarantee. PRD arena contract; test spec reproducibility vectors.

4. **Collision and simultaneous-resolution rules — RESOLVED.** Both observe same pre-tick state and choose before mutation; bounds/tick-start trail+head/action-invalid deaths, same destination and head swap are accumulated simultaneously; only survivors move; both deaths draw, one death loses; no index priority. PRD and test spec contain explicit rules and fixtures.

5. **Kaggle side-effect boundary and download robustness — RESOLVED.** One exact side-effecting wrapper command: `arena kaggle submit --kernel-id <owner/slug> --source-revision <revision> --confirm`; only submit can push. Status/download are read-only and require explicit kernel/version/run. Refusal cases, bounded polling/timeout, output staging, manifest/hash validation, atomic promotion and no overwrite are specified. Credentials remain invocation-only and never in tests/logs. Current Kaggle CLI push/status/output docs are referenced; mutable auth/runtime/quota details must be checked at implementation/user-run time.

6. **Fixed evaluation and objective acceptance — RESOLVED.** `lc-eval/1` holds out seeds 0..31; target vs deterministic flood-fill open-space baseline and target vs step-0 checkpoint; 32 seeds x 2 seat assignments = 64 games per matchup; emits per-game records, W/D/L, Wilson 95% CI, survival/collision metrics and replays. Pass condition is complete reproducible report/artifact generation, not a guaranteed/significant learning gain. Benchmark similarly requires recording data and a selected throughput rationale, no generic speed threshold.

## Preserved external caveats
Kaggle CLI docs currently describe `kernels push -p` with metadata, status and output; pin/check installed CLI and select explicit output version. T4 x2 is optional, mutable quota/runtime/output/time limits require live checking. PyTorch DDP adds communication and does not shard environment generation; benchmark one vs two only if actually available. Actual Kaggle CUDA/remote round-trip remains unverified and out of scope until user invokes it.

## Files updated
- `.omx/plans/light-cycles-arena-ralplan-draft.md`
- `.omx/plans/light-cycles-arena-prd.md`
- `.omx/plans/light-cycles-arena-test-spec.md`

## Review state
This revision is ready for a fresh Architect review. Critic review remains pending. No review approval is claimed here.
