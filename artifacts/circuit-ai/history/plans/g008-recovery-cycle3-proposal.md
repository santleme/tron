# G008 Kaggle recovery — Cycle 3 proposal

**Status: draft only.** No registry event or Kaggle submission is authorized by this draft. This proposal follows the terminal failure of the Cycle-2 benchmark and does not modify or reinterpret registry events 0–26.

## Outcome

Run a corrected paired benchmark on `pascalcoimbra/circuit-ai`. Only after a successful benchmark artifact is registered, the registration is closed, and the attestation is appended may the selected PPO training workload run. Local performance measurements remain local evidence and are not Kaggle results.

## Reconciled failure and correction

- Cycle-2 event 26 (`850b7fe3f0c2afbdf229cda83a3d2dba51bc607fa3a6efa337d7a6d70de309f7`) records benchmark attempt `g008-cycle2-benchmark-attempt-01` on kernel version 12 as terminal `ERROR`, with 115 GPU-s and 3,763 output bytes. No remote run remains pending.
- The Kaggle log shows the incumbent collector supplied one state per arena and two seat indices per arena to `_observation_batch`; the collector rejected the unequal lengths before any measured run. This is a benchmark harness execution defect, not a timing result.
- Patch `g008-incumbent-seat-observation-expansion-v1` expands each arena state for both seats. The manifest records this as a local compatibility patch over the upstream incumbent snapshot. Candidate code, principal benchmark configuration, paired seeds, warmups, selection rule, and PPO algorithm remain unchanged.
- Cycle-2 package hashes are immutable and cannot carry this correction. Cycle 3 creates new registrations and packages from the reviewed source. Registration IDs, UUIDs, attempt IDs, and changed package identities are new. The unchanged candidate component may retain its content hash; the corrected incumbent component must have a new content hash.

## Append-only Cycle-3 amendment

Append event 27 as `recovery_cycle_amendment_v2`, bound to exact event-26 hash/index, kernel identity, version 12 / `ERROR`, and a fresh Kaggle SDK quota snapshot. The amendment records `registration_bindings_sha256`, the ordered digest of the exact Cycle-3 registration records, and `preflight_context_sha256`, the digest of the exact kernel/version/status/quota snapshot. Each amendment row pins a canonical record hash that its later registration event must match. User authorization binds the proposal digest and both evidence digests. Retire Cycle-2 workload registrations at indices 21 and 22; do not reuse their registration IDs, UUIDs, or attempt IDs.

Keep the Cycle-1 and Cycle-2 validators and historical bytes unchanged. Cycle 3 names exactly one benchmark and one training workload, each with new IDs/UUIDs and independently verified package/component hashes. Registration order is benchmark then training. Both workload records pin event 27 and its hash. No training preflight is valid until the benchmark artifact, immediately following lock, and immediately following attestation are all present.

The new project baseline is 8,664 GPU-s and 34,685 output bytes: Cycle-2's 8,549 GPU-s / 30,922-byte baseline plus event 26's 115 GPU-s / 3,763 bytes exactly once. Cycle-1/2 usage is not added a second time. The v12 terminal evidence is the current kernel cursor; a successful next submission must be reconciled as a later version.

## Resource envelope

Preserve the existing project experiment cap of 21,600 GPU-s, project cap of 28,800 GPU-s, 7,200 GPU-s protected final reserve, and 5,000,000,000-byte aggregate output cap.

- Benchmark cap: 5,600 GPU-s cumulative; one T4; 2,800 GPU-s maximum timeout per attempt; 100,000,000-byte per-attempt output cap.
- Training cap and timeout: 7,200 GPU-s; one T4; 1,000,000,000-byte output cap.
- Four output category reservations remain 1,000,000,000 bytes each for benchmark, training, validation, and final evaluation.
- Worst-case experiment projection: `8,664 + 5,600 + 7,200 = 21,464 / 21,600 GPU-s`.
- Including the protected final reserve: `21,464 + 7,200 = 28,664 / 28,800 GPU-s`.
- At event 26, quota evidence reports 99,372 GPU-s remaining. The registered caps plus protected reserve require 20,000 GPU-s, leaving a projected 79,372 GPU-s; the amendment must use refreshed live quota/status evidence and fail closed if the account has less than required.
- Output projection: `34,685 + 4,000,000,000 = 4,000,034,685 / 5,000,000,000 bytes`.

Every preflight reserves the full registered attempt timeout and full per-run output cap. Terminal attempts consume only reconciled actual GPU/output; retries require unique attempt IDs and must remain within the cumulative caps. The validation/final output and GPU reserves cannot be used by the benchmark or training workloads.

## Required gates

1. Preserve the exact Cycle-1/Cycle-2 registry prefix and confirm version 12 remains terminal before event 27.
2. Pass focused regression, Cycle-1/Cycle-2 replay, Cycle-3 negative-lifecycle, CLI budget, and package identity checks. The Cycle-3 proposal SHA is frozen in the validator before registration.
3. Build benchmark and both training variant packages from the frozen source revision. Check exact hashes against registration records and validate the packages locally; do not run the heavy benchmark or training locally.
4. Freeze `registration_bindings_sha256` and `preflight_context_sha256`; show the exact package/component hashes, kernel identity/version/status, live quota, and budget projection. Obtain explicit user authorization bound to the proposal and both digests, plus fresh confirmation that no other writer will update the kernel until the benchmark is terminal and reconciled. Append the amendment only if the exact live observations still match the authorized context; otherwise do not append and repeat authorization with a refreshed context.
5. Immediately before appending the Cycle-3 amendment and two registrations, query live kernel metadata, terminal status, and quota and require their context digest to match the authorized amendment. Then append the amendment and registrations and run the first benchmark attempt through the registered lifecycle. Its live kernel/version/status/quota context must still match the authorized amendment digest. Reconcile the exact assigned version, terminal status, GPU time, and output bytes. A retry may run only after a terminal attempt; it must match the latest reconciled version/status and pass fresh quota/resource-budget preflight under the same exclusive-writer commitment.
6. On benchmark `COMPLETE` only, validate the benchmark report and execution evidence, append the artifact, immediately lock registrations, and immediately attest. Before training, request fresh writer exclusivity; the live kernel must match the benchmark's reconciled version/status and quota must cover the training attempt plus reserve. Reconcile training terminal status and artifact evidence.

## Stop conditions

Do not append or push if the reviews or tests fail, package hashes differ, exact proposal/registration/context authorization is absent, another writer is active, the live context differs from the authorized amendment digest before registration or before the first benchmark preflight, or any cap/reserve check fails. That amendment digest applies to the initial registration and first benchmark attempt only. After a dispatched attempt, retries and training must match the latest reconciled version and terminal status and must pass fresh quota/resource-budget checks. If a push is ambiguous or accepted but nonterminal, keep its reservation and writer-exclusivity commitment active; reconcile or resume only that same attempt and do not create a retry until it reaches a reconciled terminal or proven no-assignment state. A terminal benchmark `ERROR` is a failed attempt, not a Kaggle result supporting selection; retain the failure evidence and do not train from it.
