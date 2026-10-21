# G008 Kaggle recovery — Cycle 3 benchmark retry proposal

**Status: reviewed and content-frozen.** Package-specific hashes are bound separately in the event-35 record after package construction and validation. This proposal does not authorize a registry write, Kaggle submission, or training by itself. It preserves Cycle 3 and its history. It proposes one new append-only retry amendment and benchmark registration after the Cycle-3 benchmark attempt failed.

## Outcome

Run exactly one corrected paired benchmark continuation on `pascalcoimbra/circuit-ai`, with a new registration, run UUID, package identity, and attempt ID. This authorization covers one workload attempt only. Training may proceed only if this corrected benchmark reaches Kaggle `COMPLETE`, its report and execution evidence validate, and the artifact is immediately followed by a registry lock and attestation. The existing Cycle-3 training registration is not permission to train after an `ERROR` or incomplete benchmark.

Local checks may validate correctness, package structure, and deterministic behavior. Local timings are local measurements only; they do not count as Kaggle/T4 benchmark results, cannot support variant selection, and cannot replace the registered Kaggle benchmark.

## Cycle-3 attempt being retried

- Event 33 is `workload_attempt` for `g008-cycle3-benchmark-attempt-01`, registration event 28, assigned kernel version 13. Its event hash is `4f7c8d909349b3a95c91758a2fe6c7490f241ed5db9df9925ce4f600163db966` and its predecessor hash is `b0019b8174f05d802eb66ac6cb8900606727258f7cfc3440729d178e72f75b8c`.
- Event 33 is terminal `ERROR`, with 104 actual GPU-s, 4,497 output bytes, and no artifact. It is a failed attempt, not a benchmark result. Record those actual costs once in all later budget calculations; do not add them again from event 31 or 32.
- The next amendment must be event 34, type `recovery_cycle_retry_amendment_v1`, and bind `prior_attempt_event_index: 33` plus the exact event-33 hash above. The amendment must also identify kernel `pascalcoimbra/circuit-ai`, assigned/current version 13, and the reconciled terminal status `ERROR`.
- Event 35 must be a new benchmark registration, bound to event 34 and its hash. Use a new registration ID, run UUID, package hash, and attempt ID. Do not edit events 0–33, reuse the Cycle-3 benchmark attempt ID, or mutate the existing event-28 registration. Keep the event-29 training registration historical; its use remains gated on successful benchmark artifact, lock, and attestation.
- The retry package must include the reviewed correction for the event-33 failure. Freeze the source revision and exact package/component hashes only after implementation and focused regression/lifecycle tests pass. If diagnosis requires a behavioral or package change, the new event-35 registration must bind those exact changed bytes.

## Resource envelope after event 33

Preserve the project caps and reserves: 21,600 GPU-s for project experiments, 28,800 GPU-s total, 7,200 GPU-s protected final reserve, and 5,000,000,000 aggregate output bytes. Count event 33's actual resource use once:

- Reconciled project baseline: `8,664 + 104 = 8,768 GPU-s` and `34,685 + 4,497 = 39,182 output bytes`.
- Cycle-3 benchmark registration retains its frozen cumulative category cap, `workload_gpu_seconds_cap=5600`. After event 33, the amendment/accounting records the remaining benchmark budget as `5,600 - 104 = 5,496 GPU-s`; do not rewrite the registration cap to 5,496.
- The single continuation attempt may reserve at most 2,800 GPU-s and 100,000,000 output bytes. Keep its timeout at 2,800 GPU-s. Registry replay reserves the attempt and consumes reconciled actuals cumulatively against the frozen 5,600 GPU-s cap, so the 104 GPU-s from event 33 count once and leave 5,496 GPU-s available before this attempt. Any unused Cycle-3 benchmark category budget remains unspent under this authorization; it does not permit another attempt.
- Training remains capped at 7,200 GPU-s, with a 7,200 GPU-s attempt timeout and 1,000,000,000-byte output cap. Keep the separate 7,200 GPU-s protected final reserve unavailable to benchmark and training.
- Worst-case projection for this authorization, including the single attempt's full reservation, is `8,768 + 2,800 + 7,200 = 18,768 / 21,600 GPU-s` for experiments. Including the protected final reserve: `18,768 + 7,200 = 25,968 / 28,800 GPU-s`. The remaining 2,696 GPU-s within the Cycle-3 benchmark category cap after a fully reserved attempt stays unspent.
- Existing four output-category reservations remain 1,000,000,000 bytes each for benchmark, training, validation, and final evaluation. The baseline output is 39,182 bytes; reserve projection remains `39,182 + 4,000,000,000 = 4,000,039,182 / 5,000,000,000 bytes`.
- These figures are registry accounting, not a fresh Kaggle quota observation. Before event 34, query live kernel status/version and account quota; reject any queued/running kernel and require enough remaining quota for the registered retry attempt, the full 7,200 GPU-s training allocation, the protected 7,200 GPU-s reserve, and reconciliation slack. Enforce the per-attempt timeout/output limits and project caps from the registration. Abort if live quota or any cap is insufficient.

## Authorization and live context binding

The Cycle-3 authorization does not authorize this retry, event 34, event 35, or a new push. Before append, freeze three exact digests and obtain fresh explicit authorization that names all three:

1. `proposal_sha256`: SHA-256 of this content-frozen proposal.
2. `registration_bindings_sha256`: digest of the canonical event-35 benchmark registration record, including its new IDs, package/source/component hashes, variant, run config, caps, and event-34 binding.
3. `preflight_context_sha256`: digest of fresh live Kaggle evidence for kernel identity, current version, terminal status, and quota. Resource limits are frozen separately in the registration record and project budget replay. It must show version 13 / `ERROR` before registration, unless a reconciled external change requires a newly reviewed proposal and authorization.

The event-34 amendment must store those exact digests, the event-33 index/hash, the event-35 canonical registration-record hash, the refreshed quota evidence, and the user's exact authorization statement and confirmation time. Immediately before atomic append of events 34 and 35, re-query kernel metadata/status/quota and require the context digest to match. If any bound input changes, do not append or submit; refresh the proposal/context and authorization. The authorization must also confirm that no other writer will update `pascalcoimbra/circuit-ai` until the retry benchmark is terminal and reconciled.

## Execution gates

1. Preserve the registry byte-for-byte through event 33. Diagnose the event-33 Kaggle failure and implement the bounded correction. Run targeted correctness, replay, package-identity, budget, and lifecycle checks; do not run the heavy benchmark locally.
2. Build the retry package from a frozen source revision. Independently verify its package and component hashes, new registration ID/UUID, and unique attempt IDs. Keep the existing candidate/incumbent comparison definition, seed plan, and selection rule unless failure diagnosis proves one must change; any change must be explicit in the proposal and registration.
3. Calculate the bindings and fresh context digests, freeze the proposal hash after code/tests, and obtain authorization bound to all three digests plus fresh single-writer exclusivity for the benchmark through reconciliation.
4. Recheck exact live metadata, terminal state, quota, and resource budget. Atomically append event 34 and event 35 only when every digest and binding matches. Submit exactly one attempt through the registered project lifecycle. Reconcile its exact assigned kernel version, terminal status, actual GPU time, output bytes, logs, and artifacts.
5. This authorization is exhausted when the single attempt reaches any terminal outcome. Any subsequent benchmark attempt, including after terminal `ERROR` or another terminal outcome, requires a new reviewed proposal, fresh registration bindings and context, and new explicit authorization. Preserve unused Cycle-3 benchmark budget. An ambiguous or nonterminal push retains its reservation and writer lock; reconcile or resume that same attempt and never dispatch a duplicate.
6. On benchmark `COMPLETE` only, validate the paired report and provenance, append the benchmark artifact, then append its registration lock and attestation immediately. Verify all three events and hashes before considering training. Obtain fresh exclusive-writer confirmation for the training phase, recheck quota/budget, and use the existing training registration only if validators confirm it remains valid against the new successful benchmark chain; otherwise prepare a newly authorized registration. Reconcile training terminal status and artifact evidence.

## Stop conditions

Do not append or submit if implementation/tests fail, diagnosis is unresolved, the package differs from its frozen hashes, any of the three authorization digests is missing or mismatched, the live context changed, another writer is active, the kernel has a queued/running run, the registry prefix differs, or quota/caps/reserves fail. Dispatch at most one attempt under this authorization. Never train from a benchmark `ERROR`, local timing, unvalidated output, artifact without lock and attestation, or a benchmark chain that fails registry replay. After any terminal outcome, stop this retry authorization; any additional attempt requires a new reviewed proposal, fresh evidence, and new explicit authorization. Keep unused Cycle-3 benchmark budget unspent.
