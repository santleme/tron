# G008 Kaggle recovery — Cycle 2 proposal

**Status: draft only.** No registry event, Kaggle submission, or training is authorized by this draft. It proposes the separately reviewed cycle required after the registered v11 benchmark failed on an import-context defect.

## Outcome

Run a corrected paired benchmark and, only after its successful artifact is locked and attested, run the registered PPO training workload. Keep the original registry prefix and events 13–19 byte-for-byte unchanged. Do not describe local timing as Kaggle evidence.

## Evidence and change

- The first recovery registry suffix is terminal at event 19, hash `7280fb7471484e856b48d1a931cb18d97d5bd2f94949790c41defef9154c4213`: benchmark attempt 01 on kernel `pascalcoimbra/circuit-ai` version 11 ended `ERROR`, consumed 125 GPU-s, produced no artifact, and has no unresolved remote run.
- The log failure was `ModuleNotFoundError: training.ppo.seeds` while loading the frozen incumbent engine. Commit `ae8cd50` loads the same frozen engine bytes under `training.arena`, where its relative `.seeds` import resolves. The focused regression tests passed (2); the full test file was interrupted after 8 tests and is not claimed as passing.
- Existing registrations at events 14 and 15 bind immutable package hashes and a 3,600 GPU-second cumulative benchmark cap. They cannot authorize the corrected package or a full-timeout retry after 125 seconds were consumed.

## Proposed append-only authorization

This is an explicit, separately reviewed and user-authorized exception to the original plan's one-amendment/two-registration-only event whitelist. It authorizes exactly one versioned cycle-2 amendment and exactly two cycle-2 registrations; no third cycle or further registration is implied. It does not alter or reinterpret any byte or hash in events 0–19. After Architect then Critic approve this exact proposal and the user authorizes the revised cycle, add one `recovery_cycle_amendment` event at index 20, bound to event-19 hash above, the original approved seed-plan SHA `c6b16733a045e94e7310183e8be8a62cf4327b5d2bdc3eed40bb9d552ce17593`, the final proposal hash, current Kaggle version/status/quota evidence, and the exact new benchmark/train registration IDs. Extend the validator and CLI with this explicit cycle boundary; preserve validation of legacy events and the already-terminal first cycle. Do not append a second legacy `recovery_amendment` or mutate/reinterpret events 13–19.

The validator must assign each attempt to exactly one cycle and enforce separate per-cycle workload totals (cycle 1: actual benchmark 125/3,600 GPU-s and train 0/7,200; cycle 2: benchmark 0/5,600 and train 0/7,200 at amendment time). Event 20 explicitly retires and supersedes both cycle-1 registrations at indices 14–15 for future execution; no new preflight, dispatch, acceptance, or attempt may reference either. The already-terminal event-19 benchmark attempt remains historical evidence. Prevent double-counting at the cycle boundary: preserve the existing amendment-13 `campaign_gpu_seconds_before=8,424` and `campaign_output_bytes_before=26,403` as the one-time project base; derive event-20's declared current totals as base plus all terminal recorded recovery attempts through event 19, exactly 8,549 GPU-s and 30,922 bytes. The validator compares those declared totals to its computed values, but never adds them as a second base. All future project projections use the original base plus all historical and new actual usage plus outstanding reservations exactly once. Require index 20 to bind the exact prior head; exactly two cycle-2 registrations must follow before any new workload; benchmark attempts/artifact/lock/attestation must complete before any cycle-2 training attempt. Reject interleaved cycles, reused IDs/UUIDs, an extra registration, a second cycle amendment, any attempt lacking its cycle ID, and any edit or reclassification of events 0–19. The cycle-2 lock binds the successful cycle-2 artifact; post-lock attempts bind that lock. Only cycle-2's newly registered training workload remains executable, preventing double reservation/use of the unused cycle-1 training cap.

Create exactly two new registrations with newly built, independently verified package hashes. The benchmark binds the committed import fix and the same frozen incumbent/candidate implementations, config, schedules, roots, and parity/selection contract. The training registration binds those same two implementations and runs only after the new benchmark artifact, lock, and attestation. No validation/final workload is added.

## Resource envelope (conservative)

Use event 19's actual campaign usage, 8,549 GPU-s, as the new baseline; preserve the existing experiment cap of 21,600 GPU-s, overall cap of 28,800 GPU-s, and 7,200 GPU-s protected final reserve. Proposed new-cycle caps are benchmark 5,600 GPU-s with 2,800-second one-T4 timeouts, and training 7,200 GPU-s with a 7,200-second one-T4 timeout. The worst-case projection is:

- Experiment: `8,549 + 5,600 + 7,200 = 21,349 / 21,600 GPU-s`.
- Including final reserve: `21,349 + 7,200 = 28,549 / 28,800 GPU-s`.
- Account quota after the new-cycle caps and protected reserve, using event-19's 99,468 GPU-s remaining: `99,468 - 5,600 - 7,200 - 7,200 = 79,468 GPU-s`.

For outputs, event 19 leaves 30,922 actual campaign bytes (26,403 before the original amendment plus the terminal attempt's 4,519-byte log). Keep the existing 5,000,000,000-byte aggregate cap and 1,000,000,000-byte per-run hard maximum. Reserve cycle-2 categories of 1,000,000,000 bytes each for benchmark, training, validation, and final evaluation; these reserves total 4,000,000,000 bytes. The full reservation projection is `30,922 + 4,000,000,000 = 4,000,030,922`, leaving 999,969,078 bytes below the aggregate cap. Set the cycle-2 benchmark per-attempt output cap to 100,000,000 bytes and training per-attempt cap to 1,000,000,000 bytes. Before every attempt, charge prior actual output for that category plus the new full per-attempt reservation against its 1,000,000,000-byte category and recompute aggregate actual plus outstanding reservations. The 1 GB validation/final reserves remain reservation-only and unavailable to benchmark/training retries. No output reservation may be counted twice when an attempt becomes terminal.

Before any push, refresh authenticated quota, exact kernel version/status, policy, output accounting, and terminal status for every prior attempt. Require fresh user confirmation that no other writer will update the kernel through benchmark completion and reconciliation. A retry uses the identical registered package and full registered timeout; it is permitted only when the cumulative benchmark/train, experiment, quota, output, and reserve checks still pass. The projected figures are not a substitute for live preflight.

## Lifecycle and gates

1. Keep the original seed plan, migration tests, and historical registry bytes immutable. Verify the exact plan/seed hashes and existing 46 migration/consumer rejection tests cited in the accepted Cycle-5 review.
2. Sequentially review this cycle proposal: Architect approval, then Critic approval, each binding its SHA-256. Freeze the final plan bytes before implementation/registration.
3. Extend registry validation and deterministic tests for cycle-2 amendment ordering, parent-head binding, exact two registrations, cross-cycle GPU/output accounting, package hashes, terminal retry requirements, the output-category reservation rules, and rejection of interleaving, a third registration, a second amendment, or mutation of events 0–19.
4. Build and verify exact Kaggle packages; run local unit/packaging checks only. Do not run the heavy benchmark or training locally.
5. Register the two workloads and run the paired benchmark on Kaggle under the fresh exclusivity and budget preflight. Reconcile terminal status before any next event.
6. On success only: append the exact benchmark artifact, immediately close registration, append attestation, then run registered PPO training on Kaggle and reconcile its terminal evidence.

## Stop conditions

Stop before external execution if sequential reviews, fresh user authorization, the fresh no-outside-writer confirmation, exact version/quota/policy evidence, package verification, or budget/reserve checks fail. Never reuse the old failed benchmark artifact as success, exceed the project caps, rewrite prior registry bytes, or use local microbenchmarks as Kaggle results.
