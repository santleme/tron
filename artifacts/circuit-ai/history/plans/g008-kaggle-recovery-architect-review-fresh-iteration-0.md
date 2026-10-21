# G008 Kaggle recovery — Architect review, fresh iteration 0

**Verdict: ITERATE**

The plan addresses the prior Planner issues and specifies exact `/3` approval, amendment-embedded reservations, two predeclared implementation variants, and immutable legacy indices 0–12. Three blockers remain:

1. **The current `/3` seed plan contradicts the recovery benchmark-before-lock order.** `training/configs/g008-split-seed-plan.json` (SHA `0173875c2f13fc3ffe5dbae76c2b6d6e46bc8398da7f8a748988a51b345d4b24`) permits the benchmark before lock and lock before training in one section, but later requires `registration_closed` before any benchmark workload. Recovery requires benchmark → artifact → lock. Do not approve `/3` unchanged. Update the Cycle-5 seed-plan lifecycle wording during the upstream planning gate, compute a new exact hash, review those bytes sequentially (Architect then Critic), and test benchmark-before-lock/training-after-lock. This supersedes the current hard-coded candidate hash; recovery plan references must bind the corrected hash.
2. **The current Kaggle adapter cannot produce a complete historical version set.** `tools/kaggle/cli.py::_kernel_list_contains` discards version fields; `submit()` returns raw CLI push output; `poll_status()` takes an explicit version. The installed Kaggle CLI does expose `current_version_number` in kernel list JSON and its push success response includes the assigned `version_number`; no full historical-version listing exists. Require the supported push response/API to identify the exact assigned positive version, cross-check with the current-version field, and poll only that version. If successful push output omits or conflicts on the assigned version, record unresolved status and fail closed; do not infer a full set or retry until reconciled. Preserve sanitized rejection evidence rather than arbitrary stderr.
3. **Code-fix path conflicts with exactly two frozen variants.** The plan predeclares incumbent and candidate hashes, but later permits a code fix only if already in that same set, which makes a changed fix unreachable. Simplest resolution: after registration, no code/package fixes are allowed. Candidate parity failure or defect makes it ineligible; train the exact unchanged incumbent if valid, otherwise stop. Any changed implementation belongs to a separately reviewed cycle with new registration and benchmark.

## Registry and reservation evidence

The registry is an immutable 13-event chain at indices 0–12, head `0208abba4954ffd08b01d5069c674e8bc3f5d1817c95ddb0b0bb4eedd41f3fd7`; legacy artifact is index 10 and historical registrations are indices 11–12; no lock exists. Preserve this prefix and distinguish the legacy artifact from the new recovery artifact, which must immediately precede its lock. `reservation_metadata` in `recovery_amendment` closes the extra-registration ambiguity. Current validator does not support the new event types yet.

## Steelman antithesis

Keep the no-count ceiling: cumulative compute, quota, output, wall time, and immutable workload identity bound the actual cost; a fixed retry counter would obstruct recovery without guaranteeing affordability.

## Tradeoff and synthesis

Retries require reliable version identity, but the local adapter cannot enumerate all historical versions. Preserve count-free retries while requiring the supported push response to yield the exact assigned version and the live current-version field/status to corroborate it; ambiguous accepted pushes stop pending reconciliation, never trigger blind resubmission.

## Verification recommendations

- Correct and sequentially re-review the exact Cycle-5 seed-plan hash; tests must accept pre-lock benchmark, post-lock train, and reject the conflicting ordering.
- Add CLI adapter tests for exact push response version, current-version cross-check, missing/conflicting version response, sanitized rejection, explicit status polling, and no retry when accepted version is unresolved.
- Ensure only predeclared incumbent/candidate hashes are accepted, no package/source hash changes after registration/lock, and candidate failure uses the unchanged incumbent or stops.
- Replay registry indices 0–12 byte-for-byte; append amendment at 13; accept exactly two named new workload registrations; enforce recovery artifact→lock adjacency and leave legacy artifact unattested.

No implementation, tests, quota/status calls, or Kaggle submissions occurred during this review.
