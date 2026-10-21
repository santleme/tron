# G008 Kaggle recovery — Architect review, fresh iteration 1

**Verdict: APPROVE (planning only; execution remains blocked by explicit predecessor gates).**

The revised recovery plan is ready for sequential Critic review of these exact plan bytes.

## Evidence

- **Cycle-5 lifecycle/hash:** The recovery plan supersedes the old `/3` candidate hash `0173875c...4b24`, names the conflict between benchmark-before-lock and lock-before-benchmark, and requires correcting the seed-plan, test spec and affected draft before a new sequential Cycle-5 Architect→Critic review binds the exact corrected bytes/hash. Applicable consumer rejection tests must pass before G008 implementation/package/registration (`g008-kaggle-recovery-plan.md`, lines 8, 70, 102). The current `/3` file is still contradictory around lines 529 and 533; this remains an execution blocker, but the plan does not claim it is resolved.
- **Version assignment:** Before push, record latest `current_version_number`, its status and every prior attempt status. An accepted push must return one exact positive version; cross-check against post-push JSON current version and poll only that version. Missing, nonpositive or conflicting evidence stays unresolved with no guessed version or retry. A null version is allowed only for explicit pre-assignment rejection and unchanged current version, with sanitized evidence (`plan`, lines 40–48, 108). The plan does not rely on a full historical version API.
- **Adapter capability:** Current project adapter does not yet provide required version evidence: `_kernel_list_contains` checks owner/slug but discards row fields (`tools/kaggle/cli.py:101–116`); `submit()` lists in table format, checks unversioned status and returns raw stdout (`:371–389`); `poll_status()` accepts an explicit version (`:68–82`). The plan makes a supported CLI/API evidence adapter and fail-closed behavior prerequisites. Kaggle CLI docs list JSON format support for `kernels list`; implementation must verify exact fields against the pinned CLI and fail closed if unavailable. Sanitized rejection evidence preserves the existing stderr privacy boundary.
- **Variants:** Exactly two package/code variants, incumbent and candidate, are prebuilt and frozen in both registrations. No post-registration fix, rebuild or packaging exception is allowed. Candidate defect/parity failure uses the unchanged valid incumbent or stops; code changes require a separate reviewed paired cycle.
- **Reservations/history/budgets:** Reservation-only suite/opponent hashes, counts, seats, roots and resources live in the amendment, with no third registration/event. The 13-event prefix remains immutable; amendment is index 13; historical artifact 10 remains legacy/unattested. Recovery benchmark attempts/artifact are pre-lock, artifact and lock are adjacent, and training runs post-lock pinned to the exact lock. Count-free retries remain bounded by cumulative compute/quota/wall/batch/output/final-reserve limits.

## Steelman antithesis

This protocol could delay results with seed-plan corrections and CLI parsing for a user-authorized benchmark.

## Tradeoff and synthesis

Count-free retries improve recovery, but accepted pushes must map to one exact version without undocumented history APIs. Keep no internal retry ceiling; proceed only with supported push version evidence corroborated by JSON `current_version_number`, exact-version terminal status, immutable package hashes and cumulative budget checks. If those facts are unavailable, fail closed without retry.

## Recommended verification

- Test corrected Cycle-5 benchmark-before-lock/train-after-lock lifecycle bound to its new hash.
- Mock CLI JSON/version parsing, push-version cross-check, sanitized rejection, missing/nonpositive/conflicting versions, and exact-version-only polling.
- Test two-variant selection/fallback/stop and rejection of changed package bytes.
- Replay registry events 0–12 byte-for-byte; require amendment at 13, exactly two new registrations, terminal pre-lock benchmark attempt, adjacent artifact/lock, and post-lock training pinned to that lock.

No tests, code edits, Kaggle quota/status calls, or submissions occurred in this review.
