# G008 Kaggle recovery — Architect review, fresh final session

**Verdict: APPROVE (planning only; Critic is next sequential reviewer; execution remains blocked by explicit Cycle-5 and CLI-evidence gates).**

**Reviewed plan SHA-256:** `8104144a6b93b5a4efd56fac7dfde7b7dd6dce7e952f064b150162e16624402e`.

## Evidence

- **Cycle-5 predecessor:** The plan marks current `/3` candidate hash non-authoritative and blocks all seed consumers, implementation, package, registration and Kaggle work until the lifecycle is corrected, new exact hash is calculated, separate sequential Cycle-5 Architect→Critic approvals bind that hash, and applicable consumer rejection tests pass (`g008-kaggle-recovery-plan.md`, lines 8, 68–70, 101–102). Benchmark attempts/artifact run pre-lock; the recovery artifact immediately precedes lock; training runs post-lock.
- **Kaggle version/status:** The plan requires one exact positive version in the supported push response, corroborates it against JSON `current_version_number`, and polls only that version. Missing/conflicting evidence fails closed without inference or retry; explicit pre-assignment rejection requires unchanged current-version evidence (`lines 40–48, 108`). Current repo adapter still lacks this path: `_kernel_list_contains` discards list fields, `submit` requests a table and unversioned status and returns raw push output, while `poll_status` already takes an explicit version (`tools/kaggle/cli.py:68–82,101–116,371–389`). The plan treats adapter implementation/verification as a pre-execution gate.
- **Immutable variants:** The plan predeclares exactly incumbent and candidate packages, permits no post-registration fix/rebuild, and falls back to unchanged incumbent or stops (`lines 44, 76–78, 83–84, 107, 109`).
- **Registry and reservations:** Preserve immutable historical events 0–12, append amendment at 13, exactly two recovery registrations, place reservation metadata inside amendment, bind benchmark attempt/artifact before lock, require artifact-lock adjacency, and pin training to lock (`lines 52–54, 77, 89, 104, 108, 112`).
- **Outcome gates:** A miss below 10,000 joint transitions/s is reported but the bounded registered training proceeds (`lines 14, 85, 90`). Compute, quota, output, wall, batch, and final-reserve caps remain (`lines 58–62, 83, 90, 105, 110`).

## Steelman antithesis

Execution depends on unresolved upstream seed-plan repair and a version-evidence path absent from the current adapter. That integration work carries risk, and count-free retries can prolong infrastructure failures.

## Tradeoff and synthesis

Retries avoid rejecting a valid workload due to an arbitrary attempt ceiling, while cumulative resource limits do not bound every rejected push or operator loop. Retain count-free retries as requested, require exact preflight/status/version evidence and terminal resolution before retry, preserve all resource caps, and stop on ambiguity. Planning approval never bypasses the upstream gate.

## Verification recommendations

1. Test corrected Cycle-5 lifecycle/hash plus `/1`, incomplete `/2`, wrong version/hash and stale-binding consumer rejection.
2. Test pinned CLI push response and JSON `current_version_number`, cross-check, exact-version polling, explicit-null rejection, sanitization and missing/nonpositive/conflicting evidence fail-closed.
3. Test two-variant hash immutability and incumbent fallback.
4. Replay registry 0–12 byte-for-byte; validate amendment index 13, exactly two registrations, reservation payload, pre-lock attempts/artifact adjacency and post-lock training ancestry.
5. Assert training proceeds after a benchmark miss below 10k/s.

No tests, code edits, Kaggle quota/status calls, or submissions occurred in this review.
