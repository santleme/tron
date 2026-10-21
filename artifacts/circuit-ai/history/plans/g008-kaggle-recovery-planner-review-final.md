# G008 Kaggle recovery — Planner revision, final review cycle

**Verdict: READY FOR ARCHITECT AND CRITIC REVIEW.**

**Reviewed plan SHA-256:** `d7187d48832e5ffdcfe88239621c1bf62dd20fcaadb30d87a00f4f1d47f99f16`.

The plan keeps exactly two recovery registrations, removes only the fixed attempt-count ceiling for those two registrations, and preserves cumulative project, workload, account-quota, output, wall-time, and protected final-reserve limits. Every attempt freezes a finite `session_timeout_seconds`, passes it through `kernels push --timeout`, and reserves timeout × one T4 before submission; retry is allowed only if a full new reservation fits. Polling is bounded by timeout plus at most ten minutes and unresolved runs cannot be retried.

Official-source investigation corrected the unsupported `kernels list --format json(current_version_number)` assumption and established that CLI 2.2.4's version-qualified status wrapper queries latest. The plan now binds exact push assignment with exact-kernel metadata, brackets latest-status polling with version checks under an exclusive project lock and an explicit no-outside-writer confirmation, or permits version-specific polling only after a live identity contract test. It also requires contract tests for timeout serialization and server-side session termination.

The upstream Cycle-5 gate remains: fix its `/3` lifecycle contradiction; approve corrected exact hashes sequentially Architect then Critic; pass applicable consumer rejection tests before G008 consumers, workload code, packages, registrations, or Kaggle submissions. Recovery event ordering remains registrations → pre-lock benchmark attempts → successful artifact → immediately adjacent lock → post-lock training. Training proceeds even if the benchmark misses 10,000 transitions/s.

No implementation, tests, quota/status calls, packages, registrations, or Kaggle submissions occurred in this planning revision.
