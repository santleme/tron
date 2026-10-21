# G008 Kaggle recovery — Architect review

- Decision: **APPROVE (planning only)**
- Reviewed plan: `.omx/plans/g008-kaggle-recovery-plan.md`
- Exact SHA-256: `0358c3c00ffa438a3bd033939687827774da5f04ba54746c075eb4680b384d72`
- Approved Cycle-5 seed plan: `c6b16733a045e94e7310183e8be8a62cf4327b5d2bdc3eed40bb9d552ce17593`
- Review order: Architect, followed by Critic on the same plan bytes.

The lifecycle preserves events 0–12, registers the amendment and exactly two
workloads, resolves benchmark attempts before the lock, appends the benchmark
artifact immediately before `registration_closed`, and runs training after the
lock. The no-count-ceiling exception is limited to those two workloads, with
cumulative compute budgets and a full timeout × one-T4 reservation per attempt.
The reviewed Cycle-5 approvals and the 46 migration tests match the approved
seed hash.

This is planning approval only. Fresh history/quota reconciliation, pinned
CLI/SDK contracts, lifecycle implementation and tests, and live Kaggle
preflight remain required before registration or submission.
