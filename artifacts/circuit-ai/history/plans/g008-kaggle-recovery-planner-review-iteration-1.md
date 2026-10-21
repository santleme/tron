# G008 Kaggle recovery — Planner review, iteration 1

**Recommendation:** the revised recovery draft is ready for another role-specific Architect review. It incorporates all four iteration-0 Architect blockers and their event-order implications. This is Planner evidence only; it does not approve implementation, Kaggle activity, or advance to Critic before Architect completion.

## RALPLAN-DR

### Outcome

Run one paired Kaggle rollout benchmark and one bounded train-only Kaggle workload. Preserve the two GPU-hour final reserve and original G008 approval gates. The fixed submission/attempt-count ceiling remains removed only for the two named recovery registrations; cumulative compute, quota, wall-time, batch, and output caps remain.

### Principles

1. Cycle-5 `/3` is upstream authority only after separate sequential approval of the exact frozen bytes and SHA-256.
2. Benchmark selection and post-lock training are bound to exact predeclared implementation/package hashes.
3. Check the latest assigned Kaggle version and all earlier attempt statuses before push; after push, poll only the newly assigned exact version.
4. Keep validation/final reservations as metadata inside the amendment, with no extra event, registration, workload, or attempt.
5. Preserve the immutable registry history and make each transition reject invalid ordering or ambiguous status.

### Decision drivers

1. Ensure no consumer, workload, or package uses unapproved seed-plan bytes.
2. Prevent concurrent and ambiguous Kaggle executions while allowing count-free retries within resource caps.
3. Ensure the training variant is exactly the one permitted by paired benchmark evidence.
4. Satisfy reservation and event-chain requirements without expanding the recovery workload count.

### Viable options

1. **Two registrations only; prebuild and hash incumbent/candidate variants; embed reservation payload in the amendment:** selected. Preserves requested benchmark + training scope and provides exact provenance.
2. **Add validation/final workload registrations or new event types:** rejected. They are reservation-only and would expand scope beyond the recovery amendment.
3. **Permit post-lock code changes under retry evidence:** rejected. It could detach the trained bytes from the paired benchmark result.

## Revisions incorporated

- Added an upstream gate requiring new Cycle-5 Architect then Critic approval of exact `/3` bytes at SHA-256 `0173875c2f13fc3ffe5dbae76c2b6d6e46bc8398da7f8a748988a51b345d4b24`. Existing Cycle-5 reviews examined earlier plan/test/draft bytes that documented incomplete `/2`; they do not bind `/3`. Applicable migration tests must pass before consumers, workload code, or packages.
- Split Kaggle lifecycle evidence: pre-push preflight records latest known version and all prior attempt statuses; after push, record the newly assigned version and poll only it; append terminal attempt after its exact status. `kaggle_version: null` is allowed only for an explicit server rejection before assignment plus unchanged latest-version evidence.
- Froze code/source/config/seed/backend and package hashes at registration/lock. Training registration lists exact incumbent/candidate variants and threshold policy. Post-lock retries use identical package bytes; no code changes are accepted post-lock. A pre-lock fix requires a predeclared hash, deterministic validation, and a fresh paired comparison bound to the final benchmark artifact/lock; otherwise use the valid unchanged incumbent or stop.
- Placed validation/final reservation metadata in `recovery_amendment.reservation_metadata`, binding the approved `/3` hash, schedule and suite/opponent hashes, counts, seats, roots, protocol and resource bounds. No separate events or registrations are introduced.
- Specified event-order tests that preserve registry events 0–12, require amendment at index 13, reject extra/unmodeled events, enforce preflight→push response/version→exact status→terminal attempt, and require benchmark artifact immediately followed by lock whose preceding-prefix fields equal the artifact hash/index.

## Evidence and handoff

- `sha256sum training/configs/g008-split-seed-plan.json` returned `0173875c2f13fc3ffe5dbae76c2b6d6e46bc8398da7f8a748988a51b345d4b24`.
- `.omx/plans/g008-competitive-intelligence-architect-review-cycle-5-final.md` records its reviewed artifact set and notes the then-incomplete `/2`; it is not an approval of the current `/3` hash.
- `.omx/plans/g008-kaggle-recovery-plan.md` now defines the approval predecessor, immutable registration variants, reservation metadata location, status/version lifecycle, and validator obligations.

Next handoff: send this exact plan to the role-specific Architect for re-review. Launch Critic only after Architect completes and, if needed, the Planner updates and Architect reapproves the revised bytes. No implementation or Kaggle submission is authorized by this Planner review.
