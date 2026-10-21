# G008 Competitive Intelligence — Critic Review (Cycle 5 Schedule Recheck)

**Verdict: ITERATE (planning only)**

## Exact artifacts reviewed

| Artifact | SHA-256 |
|---|---|
| `.omx/plans/g008-competitive-intelligence-plan.md` | `5d97fe0441c7e176ee03e6fee1491d186d99bc04b3b30da8221178d0a7b078b0` |
| `.omx/plans/g008-competitive-intelligence-test-spec.md` | `30fcb15fdb4a1f7ea9693ccf7920514684897e70431c6d6ab6bd0186ee02f0aa` |
| `.omx/ultragoal/brief.md` | `8e001d47335d1e0ba17d730e7a78e688cf891b65fde122a3c753f667b61e4210` |
| `.omx/ultragoal/g008-brief.md` | `3ecdc84a0055be884a6c553a6716076ccfa9c153646fdacfb875dcba0a2f04fa` |
| `.omx/plans/g008-competitive-intelligence-ralplan-draft-cycle-5.md` | `373ae61d6e36402025912c5f72641a592a89f9c71b4e6d2fe5386d43da2fa487` |
| Architect review `.omx/plans/g008-competitive-intelligence-architect-review-cycle-5-schedule-recheck.md` | `83704c7399cc90d8e19ac79d6f1b5f6b20902c4127740ae9d1c06c5b1434ab9b` |
| Prior Critic review `.omx/plans/g008-competitive-intelligence-critic-review-cycle-5-recheck.md` | `2add007c2e06de84d3c9dc370c84eb850d8fc85731bb44c07cea5b2843a18cbe` |

Also read the referenced G008 requirements/context, base PRD/test contracts and `training/README.md`.

## Findings

The prior two cycle-5 blockers are resolved in the canonical documents: the Phase A → limited Phase B bootstrap order is explicit and non-circular, and the executable test contract now covers positive imitation initial/reset dispatch, exact UUID-to-run-key binding/non-aliasing, invalid IDs/cross-domain use, and schedule-spec sensitivity. The reviewed seed-plan version/hash pinning, 5,000,000–5,000,271 benchmark range with permanently unused gap, frozen validation/final identities and counts, full-roster collision scan, per-run finite bounds/digests, and unchanged G008 gates/budgets are synchronized across plan/spec/briefs/draft. Architect has approved this exact hash set. No implementation or workload is authorized by this review.

One remaining actionable ambiguity prevents approval: **the outcome-registration lock is stated but not yet defined as an enforceable lifecycle.** Plan/spec prohibit new run/ablation registrations after outcomes are observed and require a test for “post-outcome registration” (`plan:51`; `test-spec:18,43`), but there is no canonical run-registry record/state or precise lock event/meaning of “outcomes observed.” Each per-run expanded manifest proves only that its own run was registered before its own workload; by itself it cannot prove that another run was not registered after prior outcome data became accessible. Nor is it clear whether timing/benchmark results, training episode returns, validation results, or final data trigger the lock. This matters because Stage 1 produces benchmark measurements before Stage 2's train/imitation run registration; depending on the interpretation, that is either allowed or prohibited. Thus implementers still have to guess how to enforce the requested anti-adaptive-schedule rule.

## Required revision

Add one concise normative rule to plan and test spec, synchronized to both briefs, that:

1. Defines an authoritative append-only experiment/run registry (or equivalent immutable ledger) with canonical registration records binding each run UUID, run type/ablation, code/config/opponent hashes, finite limits, exact root allocation, `schedule_spec_digest`, expanded digest and preflight estimate; record the registration event before workload starts.
2. Defines exactly which event closes registration. To preserve the stated Stage 1 → Stage 2 sequence, distinguish optimization/benchmark timing and parity evidence from training/evaluation outcome data if that is intended; explicitly state whether any game-level outcomes, validation/promotion results, or final-suite data make the lock irreversible. Final-suite data must not be accessible to registration/selection paths.
3. Specify that the registry lock is monotonic and globally checked before any new train/imitation/ablation registration, not merely inferred from one run manifest. Define the permitted pre-lock registration window and ensure all subsequent scheduled runs can be registered before the lock without exceeding the 6 GPU-hour / submission caps.
4. Make tests executable against that state transition: registration before lock succeeds; duplicate UUID, post-lock registration, post-lock ablation, and root/bounds mutation fail; lock cannot be reopened; completed run records link to the pre-existing registration; benchmark evidence alone follows the explicitly chosen policy. The tests must reject a registry/manifest whose history is missing, stale, or inconsistent.

This is a protocol clarity/enforceability gap, not a request to freeze every contingent train/imitation UUID in Phase A. The global `schedule_spec_digest` plus per-run expanded/realized digest split is coherent and appropriately avoids falsely claiming future runs are already known. The revision should retain that design and all existing finite-roster and no-reassignment constraints.

## Acceptance and risk assessment

- **Gates preserved:** Yes. The exact open-space opponent remains final-only; throughput remains >=10,000 joint arena transitions/s by the five-repeat single-used-T4 gate; each final adversary remains >=60% observed wins in >=512 games with Wilson95 reported; validation-only paired bootstrap/Holm regression control, ONNX browser validation, and all quotas/budgets remain intact.
- **Finite-roster loophole:** Mostly closed. Exact upper-bound rosters, hashes, root scans, finite limits and post-outcome prohibitions are explicit. The remaining question is how the prohibition is represented and checked across runs and what data triggers it.
- **Representative implementation simulation:** A consumer can verify one submitted run's UUID/hash/limits/expanded schedule and validate outcome realization. It cannot infer from that artifact whether prior outcome data had already been observed before registration, because no canonical registry/lock history is specified. A test can therefore pass local digest checks while adaptive registration remains possible.
- **Premortem:** The likely failure is each run being locally preregistered yet a later run/ablation being appended after seeing the preceding run's scores, with no shared lock state to detect it; alternatively, teams may overinterpret any benchmark metric as an outcome and deadlock before the planned training stage.

## Summary

- Clarity: strong on schedule contents and digests; registration lifecycle/lock semantics remain underspecified.
- Verifiability: strong for root collisions, run-local digests and holdout suites; insufficient for cross-run post-outcome registration enforcement.
- Completeness: prior gate-order and imitation test gaps resolved; add the canonical registry/lock contract and transition tests.
- Big Picture: preserves the selected throughput-first, holdout-safe G008 strategy and all user-approved thresholds.
- Principle/Option Consistency (consensus): pass; retains the frozen schedule and bounded experiments without introducing DDP/Rust/PyO3/UI scope.
- Alternatives Depth (consensus): pass; a shared registry is a narrow enforcement clarification, not a redesign of digest authority.
- Risk/Verification Rigor (consensus): iterate until the registration lock has an exact trigger, authoritative state and negative tests.
- Deliberate Additions: the pre-mortem above identifies the cross-run adaptive-registration failure and the converse benchmark-lock deadlock risk.

**Stop condition:** No G008 consumer feature, benchmark/training workload, Kaggle submission, or final evaluation until this registry/lock contract is synchronized, re-approved by Architect, and then approved by Critic. Planning-only review; no code or workload performed.
