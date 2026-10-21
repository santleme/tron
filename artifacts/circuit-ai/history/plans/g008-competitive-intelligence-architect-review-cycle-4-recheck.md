# G008 Competitive Intelligence — Architect Review (Cycle 4 Recheck)

**Verdict: APPROVE (planning/handoff only)**

The only blocker from the prior cycle-4 review is closed in the canonical plan, G008 test specification, and both execution briefs. They explicitly require an in-place v2+ successor to `training/configs/g008-split-seed-plan.json`, preservation of partition ranges/G007 exclusions/open-space hashes, addition of benchmark roots and codec/model-vs-environment seed separation, full schedule collision scan and digest, hash binding in every manifest, and per-consumer negative tests rejecting v1 before any G008 consumer implementation or remote work (`plan:51`; `test-spec:16,33`; briefs: Cycle-4 contract and Stage 0).

The seed JSON itself remains v1 (`training/configs/g008-split-seed-plan.json:2`), so this is approval of the plan only. Execution remains blocked until the migration gate is fulfilled; the current v1 artifact is not adequate for execution. I independently rechecked the codec vector: 133-byte payload, SHA-256 `e1d2cd7a82ca19bc5277a87aa4819271802e14c23fb9d071f92b6b2e3d539076`, derived uint64 `16272294329707010492`.

Hard gates remain conjunctive and transparent; benchmark timing and descendants are contained; exact open-space holdout, final evaluation rules, budgets, and no-DDP boundary remain present. One minor non-blocking alignment remains: the high-level interview spec mentions evaluating Rust, whereas the canonical G008 plan excludes Rust/PyO3. The approved plan governs execution.

**Antithesis:** The v1 JSON is still present, so an executor could mistakenly start against it. **Tradeoff:** do not edit the seed authority during planning review, but do not treat a planned migration as completed. **Synthesis:** approve this plan revision while retaining a hard fail-closed Stage 0 gate requiring the actual v2+ artifact, hashes, and tests before implementation or Kaggle work.

This review is planning-only and does not authorize source edits, benchmark execution, Kaggle submissions, or training.
