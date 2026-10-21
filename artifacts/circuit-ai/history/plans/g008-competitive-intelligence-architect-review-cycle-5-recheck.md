# G008 Competitive Intelligence — Architect Review (Cycle 5 Recheck)

**Verdict: APPROVE (planning only)**

The synchronized documents remove both the seed-domain alias and benchmark-root ambiguity. The amendment specifies separate `imitation_demo` initial/reset labels and split-only dispatch, with `imitation-<canonical UUID>` bound to the same `run_id` and distinct from train UUIDs; fallback/aliasing is rejected (`cycle-5 draft:12-29`). It adds this mapping to `schedule_spec_digest`, exact-hash pinning, `/1` and incomplete `/2` rejection, and prohibits consumer/remote execution until a reviewed/frozen `/3+` successor and rejection tests exist (`draft:25-29`).

The canonical plan/test spec and both briefs match (`plan:51-56`; `test-spec:16,30-41`; briefs: `13-19`). All name benchmark roots 5,000,000–5,000,271, measured roots 5,000,000–5,000,004, warmups 5,000,256–5,000,271, and reserved unused gap 5,000,005–5,000,255. The current seed-plan is indeed format `/2`, consistently identified as incomplete. Hard gates and budgets remain intact (`draft:33`; briefs `7,10`; plan `7,45-49`).

**Antithesis:** Multiple briefs and repeated root/codec contracts risk future drift. **Tradeoff:** redundancy helps surface constraints but requires exact synchronization. **Synthesis:** retain them, then freeze one canonical `/3+` seed-plan hash and verify plan/spec/brief consistency before execution.

This approval does not authorize implementation, benchmark, or Kaggle work. Stage-0 data/schedule/rejection tests remain prerequisites.
