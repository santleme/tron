# G008 Competitive Intelligence — Architect Review (Cycle 5 Benchmark Attestation)

**Verdict: APPROVE (planning only)**

## Exact artifacts reviewed

| Artifact | SHA-256 |
|---|---|
| Plan | `e313676a42dee85579a703734d66443626d76e18d049f0e793087a9105f5237f` |
| Test spec | `0dd0ed99ba103c31bb2b96296e81cc3d7f778a7ef784d8882bd5c2f6d48e82b8` |
| Ultragoal brief | `d8257abd0802e4e34f99b496c16021ffc0e7ae70df021ffcb658afae87079bed` |
| G008 brief | `26dd8ab585a3041fef44d3ff497eb19d1ad7c06b2e1a5a61a506f8d01a9a48f8` |
| Cycle-5 draft | `29e6f4c425c4bf65e08be2c75c21a5b1843027e5f84a95bf494fcf7dbebecb11` |

## Findings

The pre-lock benchmark evidence chronology is now coherent. Each immutable benchmark artifact is created with its content hash, creation-time prefix head/index and exact prior registration; it does not claim a future lock hash. Following `registration_closed`, a separate `benchmark_attestation` event binds the immutable content hash, original prefix anchors, registration and resulting lock head. Downstream use is blocked until that attestation validates (`plan:53-59`; `test-spec:24,28`).

This attestation fits the append-only chain: it is a specified post-lock evidence event that references the already-created lock hash and does not alter the lock or benchmark artifact. Event hashing is non-self-referential. Tests cover pre-lock creation, no downstream consumption before attestation, successful exact binding, and rejection of missing/stale or mismatched content, prefix, registration, or lock (`test-spec:24,28`).

The prior concerns remain resolved: benchmark is the sole pre-lock workload and exposes only performance/parity data; validation/final are reservation-only before lock; validation operates post-lock for registered candidates; final is evaluator-only after candidate freeze. The Phase A/B migration sequence, imitation domain/UUID rules, finite roster registration, holdout boundaries, gates, budgets and scope exclusions are preserved.

**Antithesis:** The separate artifact and attestation require consumers to join two provenance records correctly.

**Tradeoff:** Delaying all benchmark evidence until after lock would be simpler but conflicts with the permitted pre-lock measurement chronology; attestation adds one validator/event type.

**Synthesis:** Preserve immutable prefix-anchored measurements and require an exact append-only post-lock attestation before downstream consumption. This keeps chronology and provenance explicit without allowing benchmark strategic outputs.

No implementation, seed-plan change, benchmark, training or Kaggle activity was performed. Critic re-review and durable Ralplan handoff remain required.
