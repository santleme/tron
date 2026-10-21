# G008 Competitive Intelligence — Architect Review (Cycle 5 Lock Final)

**Verdict: ITERATE (planning only)**

## Exact artifacts reviewed

| Artifact | SHA-256 |
|---|---|
| Plan | `f0317cade353de389dc34bb5fcb3b4c6236d85d28a6b4f8ec44ca173a7554953` |
| Test spec | `92084eaa9f84e692946e76827166e0952b074fafe94a366eff79046b40ba646c` |
| Ultragoal brief | `c2d8ddda0ca1f524ee64249f1e54559ab1dd53b1ffa5ccc616905d637b0bc5c4` |
| G008 brief | `6dd7934ade075766d360a3309c6dad94d7f2ee6206c52c96acc7c00f8040d4d8` |
| Cycle-5 draft | `a51a859ba0cb371cc02b90e4e6f767adfc5ae889ef362564139bbd747c28fda9` |

The lock hash ordering is now non-self-referential and the threat model is honest about local hash-chain limitations. Validation/final execution is correctly distinguished from pre-lock metadata reservation. One chronology mismatch remains: the plan requires all evidence to pin the future `lock_head_hash`, while benchmark performance/parity evidence is permitted before the lock exists.

Pre-lock benchmark artifacts should pin only the pre-lock prefix hash/index. After the lock, append an attestation event binding each immutable benchmark artifact content hash and its prefix provenance to `lock_head_hash`. Downstream consumers should reject un-attested, altered, or mismatched pre-lock evidence. Tests must cover prefix-only pre-lock acceptance, post-lock exact attestation, and stale/unbound benchmark rejection. Alternatively, delay benchmark evidence finalization until lock, but that introduces a timing/recording tradeoff.

**Antithesis:** Pre-lock benchmark evidence anchored only to the prefix may be weakly tied to the subsequent locked registry.

**Tradeoff:** Embedding a not-yet-existing lock hash is chronologically impossible; delaying evidence finalization complicates operational timing records.

**Synthesis:** Keep benchmark outputs immutable and prefix-anchored at creation, then append a post-lock attestation that binds their content hashes to the lock head before any downstream use.

Other reviewed constraints remain preserved: benchmark-only pre-lock work; validation/final metadata-only reservation before lock; exact imitation domains/UUID mapping; migration order; finite rosters; root gap; all gates/budgets/holdouts and implementation exclusions. No code, seed-plan, workload or Kaggle activity was performed.
