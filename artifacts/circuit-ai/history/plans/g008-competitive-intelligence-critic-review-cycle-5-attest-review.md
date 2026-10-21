# G008 Competitive Intelligence — Critic Review (Cycle 5 Benchmark Attestation)

**Verdict: APPROVE (planning only)**

## Exact artifacts reviewed

| Artifact | SHA-256 |
|---|---|
| `.omx/plans/g008-competitive-intelligence-plan.md` | `e313676a42dee85579a703734d66443626d76e18d049f0e793087a9105f5237f` |
| `.omx/plans/g008-competitive-intelligence-test-spec.md` | `0dd0ed99ba103c31bb2b96296e81cc3d7f778a7ef784d8882bd5c2f6d48e82b8` |
| `.omx/ultragoal/brief.md` | `d8257abd0802e4e34f99b496c16021ffc0e7ae70df021ffcb658afae87079bed` |
| `.omx/ultragoal/g008-brief.md` | `26dd8ab585a3041fef44d3ff497eb19d1ad7c06b2e1a5a61a506f8d01a9a48f8` |
| `.omx/plans/g008-competitive-intelligence-ralplan-draft-cycle-5.md` | `29e6f4c425c4bf65e08be2c75c21a5b1843027e5f84a95bf494fcf7dbebecb11` |
| Architect approval `.omx/plans/g008-competitive-intelligence-architect-review-cycle-5-attest-review.md` | verified APPROVE for the same five hashes |

Also read the G008 requirements/context, base MVP PRD/test contract, `training/README.md`, previous cycle-5 Architect/Critic reviews, and the current seed plan plus `training/eval/provenance.py` and `training/eval/g008_manifest.py` to check representative migration and consumer paths. The seed plan is still the known incomplete `/2`; no reviewed successor or workload is treated as present.

## Findings

The chronology amendment resolves the prior lock/hash ordering contradiction. A pre-lock benchmark artifact binds its immutable content SHA-256, creation-time prefix head/index, and exact earlier benchmark registration event/hash; it does not assert a future lock hash. The `registration_closed` event is then appended at the next index and its ordinary event hash becomes `lock_head_hash`. A distinct post-lock `benchmark_attestation` event binds the unchanged artifact hash and original prefix/registration anchors to that lock hash. The artifact is not rewritten, and downstream use is forbidden until exact attestation validation succeeds (`plan:53–59`, `test-spec:22–28`). This is non-self-referential and compatible with the append-only event chain.

The attestation does not reopen registration or enable strategic output leakage: only registered-run completion/evidence events are permitted after lock; the benchmark attestation is explicitly specified as such an evidence event, while registration and identity/schedule mutation remain forbidden. The plan keeps benchmark evidence restricted to runtime, transition/action parity and throughput/timing; internal PPO returns are discarded and no game outcomes, replays, checkpoint-quality or strategic metrics may be persisted/exposed. Pre-lock validation/final are reservation metadata only. Post-lock training/validation/final access remains constrained to registered runs, promotion-only frozen validation, and the final evaluator after candidate freeze. Test obligations cover valid prefix-only artifact creation, no pre-attestation downstream use, exact attestation, missing/stale/altered-content/wrong-prefix/wrong-registration/wrong-lock rejection, and post-lock registration rejection (`test-spec:24–28`).

Representative paths are actionable: (1) the benchmark producer can register, create a prefix-anchored immutable throughput/parity artifact, and stop before lock; (2) the lifecycle controller can lock, append one attestation per eligible artifact, and validate its exact references before exposing the artifact to later reports; (3) a later strategic consumer can require the lock pin and reject un-attested benchmark ancestry. Existing `provenance.validate_manifest` and `g008_manifest` are explicitly identified for Phase B migration checks; later consumers must provide their own equivalent guards before workload use. The test spec is specific about positive/negative assertions and the global chain’s hash/index rules, so implementers do not need to invent chronology behavior.

Prior blockers remain closed: ordered Phase A freeze then minimal Phase B migration guard/tests; exact `/3+` plan hash binding and `/1`/incomplete `/2` rejection; explicit imitation initial/reset domains and same canonical UUID binding/non-aliasing; benchmark roots `5,000,000–5,000,271` with measured roots `5,000,000–5,000,004`, warmups `5,000,256–5,000,271`, and permanently unused gap `5,000,005–5,000,255`; finite per-run registration and full upper-bound collision scans; frozen validation/final rosters. Requirements/context and both briefs retain the user-approved gates and exclusions: exact open-space final-only, >=10k joint two-agent transitions/s by five full-PPO repeats on the specified single-used T4, >=60% observed wins per unseen final opponent across >=512 games with Wilson95 reported (not an LCB gate), separate W/L/D and alternating reproducible seats, validation-only paired bootstrap/Holm anti-regression, real-browser self-contained ONNX, fixed Kaggle caps/reserve, no paid services, no unjustified DDP, Rust/PyO3, or league UI.

The local hash-chain threat model is stated honestly: partial/inconsistent edits are detected only while an independently captured anchor remains; rewriting the registry and all anchors is outside its guarantees. No unsupported signature/authenticity claim or paid dependency was added.

**Antithesis:** A separate post-lock attestation creates a two-record join that every downstream reader must validate; delaying all benchmark publication until after lock would simplify consumption.

**Tradeoff:** Delaying publication conflicts with the approved pre-lock benchmark stage and chronology. The additional event and validator are a bounded complexity cost; they preserve immutable pre-lock measurements while preventing downstream use until tied to the locked registration prefix.

**Synthesis:** Keep artifacts immutable and prefix-anchored at creation; require the exact append-only post-lock attestation before downstream consumption. The specified tests enforce the join and verify that attestation does not permit new registration or strategic-data access.

## Deliberate additions

- **Pre-mortem:** The likeliest implementation failure is a report/import consumer accepting the artifact’s content hash without validating both its original prefix/registration and the separate lock attestation, or a producer rewriting the artifact after attestation. The plan addresses both through mandatory exact-anchor validation and immutability tests. Another plausible failure is mistakenly requiring the lock hash inside the artifact at creation, recreating the impossible forward reference; the explicit creation/attestation lifecycle and test cases prevent that.
- **Expanded verification:** In addition to the listed tests, implementation should test duplicate attestation rejection/idempotency and verify that appending the attestation leaves the artifact bytes/hash unchanged, while lock state remains closed and every post-lock registration attempt still fails. These are direct checks of already-specified immutability and no-reopen invariants, not new acceptance gates.

## Summary

- Clarity: pass; the creation-time prefix versus post-lock attestation distinction is explicit and synchronized.
- Verifiability: pass; event order, references, immutable content, downstream denial and negative cases are executable obligations.
- Completeness: pass; migration, imitation, roster, root, holdout, gate, budget and scope rules remain specified.
- Big Picture: pass; preserves the throughput-first G008 sequence without allowing benchmark strategic information to select checkpoints.
- Principle/Option Consistency (consensus): pass; remains parity-first, append-only, holdout-safe and scope-bounded.
- Alternatives Depth (consensus): pass; the simpler delay-until-lock alternative is considered, with a concrete chronology tradeoff; attestation is the minimal fit for the approved sequence.
- Risk/Verification Rigor (consensus): pass; prior chronology defect is closed and test cases cover tampering/stale links and forbidden access.
- Deliberate Additions: pass; credible failure modes and supplemental immutability/no-reopen checks are included.

**Stop condition:** Planning-only approval. This review does not authorize seed-plan migration, consumer/feature implementation, benchmark workload, training, ONNX export, or Kaggle activity. Proceed only through the Ralplan handoff; implementation remains gated by the approved Phase A/Phase B sequence and durable Ultragoal authority.
