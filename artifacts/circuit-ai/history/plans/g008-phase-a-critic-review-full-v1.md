# G008 Phase-A Candidate Review — Critic (full v1)

**Verdict: APPROVE the exact Phase-A candidate bytes below.** This approval is limited to the seed-plan, schedule specification, frozen benchmark/validation/final rosters, and materialized roster sidecar. It is not approval to execute Phase B or any workload.

## Exact reviewed bytes

| Artifact | SHA-256 |
|---|---|
| `training/configs/g008-split-seed-plan.json` | `0173875c2f13fc3ffe5dbae76c2b6d6e46bc839da7f8a748988a51b345d4b24` |
| `.omx/evidence/g008-phase-a-frozen-rosters-candidate.json` | `9b3f33a3965539d0b5e178c5b59881a4c7d3d26a66bb9850e13a010d5d52935d` |
| `.omx/evidence/g008-phase-a-roster-slots-v1.jsonl.gz` | `59ef00be9c85e815bb66d6964b306ca589162c6baefa9e5dae240a8790a9a958` |

The plan SHA is the independently computed current value; the malformed expected hash text supplied for review is not used. The evidence manifest pins this same plan SHA. I read the exact artifacts, the approved Cycle-5 plan/test-spec and both Ultragoal briefs, and the preceding Architect review (`.omx/plans/g008-phase-a-architect-review-full-v1.md`, SHA-256 `e84e8c5c6653f447cce4c8e7680990d4a8cb38986a7e9387221bb985653615a1`).

## Independent checks

- The plan parses as `circuit-ai-g008-split-seed-plan/3`; the canonical embedded schedule-spec payload hashes to the manifest's `9b3878688d8494211cbde92c7f671653239956a97161263277e0db19b2e7ffec`. The model/PPO seed remains 1234, G007 seeds 0–31 remain excluded, and train/imitation/validation/final ranges and `env-seed/v1` mapping conform to the approved plan.
- The exact benchmark partition has roots 5,000,000–5,000,271; measured roots are 5,000,000–5,000,004, warmups 5,000,256–5,000,271, and the intervening 5,000,005–5,000,255 remain explicitly unused. The sidecar contains 191,744 benchmark slots, which equals the complete upper bound for five paired measured roots plus all sixteen assigned warmup roots. Schedule order, all declared root assignments, reset maxima, slot IDs, and canonical row ordering were independently checked.
- I streamed and independently validated all 192,768 JSONL slots: raw stream size 79,416,409 bytes and SHA-256 `50fb3f80be8f63c4a7bd673b48d5ac20dea0879a7f99e277fc31e922323d4b93` match the manifest; all derived seeds recompute from the codec; all slot IDs recompute; all rows are canonical JSON and ordered. Counts are benchmark 191,744, validation 512, final 512. I recomputed all three expanded schedule digests and they match the manifest: benchmark `84c36b9f3a66770754831a7fbfe28b03d660be4219212729a2e4a9333f51e975`, validation `5b110b11db97020bcd2ba9f4fe935c77e9fc6a9439c1a4d645da2be200bff5b0`, final `1ea6b03acd3fd39386635289b35cb9d3be9de46b09d3423a5c4ff8e61285608e`.
- Independent seed grouping yields 130,688 unique seeds and 62,080 duplicate groups. Every duplicate is exactly the same measured paired tuple with incumbent/candidate slot identities; there are no disallowed duplicates or cross-suite collisions. Recomputed complete collision reports equal those in the manifest.
- Validation is exactly `g007-step-300`, 256 roots × seats 0 and 1 = 512 games. Final is exactly one `open-space` opponent, 256 roots × seats 0 and 1 = 512 games, with roster mutation forbidden. No `open-space` opponent identity is allocated to train, imitation, benchmark, or validation; the policy is explicitly inaccessible until candidate freeze and final evaluation. Its source/config fingerprints independently match current code/config. The candidate correctly identifies these fingerprints as bounded identity scope, not a full game-level manifest.
- The validation checkpoint, ONNX, metadata and G007 run manifest hashes, plus current evaluator/loader/engine fingerprints, match their referenced files. Historical G007 training-code hash remains unavailable and is explicitly `null` with a caveat; no claim of historical training provenance is made.
- The registry/lock chronology and pre-lock benchmark/post-lock attestation design are consistent with the approved plan. Candidate status is unreviewed and it explicitly grants no workload authorization.

## Bounded caveat

The historical G007 training-code identity for the validation checkpoint cannot be reconstructed from available artifacts. The frozen manifest states that limitation and binds only available model/artifact hashes and current G008 runtime identities. Preserve that caveat in any later evidence/report; do not describe current evaluator code as the original G007 training build.

## Stop condition / handoff

Phase A is approved only for the exact hashes above. Since these bytes match the Architect-reviewed bytes, the narrowly scoped Phase-B migration guards/schema/negative tests specified in Cycle 5 may now begin. Do not modify the frozen plan, manifest, or sidecar; any byte change invalidates both approvals and requires renewed review. This review does **not** authorize benchmark execution, training/imitation, validation/final evaluation, Kaggle submission, or other workload. Workload remains gated on Phase-B tests passing, all intended registrations and budget checks, and the irreversible registry lock as specified in the approved plan.
