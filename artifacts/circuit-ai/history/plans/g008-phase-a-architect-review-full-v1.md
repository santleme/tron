# G008 Phase-A Candidate Review — Architect (full v1)

**Verdict: APPROVE the exact Phase-A candidate bytes below.** This review covers the `/3` seed plan, frozen opponent rosters, schedule digests, and materialized benchmark/validation/final tuple sidecar only. It is not a Critic approval and does not independently authorize Phase B, benchmark execution, training, evaluation, or Kaggle work; continue only after the required sequential Critic approval for these exact bytes.

## Exact artifacts reviewed

| Artifact | SHA-256 |
|---|---|
| `training/configs/g008-split-seed-plan.json` | `0173875c2f13fc3ffe5dbae76c2b6d6e46bc839da7f8a748988a51b345d4b24` |
| `.omx/evidence/g008-phase-a-frozen-rosters-candidate.json` | `9b3f33a3965539d0b5e178c5b59881a4c7d3d26a66bb9850e13a010d5d52935d` |
| `.omx/evidence/g008-phase-a-roster-slots-v1.jsonl.gz` | `59ef00be9c85e815bb66d6964b306ca589162c6baefa9e5dae240a8790a9a958` |

The first hash differs from the malformed expected hash text in the review request; it is the independently computed SHA-256 of the current file. The evidence manifest pins this same plan hash.

## Independent checks

- The plan parses as `circuit-ai-g008-split-seed-plan/3`, retains model/PPO seed `1234`, historical G007 exclusion `[0,31]`, and the approved partition bounds. Benchmark range is `[5000000,5000271]`, measured roots `[5000000,5000004]`, warmups `[5000256,5000271]`; the intervening roots `[5000005,5000255]` are explicitly unused and cannot be reassigned (`training/configs/g008-split-seed-plan.json`, `partitions`, `schedule`, `environment_seed_codec`). This matches the Phase-A and root contracts in `.omx/plans/g008-competitive-intelligence-plan.md:51,61-65` and `.omx/plans/g008-competitive-intelligence-test-spec.md:18,43-45`.
- Canonical SHA-256 of the embedded `schedule_spec_payload` independently recomputes to `9b3878688d8494211cbde92c7f671653239956a97161263277e0db19b2e7ffec`, matching the manifest claim. All three suite `expanded_schedule_digest` values independently recompute from the exact ordered sidecar rows plus their frozen collision scan:
  - benchmark: `84c36b9f3a66770754831a7fbfe28b03d660be4219212729a2e4a9333f51e975`
  - validation: `5b110b11db97020bcd2ba9f4fe935c77e9fc6a9439c1a4d645da2be200bff5b0`
  - final: `1ea6b03acd3fd39386635289b35cb9d3be9de46b09d3423a5c4ff8e61285608e`
- The gzip sidecar decompresses to exactly 79,416,409 bytes with SHA-256 `50fb3f80be8f63c4a7bd673b48d5ac20dea0879a7f99e277fc31e922323d4b93`; it contains 192,768 LF-terminated canonical JSONL rows in the prescribed order and schema. All 192,768 derived seeds independently recompute from `env-seed/v1`; slot IDs independently verify using `backend_id=shared` for non-paired slots and the exact incumbent/candidate identity for measured paired slots. Slot IDs are unique; ordering is monotonic by the specified schedule tuple.
- Independent grouping finds 62,080 duplicate-seed groups, all and only exact same-tuple measured-pair reuses (two sides); zero disallowed duplicates and zero cross-suite collisions. The complete duplicate groups match the evidence manifest. Counts are benchmark 191,744, validation 512, final 512; unique derived seeds total 130,688. The benchmark roots and warm-up assignment coverage match the plan. The measured schedule correctly maps each root/arena's reset `g=1..96` to `update=(g-1)//32`, `game=g`, `reset=((g-1)%32)+1`, consistent with `plan.md:64-65`.
- Validation freezes only `g007-step-300`, exactly 256 roots × seats 0 and 1 = 512 games. Final freezes only `open-space`, the same 512-game/seat pattern, with mutation forbidden. No open-space identity is assigned to train, imitation, benchmark, or validation; the final roster and “no read/use before candidate freeze” constraints are explicit. These match `plan.md:15,51,65,86,89,97` and test spec `18,54`.
- Recomputed the current source fingerprint for `open_space_action` and current G008 validation runtime file hashes, plus the frozen G007 checkpoint, ONNX, metadata, and run-manifest hashes; all match the candidate. The historical G007 training-code SHA remains unavailable and is honestly recorded as `null` with a caveat; the candidate does not misrepresent current evaluator identity as historical training provenance.
- The registry/lock chronology and benchmark-attestation lifecycle in the evidence preserve the approved non-self-referential lock and pre-lock artifact design (`plan.md:53-59`). The candidate explicitly denies workload authorization on its own.

## Bounded caveat

The frozen validation opponent’s historical training-code identity cannot be reconstructed from the existing G007 artifacts. The manifest binds the available immutable model/artifact hashes and current G008 loader/engine/evaluator hashes and states this limitation plainly. This is not a Phase-A blocker under the approved plan; keep the caveat in every report and do not imply original G007 build provenance.

## Stop condition

Phase A is approved for these exact hashes, pending Critic’s sequential review. No edits to the candidate or implementation were made. This report alone does not authorize migration guards or any workload. Once Critic approves the same exact bytes, Phase A may be treated as frozen and the plan’s narrowly scoped Phase-B migration guard/tests may begin; all workload remains gated on Phase B passing (`plan.md:51,68-70`).
