# G008 Stage 0 evidence addendum — 2026-10-20

This addendum supersedes the stale split-plan/schema hashes in `g008-stage0-audit.md`.
It records current Stage 0 inputs and deliberately does not claim CUDA training,
throughput, strategy learning, or G008 approval.

## Frozen split and provenance artifacts

| Artifact | SHA-256 |
|---|---|
| `training/configs/g008-split-seed-plan.json` (superseded pre-correction candidate) | `0173875c2f13fc3ffe5dbae76c2b6d6e46bc8398da7f8a748988a51b345d4b24` |
| `.omx/evidence/g008-phase-a-frozen-rosters-candidate.json` | `9b3f33a3965539d0b5e178c5b59881a4c7d3d26a66bb9850e13a010d5d52935d` |
| `.omx/evidence/g008-phase-a-roster-slots-v1.jsonl.gz` | `59ef00be9c85e815bb66d6964b306ca589162c6baefa9e5dae240a8790a9a958` |
| `schemas/lc-split-manifest-3.schema.json` | `8e91b4a75384b35206ff584aadaf02f3a16b0093abdb623a5ae9fa181d668c72` |
| `training/eval/provenance.py` | `bc081b237388dfd9a1ea5ec0e309945e07bc55c980267c672b2e129ec9a1c67f` |
| `training/eval/g008_manifest.py` | `aa47212a4f089f7ec1a745fe06c0ff85c6a2deab8018d876c0f98cb62b4bcbbb` |
| `.omx/plans/g008-phase-a-architect-review-full-v1.md` | `e84e8c5c6653f447cce4c8e7680990d4a8cb38986a7e9387221bb985653615a1` |
| `.omx/plans/g008-phase-a-critic-review-full-v1.md` | `35c5203caf03e19e84066334cdb18444c09c1f63d57a390d66c963c2e8aae7eb` |

The reviewed full roster contains 192,768 schedule slots: 191,744 benchmark,
512 validation, and 512 final. Validation uses only G007 step-300; the exact
`open-space` policy remains final-only. The review artifacts contain the exact
schedule and collision audit.

## Verification and execution boundary

- Full local test suite: **127 passed, 4 skipped** on 2026-10-20.
- Kaggle kernel `pascalcoimbra/circuit-ai/6`, source revision
  `g008-r2-173f821e2695d5d4`, was submitted; the latest exact-version status was
  `KernelWorkerStatus.QUEUED`. Status snapshot:
  `.omx/evidence/g008-kaggle-status-v6.json`.
- Submission receipt and pre-registered workload/source hashes are recorded in
  `.omx/evidence/g008-kaggle-submission-v6.json`,
  `.omx/evidence/g008-principal-baseline-registration.json`, and
  `.omx/evidence/g008-principal-baseline-source-freeze.json`.
- No output has been downloaded and no CUDA utilization, benchmark result, or
  learned behavior is evidenced yet. Stage 0 is groundwork only.
- The quota figures in `g008-stage0-audit.md` are the last available snapshot;
  they are not represented here as a fresh quota query.

## Cycle-5 correction status

The approved corrected seed plan is `/3`, SHA-256
`c6b16733a045e94e7310183e8be8a62cf4327b5d2bdc3eed40bb9d552ce17593`;
its canonical schedule digest is
`ca13f87ee6b45f7b4d340f6c62715d4980f9966468e5511f7a06953f056c8ec5`.
Final-byte approval was sequential: Architect
`.omx/plans/g008-cycle5-architect-review-final.md`, then Critic
`.omx/plans/g008-cycle5-critic-review-final.md`. Both bind the exact corrected
seed-plan hash and matching plan/spec/draft hashes. The applicable provenance,
manifest, and schedule migration suite passed 46 tests. The earlier candidate
hash in the table above is superseded and is not authority for consumers. The
prior quota and kernel status records remain historical; current Kaggle
preflight is still required.
