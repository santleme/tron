# G008 Phase-B migration matrix

Status: implementation/consumer audit only. No registry or lock, Kaggle submission, training, benchmark workload, evaluation, or strategic outcome was performed.

## Frozen authority

- Approved seed plan: `training/configs/g008-split-seed-plan.json`, SHA-256 `c6b16733a045e94e7310183e8be8a62cf4327b5d2bdc3eed40bb9d552ce17593`, format `circuit-ai-g008-split-seed-plan/3`.
- Canonical `schedule_spec_digest`: `ca13f87ee6b45f7b4d340f6c62715d4980f9966468e5511f7a06953f056c8ec5`.
- Sequential final-byte approvals: Architect `.omx/plans/g008-cycle5-architect-review-final.md`, then Critic `.omx/plans/g008-cycle5-critic-review-final.md`; each binds this seed hash and the matching reviewed Cycle-5 plan, test-spec, and draft hashes.

## Boundary matrix

| Boundary / consumer | Existing use found | Phase-B guard and negative coverage |
|---|---|---|
| `training.eval.provenance.validate_manifest` | G008 frozen split-manifest validation; no external production callers found in repository | Uses `lc-split-manifest/3` schema. Rejects `/1`, incomplete `/2`, unknown version, bad manifest digest, wrong exact `/3` seed-plan bytes/hash, and missing/wrong/stale global or per-run schedule bindings. `validate_manifest`, `load_manifest`, `rows_for_usage` and `load_selection_rows` now require caller-supplied per-run expanded and realized schedule artifacts; missing maps fail closed. Each binding is validated against those full artifacts via `validate_schedule_binding` before rows are exposed, so resealing arbitrary hash-shaped references is rejected. Existing split isolation/open-space rules are retained. |
| `training.eval.g008_manifest.load_manifest` / `rows_for_usage` | G008 validated loader; helper returns only rows allowed by requested usage | Both call `validate_manifest` with required schedule artifacts before exposing rows. Tests verify the loader and direct usage selector reject a manifest resealed with forged expanded/realized hashes. Loader tampering and selection separation remain covered; old manifest versions fail through the same guard. |
| Schedule creation/binding in `training.eval.provenance` | `build_expanded_schedule`, `build_schedule_aggregate`, `validate_schedule_binding`; no downstream benchmark importer found | Seed-plan reader requires the exact approved `/3` bytes. Expanded schedule recomputes the current global spec digest. Binding verifies exact seed-plan SHA, schedule spec, expanded parent, realized digest and realized payload. Manifest consumers must provide the exact run-keyed expanded/realized artifacts; there is no hash-reference-only mode. Tests reject stale seed plan/spec, wrong parent and missing fields. |
| Benchmark/report import | No G008 benchmark/report import consumer found. `training/eval/lc_eval.py` and `apps/viewer/src/artifacts.ts` consume fixed G007 `lc-eval/1` only. | No G007 behavior changed. Any later G008 importer is required by the approved plan to bind to these guards and add equivalent negative tests before consuming workload artifacts. |

## Files changed / verified in Phase B

- `training/eval/provenance.py` — pinned exact Phase-A plan and upgraded validator/schema selection to `/3`.
- `training/eval/g008_manifest.py` — audited, unchanged; it already validates before exposing rows.
- `schemas/g008-split-seed-plan-3.schema.json` — exact `/3` seed-plan schema.
- `schemas/lc-split-manifest-3.schema.json` — successor schema with required per-run schedule-binding records.
- `tests/eval/test_provenance.py` — `/3` fixtures and rejection tests for old/incomplete versions and stale/wrong bindings.
- `tests/eval/test_g008_schedule.py` — fixes `run_keys` NameError (`keys` local), verifies frozen `/3` benchmark root allocations, exact schedule digest, and stale schedule binding rejection.

## Compatibility boundary

`lc-eval/1` remains unchanged: 64 games per opponent, existing seed/seat contract and viewer validator. G007 reports are not interpreted as G008 split manifests. The historical `lc-split-manifest-2.schema.json` remains in place but is not selected by G008 validation.

## Validation

`PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider tests/eval/test_provenance.py tests/eval/test_g008_manifest.py tests/eval/test_g008_schedule.py -q` — 46 passed, including migration rejection and resealed forged-schedule rejection through loader and row-selection boundaries. This is guard/schedule unit coverage, not evidence of G008 training, benchmark performance, CUDA use, or learning.
