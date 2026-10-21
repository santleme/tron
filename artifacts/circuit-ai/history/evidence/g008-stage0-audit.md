# G008 Stage 0 audit (evidence snapshot)

**Status:** verified groundwork; not a G008 pass and not evidence of new CUDA training or learning. No G008 Kaggle kernel was submitted in this stage.

## Frozen inputs

- `training/configs/g008-principal-benchmark.json` wraps the unmodified G007 trial-02 principal config. Canonical compact/sorted config SHA-256: `0ec65b7978e26ecb13b893e76ec358d48c41428db96b0a60dd810f9f24f0e484`.
- `training/configs/g008-split-seed-plan.json` assigns disjoint train, imitation-demo, validation, and final seed bands; excludes historical G007 seeds 0–31; and allocates validation/final as 256 seeds × 2 target seats = 512 game conditions per opponent.
- The current `training.eval.lc_eval.open_space_action` source SHA-256 is `9e5103176218ca092206ba50899a35c43ad488ce825a95af487e76798ce60123`; its declared canonical config SHA-256 is `4043dda35bb8cc2adc43112cb5b00eff50ac721eb062b15d0d83b7ae41510b7f`.
- Seed-plan artifact SHA-256: `ea943f451d6bc0b8d7b0cc054e7e8eb17d65241831298b31794ca9b2846460b4`.
- `schemas/lc-split-manifest-1.schema.json`, `training/eval/provenance.py`, and `training/eval/g008_manifest.py` define a G008-only validated manifest/selection boundary. It validates the frozen seed-plan digest and per-split seed ranges; exact `open-space` code/config identity is rejected outside final evaluation. The final-selection helper returns only `checkpoint_selection` rows.

## Verification performed

- `PYTHONPATH=. python3 -m pytest -q`: **87 passed, 4 skipped**.
- `python3 -m compileall -q training/eval tests/eval`: passed.
- Fresh Kaggle CLI probe: version **2.2.4**; existing private kernel `pascalcoimbra/circuit-ai` reports `COMPLETE`; quota reports GPU **2.23 h used / 27.77 h remaining** (refresh 2026-10-24T00:00Z). Auth file exists with restrictive permissions; no credential contents were read or printed.
- Local runtime probe: Python **3.12.3**, NumPy **1.26.4**; PyTorch, Numba, and ONNX Runtime are absent. Local CUDA training/benchmarking therefore was not run.
- Historical G007 baseline from `artifacts/circuit-ai/runs/lc-circuit-ai-smoke-r4/benchmark-results.json`: 3 repeats, median 1,275.171 joint environment steps/s, rollout 3.194609 s, optimizer 0.017926 s, and 36/36 CUDA updates on one recorded T4. This is historical evidence, not a G008 measurement.

## Boundaries and remaining evidence

- G007 `lc-eval/1` remains unchanged and still names `open-space` as a historical evaluator opponent. It must not be reused as a G008 train/validation source.
- No active G008 train/validation/final consumer exists yet. The new manifest loader is a prescribed boundary for future consumers, not proof that a current training process enforces it.
- No game-level manifest or new training episodes exist yet; actual opponent/data/checkpoint hashes and rows must be materialized and validated before the first G008 data-consuming run.
- The live quota probe does not prove the current Kaggle image's Python/PyTorch/CUDA versions or T4 availability; these must be recorded by a bounded Kaggle runtime job before claiming a G008 CUDA result.
- This workspace has no `.git` repository, so Git status/diff/commit evidence is unavailable.

## Artifact fingerprints

```text
8a9436d9cdad8f2f1873965bb96bca536e9577b0b9646b148ae6ba55d81b0681  training/configs/g008-principal-benchmark.json
ea943f451d6bc0b8d7b0cc054e7e8eb17d65241831298b31794ca9b2846460b4  training/configs/g008-split-seed-plan.json
8d5fb2e4b5271c41890ca725e0cdd397d51d1cf603ac804d2a5e6f26b887ac40  schemas/lc-split-manifest-1.schema.json
e8f657782c36b94edb2916a654f4980d02d90b14d5363c0ff9ba9495912c80e8  training/eval/provenance.py
aff3014ef57a889bff8b7414b695935b64006c266c20c069f1464c494f86492e  training/eval/g008_manifest.py
1e30345b8ff8fb92ee3dec0c92262c2ed9a3300365bcb66ac455e143c53c2b29  tests/eval/test_provenance.py
```
