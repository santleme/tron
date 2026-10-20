# CIRCUIT AI — Kaggle CUDA evidence

Runs executed on the private Kaggle kernel `pascalcoimbra/circuit-ai`. No billing settings or paid services were changed. The Kaggle CLI authentication check succeeded; credentials were never printed or copied into this workspace. At the start of this run, the account showed 28.09 GPU-hours remaining.

## Real GPU run and throughput

Kaggle kernel version **4** (`lc-circuit-ai-smoke-r4`, 2026-10-19) confirmed `PyTorch 2.11.0+cu128`, CUDA 12.8, CUDA available, and **two visible Tesla T4s**. The smoke ran on `cuda` (one GPU), executed one CUDA PPO optimizer update over 16 arena steps, exported step-0/step-1 checkpoints, and generated the 3-candidate throughput sweep. It is integration evidence only, not learning evidence.

The repeat-measured sweep selected the largest candidate by median update-only environment steps/s:

| Arenas × rollout / minibatch | Median env steps/s | Median rollout | Median optimizer | Rollout:optimizer |
| --- | ---: | ---: | ---: | ---: |
| 32 × 32 / 256 | 1,204.8 | 0.832 s | 0.0176 s | 47.2× |
| 64 × 32 / 512 | 1,265.5 | 1.601 s | 0.0173 s | 92.5× |
| 128 × 32 / 1,024 | **1,275.2** | 3.195 s | 0.0179 s | 178.2× |

Each candidate used 3 measured repeats and 36 CUDA optimizer updates total. Checkpoint I/O was excluded from candidate selection. The larger batch improved throughput modestly; Python rollout/arena stepping, not the optimizer, dominated. Two-GPU DDP was **not** implemented: at the measured 128-arena operating point, even an impossible zero-cost halving of the optimizer component alone would improve a 3.2125 s update by only about **0.28%** (inference from measured split, before communication overhead). This is not a direct dual-GPU-vs-single-GPU experiment; no claim of measured DDP speedup is made.

## Self-play and learning evidence

Kaggle kernel version **5** (`lc-circuit-ai-selfplay-300`, 2026-10-19) trained 300 PPO updates with 128 independent arenas, 32 rollout ticks, minibatch 1,024, one optimizer epoch, seed 1234, and latest-policy self-play. The run used one T4 (`cuda`), while two T4s were visible.

- 1,228,800 arena steps; **1,200 CUDA optimizer updates**.
- 857.35 s training wall time; 1,439.3 update-only / 1,433.3 wall-inclusive environment steps/s.
- Mean rollout 2.826 s/update vs optimizer 0.0201 s/update (140.3×).
- The new model masks the illegal reverse action in both training and inference. The earlier smoke revealed frequent `reverse_action` deaths; this was fixed before version 5.

Fixed `lc-eval/1` comparisons use the same 64 held-out seed/seat pairs (seeds 0–31, both seats). Win-rate intervals are Wilson 95%; paired survival intervals are 20,000 deterministic bootstrap resamples (seed 42). Full rows and report SHA-256s are in `reports/checkpoint-comparison.json`.

| Checkpoint | vs step-0 checkpoint (W/D/L) | Win rate, Wilson 95% | vs open-space mean survival | Paired survival change vs step-0 |
| --- | ---: | ---: | ---: | ---: |
| step-0 | 26/12/26 | 40.6% [29.5%, 52.9%] | 6.42 ticks | reference |
| step-100 | 61/1/2 | 95.3% [87.1%, 98.4%] | 19.52 ticks | +13.09 [10.97, 15.31] |
| step-200 | 59/1/4 | 92.2% [83.0%, 96.6%] | 19.67 ticks | +13.25 [11.22, 15.27] |
| step-300 | **58/2/4** | **90.6% [81.0%, 95.6%]** | **19.55 ticks** | **+13.13 [11.23, 15.00]** |

This is evidence of learning against the network's own initialization: step-300 decisively beats step-0 on the held-out matrix and survives about three times longer against the rule baseline. It is **not** evidence of a strong general player: step-300 still scored only 1/4/59 against the stronger open-space heuristic (1.6% wins; Wilson 95% [0.3%, 8.3%]). Performance plateaued around step-100 rather than improving monotonically. The next training research should focus on stronger opponent pools/reward shaping and expanding the evaluation set, not DDP.

## Browser and artifact verification

The viewer-ready derived run passed local manifest/schema/file-hash/replay validation. A Playwright Chromium run loaded `step-300.onnx` plus its metadata, showed **VERIFIED · WASM**, then completed a fresh match with no browser page/console errors. It caught and verified a CSS bug where the initial empty-state prompt remained visible over a finished match; `[hidden]` is now explicitly honored. The browser also loaded the complete run folder and showed **RUN BUNDLE VERIFIED**, including 128 replays, evaluation, checkpoints, 1,228,800 training steps and 1,200 CUDA optimizer updates. Durable browser evidence is archived at `reports/browser-acceptance/viewer-acceptance.json` with screenshot `reports/browser-acceptance/viewer-acceptance.png`.

The Kaggle image's ONNX exporter wrote weights as `.onnx.data` sidecars, which the viewer's intentionally single-file checkpoint flow could not load. The untouched Kaggle bundle is retained at `runs/lc-circuit-ai-selfplay-300-kaggle-output/`; review also found that its original manifest omitted hashes for those sidecars, so it is retained as source evidence but correctly fails the strengthened validator. Both Python download validation and browser bundle validation now reject any present `.onnx.data` sidecar, even if listed and hashed, because the viewer does not support external-data ONNX. Tests cover listed and unlisted sidecars. `runs/lc-circuit-ai-selfplay-300-viewer-ready/` retains the exact `.pt` CUDA-trained checkpoints and re-exports their ONNX graphs with inline weights using PyTorch CPU; the manifest/metadata hashes are recomputed and the original remote ONNX/sidecar hashes are listed in `remote-export-hashes.json`. This derived bundle passes the viewer's full bundle validator. The viewer now bundles the explicit Vite WASM assets and imports the WASM-only ONNX Runtime entry point; production JS fell from 451 KB to 111 KB and the unneeded 28 MB JSEP WASM asset was removed.

## Verified failures and fixes

1. Kaggle kernel version 3 completed training but failed during Kaggle's `nbconvert`: pinning old `jsonschema==4.10.3` removed a `pyrsistent` dependency. The requirement is now `jsonschema>=4.18,<5`, preserving Kaggle's existing version; version 4 completed successfully.
2. Kaggle downloads nest outputs at `arena-output/<run_id>/`; the downloader incorrectly expected a root-level manifest. It now finds and promotes only the unique matching run subtree. Regression tests cover nested and sibling outputs.
3. The original policy sampled illegal reverse turns. The PPO actor now masks the opposite heading before rollout sampling and exported inference; a Torch regression test checks all four headings.
4. ONNX Runtime initially fetched a nonexistent bare WASM URL (the browser received HTML). Explicit Vite asset URLs plus the WASM-only package entry repaired browser inference.
5. ONNX external data sidecars broke the viewer's one-file upload contract. New exports set `external_data=False`; a regression test verifies the single-file model. The downloaded version-5 raw artifacts are preserved, and a browser-ready derived export is also supplied.
6. Review found the original raw Kaggle manifest omitted hashes for its ONNX external-data sidecars. Both validators now fail closed on any external-data sidecar (listed or not); the untouched raw export remains archived for provenance, while the sidecar-free derived bundle is the validated viewer artifact.

## Reproduction and locations

- Kaggle benchmark and one-update smoke bundle: `runs/lc-circuit-ai-smoke-r4/`.
- Untouched Kaggle 300-update output: `runs/lc-circuit-ai-selfplay-300-kaggle-output/`.
- Browser-ready 300-update bundle: `runs/lc-circuit-ai-selfplay-300-viewer-ready/`.
- Held-out evaluation reports/replays: `reports/step-{0,100,200,300}/`.
- Paired comparison/bootstrap summary: `reports/checkpoint-comparison.json`.
- Load the viewer-ready folder with **Load Run Folder**, or load `checkpoints/step-300.metadata.json` followed by `checkpoints/step-300.onnx`.

Local verification after fixes: Python suite **76 passed + 8 subtests**; viewer tests **11 passed**; TypeScript typecheck and Vite production build passed; Chromium model/match and full-bundle flows passed (durable evidence linked above).
