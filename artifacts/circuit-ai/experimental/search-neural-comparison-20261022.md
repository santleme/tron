# Simultaneous search and Policy-Value results (prior campaign snapshot)

> Historical snapshot from the initial planner/neural comparison. The subsequent 12×12 decisive Rust-vs-selected-DAgger1 and spatial-checkpoint tournaments, robustness screens, and current candidate selection are documented in [rust-dagger-comparison-20261022.md](rust-dagger-comparison-20261022.md). Where candidate settings or conclusions differ, the subsequent report is current.

All tournaments below ran in isolated Kaggle CPU kernels with GPU and internet disabled. They used canonical arena transitions, fresh development seeds, both seat assignments, and validated output manifests. No official registry was modified and no reserved final evaluation was used.

## Planner selection

| Matchup | Games | Wins / draws / losses for left side | Score rate (draw = 0.5) | 95% seed-cluster bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| Fixed-depth vs DUCB1-Tuned | 512 | 455 / 44 / 13 | 0.932 | 0.914–0.948 |
| Fixed-depth vs Regret Matching | 512 | 430 / 61 / 21 | 0.899 | 0.879–0.920 |
| Fixed-depth vs v15 step-236 spatial checkpoint | 512 | 491 / 20 / 1 | 0.979 | 0.968–0.988 |
| Fixed-depth vs interceptor heuristic | 512 | 412 / 85 / 15 | 0.888 | See campaign analysis |

References: `simsearch-native-decisive-20261022-f45a9c3d/analysis.json`, `simsearch-decisive-20261022-c0d69524afcb/decisive-analysis.json`, and `simsearch-v15-step236-20261022-a0995b17/analysis.json` in this directory tree.

## Policy-Value checkpoint

The compact MLP was trained on a T4 for 60 epochs, using 13,863 examples from 256 development seeds. The grouped holdout had 1,296 examples from 26 seeds; policy loss was 1.1075 and value MSE was 0.1961. The checkpoint and ONNX hashes are recorded with the validated training output in `pv-train-scale-dev-20261022-c14a8e06/`.

The checkpoint was tested as a standalone policy against fixed-depth on two independent 256-seed blocks:

| Search configuration | Fixed-depth W/D/L | Fixed-depth score rate | 95% seed-cluster bootstrap CI | Mean decision time (search / policy) |
| --- | ---: | ---: | ---: | ---: |
| Depth 3, 56 edges vs policy | 441 / 67 / 4 | 0.927 | 0.910–0.942 | 10.21 ms / 1.08 ms |
| Depth 1, 16 edges vs policy | 447 / 47 / 18 | 0.919 | 0.898–0.938 | 1.31 ms / 0.96 ms |
| Depth 1, 12 edges vs policy (fresh 256-seed follow-up) | 439 / 52 / 21 | 0.908 | 0.887–0.928 | 0.902 ms / 0.556 ms |
| Depth 1, 9 edges vs policy (fresh 256-seed follow-up) | 415 / 54 / 43 | 0.863 | 0.836–0.889 | 1.431 ms / 1.115 ms |
| Depth 1, 8 edges vs policy (fresh 256-seed follow-up) | 329 / 73 / 110 | 0.714 | 0.678–0.750 | 0.962 ms / 0.768 ms |
| Python depth 1, 9 edges vs policy (same-host native campaign) | 416 / 57 / 39 | 0.868 | 0.843–0.894 | 1.331 ms / 0.931 ms |
| Rust depth 1, 9 edges vs policy (same-host native campaign) | 412 / 63 / 37 | 0.866 | 0.842–0.890 | 0.373 ms / 0.834 ms |
| Rust depth 1 vs Python depth 1 (same-host, same seeds) | 210 / 92 / 210 | 0.500 | 0.500–0.500 | 0.341 ms / 1.279 ms |

The depth-1/16-edge setting was first calibrated over 32 seat-balanced games (1.53 ms vs 1.50 ms per decision). In its 512-game follow-up, search averaged 1.306 ms vs 0.962 ms for the network, so the result was not latency matched. Three CPU-only Kaggle runs compared edge budgets 12, 9, and 8 on 256 fresh seeds with both seat assignments per run. Budget 12 scored 439/52/21 (W/D/L), score 0.908 (95% seed-cluster bootstrap interval 0.887–0.928); latency was 0.902 ms mean / 1.802 ms p95 for search and 0.556 / 0.677 ms for PV. Budget 9 scored 415/54/43, score 0.863 (0.836–0.889); latency was 1.431 / 2.911 ms vs 1.115 / 1.331 ms. Budget 8 scored 329/73/110, score 0.714 (0.678–0.750); latency was 0.962 / 1.527 ms vs 0.768 / 0.921 ms. Absolute latency varies between Kaggle CPU hosts. None of these Python runs is strictly equal-latency: search mean was 1.25–1.62x PV and p95 was 1.66–2.66x. Budget 9 completed depth 1 on 97.1% of decisions; budget 8 did so on 65.8%. A replay audit confirmed all 116 budget-8 `invalid_action` deaths happened when no legal move existed, so they were forced deaths.

A same-host campaign ran the Python and native Rust planners at edge budget 9 against PV and directly against each other, with 256 fresh seeds and both seats in all three 512-game matchups. Python scored 416/57/39 against PV (score 0.868, 95% interval 0.843–0.894; 1.331 ms mean / 2.725 ms p95). Rust scored 412/63/37 (0.866, 0.842–0.890; 0.373 / 0.679 ms), a 3.57x lower mean decision latency than Python with essentially identical strength. In the direct Rust-vs-Python matchup the result was 210/92/210 and score rate 0.500; both implementations completed depth 1 at nearly the same rate and used the same transition/leaf counts. The same-host run establishes the Rust port preserves measured strength while reducing latency by 72% on average and 75% at p95. Rust is now the selected planner implementation. The standalone PV remains unselected because it scored below both implementations in this campaign. Additional neural training is not justified; the checkpoint was not promoted or used for a reserved evaluation.

## Implementation and viewer

The Kaggle runner now supports only the explicitly hash-pinned experimental Policy-Value checkpoint, loads it through the shared v2 model factory, and records CPU inference timings. An initial failed package attempt exposed an unwanted import from the PPO runtime; the model definition was moved to `training/search/policy_value.py`, so isolated search packages no longer need PPO files.

The viewer offers the fixed-depth planner as Rust/WASM alongside ONNX Policy-Value. The same Rust source now also builds as a native Kaggle shared library; its 512-game head-to-head with Python preserved score while reducing mean decision latency 3.57x. Search runs once per simultaneous tick and samples each rider's mixed equilibrium policy from its derived match stream. The optimized production WASM artifact is generated during `dev` and `build` and weighs 33.1 KB (14.6 KB gzip). The browser engine continues to resolve all game transitions.

Local evidence: 40 focused Python tests passed (2 PyTorch-only checks skipped because local PyTorch is absent); viewer typecheck passed; all 15 viewer tests passed; production build passed; all four Rust tests passed; planner outputs matched Python on 53 deterministic states; and Rust transition states matched the canonical TypeScript engine on 1,375 joint transitions. A T3 preview browser was unavailable in this headless environment, so visual interaction was not separately inspected.

## Decision

Rust fixed-depth is the evidence-backed primary planner: its native Kaggle build matched the Python planner and delivered 3.57x lower mean decision latency against the same Policy-Value opponent. Keep search evaluations on Kaggle CPU; retain the Policy-Value checkpoint only for further justified experiments and do not use final/open-space seeds for tuning. Strict equal-latency strength against the Policy-Value network remains unmeasured. The 9-edge setting is the current stronger low-budget candidate; budget 8 reduces strength without closing the latency gap. The original final goal of independent evaluation above 60% against open-space remains unmeasured by design.
