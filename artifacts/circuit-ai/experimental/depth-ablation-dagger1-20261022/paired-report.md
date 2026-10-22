# Paired search depth/budget ablation against DAgger1

Both variants used the same 64 newly generated development seeds, with both seats per seed (128 games per variant). The comparison is nonofficial and used no reserved evaluation seeds. DAgger1 was pinned to `dagger1-selected-20261021` by the existing checkpoint SHA allowlist.

| Variant | Search configuration | W / D / L | Win rate | Mean decision | p95 decision | Completed depth | Transitions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | depth 3, edge budget 56 | 124 / 3 / 1 | 96.88% | 1.392 ms | 2.699 ms | 2.030 | 131,387 |
| Deeper | depth 4, edge budget 112 | 125 / 3 / 0 | 97.66% | 2.614 ms | 5.161 ms | 2.851 | 295,022 |

Seat-paired seed analysis (score = win 1, draw 0.5, loss 0) found a mean score change of **+0.0078** for depth-4: 4 seeds improved, 58 tied, and 2 regressed. A seed-cluster bootstrap 95% interval was **[-0.0117, +0.0273]**, which includes zero. Seven of 128 seat games changed outcome. The deeper variant used about 1.88× the mean decision time and 2.25× the transition evaluations.

This is not enough evidence to replace the frozen depth-3/56 viewer candidate. Keep the cheaper configuration as the current leader; continue depth changes only with a stronger adaptive hypothesis. Full reports, game traces, replay schemas, hashes, and package manifests are stored under `baseline-d3-e56/` and `depth4-e112/`.
