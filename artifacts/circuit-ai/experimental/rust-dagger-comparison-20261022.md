# Rust fixed-depth versus selected spatial checkpoints

## Scope and protocol

This is an EXPERIMENTAL development report. Kaggle tournaments were CPU-only, with internet and GPU disabled. Each decisive 12×12 matchup used 256 new seeds and both seat assignments (512 games). Robustness runs used 64 seeds per opponent and both seats (128 games per matchup). Outcomes use a seed-cluster bootstrap interval, keeping both seat assignments together. All inputs, outputs, code provenance hashes, seed ranges, and runtime details are indexed in [campaign-manifest.json](rust-search-campaigns-20261022/campaign-manifest.json).

No reserved final set or `open-space` opponent was accessed for selection. The official registry and other worktrees were not modified. This report does not claim official evaluation or final approval.

## Main result

On 12×12, the Rust fixed-depth planner beat the latest selected DAgger1 checkpoint by **478 wins, 25 draws, and 9 losses in 512 games**. The score (draws count as half) was **0.9580**, with a 95% paired seed-cluster bootstrap interval of **[0.9434, 0.9717]**. Both seats were balanced: seat 0 was 238/13/5 and seat 1 was 240/12/4 (W/D/L).

The checkpoint was `dagger1-selected-20261021`, SHA-256 `f01978c5ba9edd4f7d544a91115f3739d22c543731d90d3cdc4ea0d96f9a96af`, staged read-only from the selected experimental artifact. The tournament used the checkpoint's legal-action safety mask. Rust's mean decision latency was **1.437 ms**, p95 **2.950 ms**; DAgger1 was **2.209 ms** mean, p95 **2.774 ms**. Rust was faster on average, while its p95 was about 6% higher, so this is strong competitive evidence but not a strict equal-latency comparison. The Rust planner completed a mean depth of **2.070** under max depth 3 and edge budget 56.

Against the previous selected spatial checkpoint, Rust scored **486/17/9 W/D/L** in 512 games (score **0.9658**, interval **[0.9521, 0.9785]**). Rust latency was **1.476 ms mean / 2.945 ms p95**, versus **2.258 / 2.540 ms** for the spatial policy. Mean completed depth was 2.073.

All 18 losses across those two decisive checkpoint comparisons ended with Rust receiving `invalid_action`; the replay audit showed these were no-legal-move terminal states, rather than an illegal move that discarded an available escape. That distinction is important: the reported `invalid_action` cause is the engine's terminal representation when no action exists. The top-level report still records avoidable deaths independently.

## Board-size robustness

| Arena | Opponent | W/D/L | Score | 95% seed-cluster interval | Rust mean / p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 8×8 | cruise | 118/7/3 | 0.9492 | [0.9141, 0.9805] | 0.703 / 1.518 ms |
| 8×8 | local-mobility | 107/11/10 | 0.8789 | [0.8320, 0.9219] | 0.733 / 1.552 ms |
| 8×8 | interceptor | 98/23/7 | 0.8555 | [0.8125, 0.8984] | 0.711 / 1.547 ms |
| 16×16 | cruise | 126/2/0 | 0.9922 | [0.9805, 1.0000] | 2.088 / 4.177 ms |
| 16×16 | local-mobility | 123/4/1 | 0.9766 | [0.9531, 0.9961] | 2.309 / 4.254 ms |
| 16×16 | interceptor | 105/22/1 | 0.9063 | [0.8711, 0.9414] | 2.270 / 4.222 ms |

These are robustness screens, not decisive 512-game blocks. Interceptor was the strongest of the three development heuristics at both sizes and is the clearest source of draws and avoidable collision pressure. The 16×16 run also confirms native search accepts the full 256-cell board bound. Kaggle completed the 8×8 block in 1.751 s and 16×16 in 14.035 s wall time on four Xeon CPU workers.

## Pilot and previous planner evidence

The corrected 8-seed DAgger pilot produced 15/1/0 for both Python fixed-depth and native Rust against DAgger1. Python mean search time was 91.53 ms; native Rust was 1.328 ms mean / 2.662 ms p95. It served only as a packaging/runtime check. Two earlier pilot versions failed while loading the `state_dict` key and were not counted as results.

The prior report [search-neural-comparison-20261022.md](search-neural-comparison-20261022.md) contains the already completed comparisons against DUCB, Regret Matching, heuristics, the earlier CNN, and standalone Policy-Value. Those experiments are intentionally not repeated here. In particular, neural training is not justified by these new results.

## Search changes and measured runtime

The planner uses simultaneous joint transitions and solves the resulting zero-sum action matrices. Its existing search uses iterative deepening, a configurable depth and edge budget, exact state/depth transpositions, territorial flood-fill/Voronoi and articulation features, and early stopping once territories separate. This pass removed a write-only visited-set path, reused flood-region results for leaf scoring, and moved transposition hits ahead of flood-fill work. The changes preserve transition semantics; parity still matches the canonical TypeScript engine.

The final bundled WASM was **33,067 bytes**. On 40 deterministic 12×12 fixtures and 2,000 decisions, Chromium Headless 156 measured **0.493 ms mean, 0.5 ms median, 0.9 ms p95, and 1.0 ms p99** (depth 3, edge budget 56; average completed depth 2.1). The separate Node/V8 benchmark measured 0.500 ms mean and 0.917 ms p95. These are local runtime timings, not Kaggle timings. Kaggle native Rust at 12×12 measured roughly 1.44–1.48 ms mean in the decisive tournaments.

The HTML viewer now defaults to Rust/WASM and keeps Policy-Value ONNX selectable. A real Chromium smoke match with seed `8500000000000002` ended in CYAN victory after 45 ticks; its downloaded `lc-replay/1` replay reloaded as `REPLAY READY`, and the browser reported no page errors. The viewer's canonical TypeScript engine continues to resolve every game transition.

## DAgger1 hybrid decision

DAgger1 is a four-logit policy network, not a policy-value model: it has no scalar value output. The search leaf evaluator requires a state value, while the DAgger observation also distinguishes each rider's trail planes; the compact search state stores only combined occupancy. Plugging logits in as a value or biasing/pruning simultaneous actions would therefore be an ungrounded change to the matrix game. A policy-rollout evaluator would require preserving per-rider trail identity through compact transitions and validating rollout calibration; that has not been implemented or claimed as a tested gain. The network was consequently not inserted into the planner, and no hybrid is selected. The direct controlled checkpoint comparisons above supply the stronger current evidence.

## Frozen experimental candidate

The selected artifact is [fixed-depth-candidate-20261022/candidate.json](fixed-depth-candidate-20261022/candidate.json), with Rust source and WASM SHA-256 values pinned there. It is frozen as an **experimental, unsubmitted** candidate for a future official evaluation. No official registration or reserved evaluation occurred in this worktree.

## Verification

- Native Rust unit tests: 4 passed.
- Python tournament/Kaggle preparation/download/analysis tests: 26 passed.
- WASM-vs-Rust parity: 53 deterministic states; canonical TypeScript transition parity: 1,375 joint transitions.
- Viewer typecheck and tests: 15 passed; production build passed.
- Chromium: WASM benchmark, complete match, replay export/import, and zero page errors.
- `git diff --check`: passed.
