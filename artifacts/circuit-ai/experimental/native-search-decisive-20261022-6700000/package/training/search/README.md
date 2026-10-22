# Simultaneous search prototype

This package is a third, independent search approach for the canonical arena.
It is an exploratory implementation, not an approved/promoted agent. The
compact rules kernel consumes the same state contract as
`training.arena.engine`; it does not replace that canonical engine.

## Current implementation

- `bitboard.py` stores occupied cells as a Python integer bitset and resolves
  both actions from one immutable pre-tick state. Action masks handle
  reversal, walls, occupied heads and trails. Same-destination and head-swap
  collisions remain joint-action checks.
- `territory.py` implements bitset flood fill, connected components,
  head-to-head Voronoi distances, an iterative block-cut chamber tree, and a
  bounded antisymmetric territory evaluator. Separated heads are scored by
  reachable component size.
- `matrix.py` solves small zero-sum payoff matrices by vertex enumeration;
  this is intended for the arena's small action matrices, not general linear
  programming.
- `planners.py` contains iterative-deepening simultaneous matrix search,
  stacked Decoupled UCB1-Tuned MCTS, and stacked bandit Regret Matching MCTS.
  Random choices are seeded explicitly. Each API returns a legal action,
  four-action policy, value, logical transition count, leaf count, and search
  counters.

The local API can be imported with:

```python
from training.search import (
    decoupled_ucb1_tuned, fixed_depth, from_arena_state, regret_matching,
)
```

The canonical Python engine remains the rules authority. A fused Rust territory
evaluator is available to Kaggle CPU tournaments, while a separate full
fixed-depth Rust implementation runs in the viewer as WebAssembly. The compact
Policy-Value trainer exports viewer-compatible ONNX; the v2 model factory is
shared by training and isolated CPU inference. Tournament configurations pin
approved development checkpoint hashes and exclude official registries,
reserved splits, and the final-only `open-space` opponent.

Across two decisive Kaggle CPU planner campaigns, fixed-depth scored 0.932
(95% seed-cluster bootstrap interval 0.910–0.942) against DUCB1-Tuned and
0.899 (0.879–0.920) against Regret Matching, with 512 seat-balanced games per
matchup. In a separate 512-game campaign it scored 0.888 against the strongest
development heuristic, interceptor (the other heuristic results were higher).
The Policy-Value MLP did not improve on search: against depth-3 fixed-depth it
scored 0.073 at roughly one-tenth the decision latency; at matched latency,
depth-1 fixed-depth scored 0.919 (95% interval 0.898–0.938) in 512 games.
These are exploratory development comparisons, not the reserved final
open-space evaluation.

Analyze a validated output with seed-cluster bootstrap intervals and per-seat
latency summaries, collision causes, deaths avoidable by another legal move
against the observed opposing action, and survival after the territories
disconnect. A bounded Hoeffding interval is included alongside the bootstrap
interval so a perfect small sample does not report zero uncertainty:

```bash
python3 -m training.search.analysis artifacts/search-run \
  --output artifacts/search-analysis.json
```

## Correctness checks

The compact transition is checked against the repository's shared collision
fixtures and Python canonical engine. Matrix tests use known simultaneous
games, including matching pennies. Search tests check legal outputs,
seed-repeatability, bounded transition counts, and both player perspectives.
These checks establish code invariants only; they do not measure strength.

Run the focused local checks with:

```bash
python3 -m pytest -q tests/python_engine/test_simultaneous_search_kernel.py \
  tests/python_engine/test_simultaneous_planners.py tests/python_engine/test_engine.py
```

Search-strength claims must use new development seeds, both seat assignments,
and the validated output manifest. Do not use the final-only `open-space`
opponent or any reserved final set for selection or tuning. Existing official
workloads remain separate.

## Research references

- Lanctot et al., [Monte Carlo Tree Search for Simultaneous Move Games: A Case
  Study in the Game of Tron (2013)](https://dke.maastrichtuniversity.nl/m.winands/documents/sm-tron-bnaic2013.pdf).
  Their older, differently configured Tron experiments reported a 62.3%
  result for Decoupled UCB1-Tuned; that figure is a prior, not a prediction for
  this engine or its final opponent.
- Becker and Sunberg, [Simultaneous AlphaZero: Extending Tree Search to Markov
  Games (2025 preprint)](https://arxiv.org/abs/2512.12486). The newer arXiv
  version is dated 2026-08-13. Its regret/value transfer ideas motivate later
  experiments but have not established a Tron gain.
- Mahlau et al., [Mastering Zero-Shot Interactions in Cooperative and
  Competitive Simultaneous Games — Albatross (ICML 2024)](https://proceedings.mlr.press/v235/mahlau24a.html).
  Its bounded-rational opponent model and results on Battlesnake/Overcooked
  do not establish transfer to this deterministic zero-sum arena.
- Iouri Khramtsov, [TronBot](https://github.com/ikhramts/TronBot). Its
  tree-of-chambers and separation-aware evaluation motivate territorial
  features; its rules, implementation constraints and memory profile differ.
- DeepMind, [OpenSpiel](https://github.com/google-deepmind/open_spiel) and
  [algorithm catalogue](https://openspiel.readthedocs.io/en/latest/algorithms.html).
  OpenSpiel supplies simultaneous-game reference semantics and algorithms;
  the repository's Python engine remains the rules authority.
