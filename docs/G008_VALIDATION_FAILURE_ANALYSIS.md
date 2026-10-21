# G008 v2 validation failure analysis

**Date:** 2026-10-21  
**Scope:** existing registered train/validation evidence only. No Kaggle calls, training, replay simulation, or analysis of the consumed final roster was performed.

## Evidence binding

The six reports below are the per-game validation outputs for `g008-competitive-20261021-v2`. Each report's SHA-256 and checkpoint SHA-256 match the corresponding `competitive_validation_result_v2` event in `artifacts/circuit-ai/registry/g008-experiment-registry.jsonl`. Each contains 512 episodes against `g007-step-300` and records `mode=validation`.

| Pair | Arm | Registry event | Retained report | Report SHA-256 |
|---|---|---:|---|---|
| 1 | baseline | 143 | `history/evidence/g008-recovery/row62-a6/arena-output/c19ca811-b968-562f-a813-6f249547109d/g008-evaluation.json` | `ea8f93e275dca2167feece59d712cab4a9140b97f6172f53f562d155c8ddd438` |
| 1 | mixed | 148 | `history/evidence/g008-recovery/row63-a1/arena-output/c0a15915-5b4e-5b2f-b148-392b8dbaee42/g008-evaluation.json` | `8059115044fb3fbdbaba2f8fc7f8d5029aaace185b4d678eb75ca44ade3e823a` |
| 2 | baseline | 153 | `history/evidence/g008-recovery/row64-a1/arena-output/d5fecabc-d830-5efa-8fa9-ffa0d8918f07/g008-evaluation.json` | `f57b6ec67c00968db346384ca45d4611e7dd4f9424a8a7b9d40b69753636a66d` |
| 2 | mixed | 158 | `history/evidence/g008-recovery/row65-a1/arena-output/8ab2a371-15ea-51c1-b6ad-d523ea941e63/g008-evaluation.json` | `fbfaae8ae31660bf1da0338f3be4950ca627f284d22fdcdfd1deb00b537663e0` |
| 3 | baseline | 163 | `history/evidence/g008-recovery/row66-a1/arena-output/3c62c4e7-feeb-52fc-9e70-92962f1e6502/g008-evaluation.json` | `1213b1f49f78a5a2300603d124f1d4eca8fe8d1b627cb506fcf11359751ef1ff` |
| 3 | mixed | 168 | `history/evidence/g008-recovery/row67-a1/arena-output/729d0478-dfed-5a76-9d2e-7d3e633523e4/g008-evaluation.json` | `c22b229c6086fdb58e0ee1c99784ca2cff043f2126583469ce415c6e1b1341ef` |

Paths are relative to `artifacts/circuit-ai/`. The registry is authoritative for complete event and output-manifest hashes.

## Paired validation outcomes

The campaign's paired comparison uses score `(wins + 0.5 * draws) / games`; this is not observed win rate and is not the final promotion criterion. The registered paired deltas for mixed minus baseline were `-0.008789`, `-0.052734`, and `-0.050781` (mean `-0.037435`, 95% interval `[-0.099109, 0.024239]`). All three pairs regressed.

| Pair | Arm | Wins | Draws | Losses | Observed win rate | Score incl. half draws | Locally avoidable terminal losses |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 276 | 42 | 194 | 53.91% | 58.01% | 184 / 194 (94.8%) |
| 1 | mixed | 264 | 57 | 191 | 51.56% | 57.13% | 183 / 191 (95.8%) |
| 2 | baseline | 297 | 33 | 182 | 58.01% | 61.23% | 175 / 182 (96.2%) |
| 2 | mixed | 267 | 39 | 206 | 52.15% | 55.96% | 199 / 206 (96.6%) |
| 3 | baseline | 304 | 37 | 171 | 59.38% | 62.99% | 166 / 171 (97.1%) |
| 3 | mixed | 275 | 43 | 194 | 53.71% | 57.91% | 177 / 194 (91.2%) |

The frozen candidate was the pair-2 baseline checkpoint. Its 61.23% validation score includes half-credit for draws; its observed win rate on that validation roster was 58.01%. Neither number is evidence of meeting the independent final requirement.

## Failure signals in the six reports

- Across 1,138 validation losses, 1,084 (95.3%) were classified as `avoidable_decision`: the report found at least one locally safe action at the terminal decision when the observed opponent action is held fixed.
- The other loss categories were 51 `no_safe_move_trapped`, 2 `collision_without_safe_alternative`, and 1 `wall`.
- Candidate-side terminal causes were `occupied_trail` in 957 / 1,138 losses (84.1%), `occupied_head` in 165 (14.5%), and `out_of_bounds` in 16 (1.4%).
- Loss durations had median terminal ticks of 12–14 across the six reports. Losses were balanced enough across seats that these aggregates do not point to one fixed seat as the main cause.

These are useful signals for action safety and space preservation. They are **not causal proof**: the counterfactual holds the opponent's actual action fixed for one step, so a different candidate action could change later opponent behavior and the game trajectory. They also describe validation against `g007-step-300`, not the protected `open-space` final roster.

## Next-cycle implication

Do not repeat the tested mixed schedule without a new, justified change. In a newly authorized cycle, keep the baseline and test a narrowly specified change to safe-action selection or space preservation using paired training roots and validation only. Predeclare whether the target is fewer locally avoidable terminal actions, fewer `occupied_trail` losses, or stronger win outcomes; require multiple validation seeds and do not infer promotion from these diagnostics. Keep the final roster fresh, hidden, and unused until one candidate is frozen. The current campaign is fully consumed, and this analysis does not authorize a new workload.
