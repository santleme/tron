# Independent open-space assessment — Rust fixed-depth

- Run: `ind-open-space-d69feb58a3564faea462`
- Kaggle kernel: `pascalcoimbra/tron-independent-open-space`, version `2`
- Status: complete; **nonofficial independent assessment**, not a G008 result
- Seed plan: 256 new seeds, both seats per seed, 512 games total
- Seed-plan SHA-256: `da21e3504455e5039bbf90000acd187f97ea271379466fb4ffb071a6f3f8d709`
- Candidate: `fixed-depth-rust-wasm-20261022`
- Rust source SHA-256: `4ebf0b4c8518bb05e76e74180d42ccae00e4c1ae19e542908af9ad1e8f6b13b2`
- WASM SHA-256: `e601cbf991681c747c3b87a6d1fbc683ae3a2d7b9366a2567e5128917da4ce0a`
- Opponent: canonical `open-space` action implementation, pinned by the package source manifest

## Result

| Measure | Result |
| --- | ---: |
| Wins / draws / losses | **473 / 26 / 13** |
| Observed win rate | **92.38%** |
| Wilson 95% interval for win rate | **89.76%–94.38%** |
| Draw-half score rate | 94.92% |
| Seat 0 W/D/L | 237 / 12 / 7 |
| Seat 1 W/D/L | 236 / 14 / 6 |
| Mean candidate decision time | 0.743 ms |
| p95 candidate decision time | 1.906 ms |
| Mean completed search depth | 2.286 |

The candidate stayed frozen during the assessment. The result is kept outside the OFFICIAL registry and its reserved evaluation set. It was not used to tune this candidate. G008's final evaluation had already been consumed, so this fresh seed set is evidence for an independent nonofficial assessment, not a replacement official G008 result.

## Runtime and validation

The run used Kaggle CPU only (4 worker processes), with GPU and internet disabled. Kaggle reported Python 3.13.15 on Linux `6.18.48+-x86_64-with-glibc2.39`; elapsed runtime was 4.183 seconds. The exact native Rust library was compiled from the pinned Rust source, and its hash is recorded in `report.json` and the source manifest.

The downloaded package validator checked the complete 512-game paired roster, W/D/L and Wilson calculations, every output hash, all 512 `lc-replay/1` schemas and digests, and reconstructed every replay through the canonical engine. The seed plan and source manifest are included beside the report and replays.
