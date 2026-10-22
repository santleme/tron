# Fixed-depth simultaneous-search WASM prototype

This Rust `cdylib` implements the compact fixed-depth simultaneous planner and territory evaluator. The viewer exposes it as the **Fixed-depth search · Rust/WASM** agent mode; the existing ONNX Policy-Value mode remains available.

Build and check it from the repository root:

```sh
cd apps/viewer
npm ci
npm run build
cd ../..
node --experimental-strip-types apps/viewer/scripts/search-wasm-smoke.mjs
rustc --test --edition=2021 apps/viewer/search/territory.rs -o /tmp/circuit-search-tests
/tmp/circuit-search-tests
cd apps/viewer && npm run bench:search-wasm
```

The viewer build runs `scripts/build-search-wasm.mjs`, which requires `rustc`
with `wasm32-unknown-unknown` installed. It uses size-optimized LTO; the
current production module is about 33 KB before HTTP compression. Search mode
uses depth 3 and a 56-edge decision budget on a 12×12 board. The mixed
equilibrium policy is sampled for each agent using the match's derived random
stream.

The C ABI exports `circuit_input_ptr`, `circuit_input_len`, `circuit_output_ptr`, `circuit_output_len`, and `circuit_plan`. JS writes a 46-byte packet into the input buffer:

| Bytes | Meaning |
| --- | --- |
| 0..1 | width, height (each at least 2; at most 256 cells total) |
| 2..3 | player 0 and 1 row-major head indices |
| 4..5 | headings (`0=up, 1=right, 2=down, 3=left`) |
| 6 | result (`0=ongoing, 1=player 0 win, 2=player 1 win, 3=draw`) |
| 7 | alive flags: bit 0 player 0, bit 1 player 1 |
| 8..9 | maximum iterative-deepening depth, reserved byte |
| 10..13 | edge budget, unsigned 32-bit little-endian |
| 14..45 | 256 occupancy bits, least-significant bit first |

The output has 15 `f64` slots: value from player 0's view; four policy probabilities for each player; both legal-action masks; a legal deterministic fallback action for player 0 (lowest direction wins ties); completed depth; transition-edge count; and leaf-evaluation count. If the budget is too small to finish depth one, the result uses a uniform distribution over legal root moves and reports depth zero.

Transitions resolve both proposals together. Per-player legal masks exclude only individual invalid moves (reverse, wall, occupied cell); same-destination and head-swap interactions remain joint-transition properties. The smoke harness compares planner policies, masks, values, search depth, and work counts with Python search over 53 deterministic states. Its parity-only build also exposes a transition projection, checked against the canonical TypeScript `step` over all joint direction/INVALID proposals on those states and terminal fixtures. Production builds omit this test export.

The viewer search mode supports boards of at most 256 cells. Planner strength
was selected from fresh CPU-only Kaggle tournaments. The trained Policy-Value
checkpoint remains an experimental comparison artifact; on the 512-game
cost-matched comparison, fixed-depth scored 0.919 (95% seed-cluster bootstrap
interval 0.898–0.938) against that standalone policy.

`npm run bench:search-wasm` measures 2,000 warm WASM decisions on deterministic
12×12 positions and reports mean, p50, p95, p99, and completed depth. It uses
Node's V8 WebAssembly runtime in a headless process, not a browser timing; use
the Kaggle native tournament report for native CPU timings.
