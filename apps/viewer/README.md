# Gridline — local arena viewer

A local-only browser spectator for the Light Cycles self-play project. The fixed-depth simultaneous-search planner compiled to Rust/WASM is selected by default; choose the same uploaded ONNX policy for both agents when you want another checkpoint. No Python inference service, hosted account, or remote request is needed. Run `npm ci && npm run dev` from this directory and open the printed local URL. `npm run build` creates a static bundle in `dist/`.

The viewer builds `search/territory.rs` into a size-optimized WASM module before `dev` and `build`. This requires `rustc` with the `wasm32-unknown-unknown` target. The current search mode uses depth 3 and a 56-edge budget on a 12×12 board; it runs the simultaneous search once per tick and samples each rider's equilibrium policy from the match's derived random stream. It does not require an ONNX file. The policy-value mode continues to require verified ONNX metadata and checkpoint bytes.

## Loading a trained model

1. Load the checkpoint `.metadata.json` via **Verify checkpoint metadata**. Valid metadata is mandatory before a model can be played.
2. Load the matching `.onnx` file. The viewer verifies the full metadata contract, digest, schemas, runtime/opset, architecture dimensions, ONNX/PyTorch artifact links, loaded ONNX SHA-256 and filename, interface, and tensor shape before creating an inference session. Fresh arenas use the declared checkpoint width and height.
3. Select a fresh seed (values 32 and above avoid the held-out evaluation range 0–31) and start. Both policies infer in-browser from the same pre-tick state.
4. Pause/resume, restart, scrub recorded frames and save a completed match as `lc-replay/1` JSON.

A metadata-less or mismatched ONNX file is rejected, but replay, evaluation and run-manifest inspection work without loading a model. A run-manifest import verifies its own digest/schema labels/safe paths only; it does not validate referenced files.

A model must accept float32 `observation` shaped `[batch, height, width, 12]` and expose four action logits (the optional value output is ignored). The policy is deterministic argmax in `UP, RIGHT, DOWN, LEFT` order. The browser engine owns game transitions; visual effects never affect state.

## Inspecting evidence

Use **Load run folder** with Chromium/WebKit folder selection to validate every manifest-listed file SHA-256, checkpoint metadata/model links, evaluation digest/identity, replay digest/seed/engine transitions and training-metrics summary. Only then is the bundle marked verified; select an included verified replay from the archive. The bundle displays optimizer updates/loss, environment steps per second, chosen device and recorded CUDA optimizer updates alongside evaluation win rate, W/D/L and mean survival. A run-manifest-only import checks only its integrity/schema/path fields. Or load an `lc-eval/1` report individually to view its win rate, W/D/L, mean survival and held-out report details. Load an `lc-replay/1` file to validate its canonical digest and rebuild the recorded states through the TypeScript engine before showing the timeline. Unsupported schema, digest mismatch, or transition mismatch fails closed with a visible toast.

For fully offline deployment, serve the generated `dist/` folder locally (e.g. `npm run preview`). Dependencies, including ONNX WASM assets, are bundled by Vite; the UI itself uses system fonts and makes no third-party requests.
