# Light Cycles v1 contract profile

The JSON Schemas in this directory are the language-neutral wire contracts. They use JSON Schema Draft 2020-12 and reject unknown top-level/record fields unless a field is explicitly an extension object (`training_config`, `software`, `device`, `benchmark_records`, and metric `values`).

## Canonical JSON and digests

Digest-bearing v1 JSON uses UTF-8 canonical JSON with recursively sorted object keys by Unicode scalar/code-point order, array order preserved, no insignificant whitespace, NFC strings, and finite numbers only. Non-NFC input is rejected (not rewritten). Number spelling follows ECMAScript `JSON.stringify` number serialization (the RFC 8785/JCS number profile); strings are JSON-escaped per ECMAScript. Implementations MUST reject non-finite values, duplicate object keys, and non-NFC strings before hashing. Integer-valued JSON numbers MUST be in the interoperable range `[-9007199254740991, 9007199254740991]`; this avoids silently rounded JSON integers in JavaScript. Any unsigned 64-bit seed is represented on the wire as a canonical base-10 **string** (`"0"` through `"18446744073709551615"`), never as a JSON number. This is the sole integer representation exception needed for exact seed interoperability.

For `lc-run/1`, compute `manifest_sha256` over the canonical manifest after removing `manifest_sha256`; `files` MUST NOT list the manifest itself. For `lc-replay/1`, compute `replay_sha256` over the canonical replay after removing `replay_sha256`. For checkpoint metadata, compute `metadata_sha256` the same way. SHA-256 text is 64 lowercase hexadecimal characters. File `sha256` values hash exact file bytes, not parsed/canonical content.

## Seed derivation and PRNG

For each stream, hash these bytes in order:

```text
UTF8("lc-arena/seed/v1/" + stream_label) || 0x00 ||
LE64(master_seed) || LE64(index_0) || ... || LE64(index_n)
```

Labels are exactly `episode`, `agent/0`, and `agent/1`; episode streams include the arena index, agent streams include arena index then agent index. Indices are unsigned 64-bit little-endian values. Interpret digest bytes `[0:8]` and `[8:16]` as little-endian unsigned `initstate` and `initseq`. Seed PCG-XSH-RR 64/32 by setting state to zero, setting odd increment `(initseq << 1) | 1` with uint64 wraparound, advancing once, adding `initstate` with uint64 wraparound, then advancing once. Output uses multiplier `6364136223846793005`, uint64 wraparound, XSH-RR transform and 32-bit rotation. Seed strings are parsed exactly as unsigned 64-bit integers before derivation.

## Frozen v1 discriminators

`action_schema` is `lc-action/1`, with action order `UP`, `RIGHT`, `DOWN`, `LEFT` (indices 0..3). Observation is `lc-observation/1`: a row-major `[height][width][12]` integer tensor with channels 0..3 `[self_head, opponent_head, self_trail, opponent_trail]`, channels 4..7 one-hot self heading `[UP, RIGHT, DOWN, LEFT]`, and channels 8..11 one-hot opponent heading in the same order. Each cell value is 0 or 1; heading planes are all zero except at the corresponding current head cell and heading. Observation origin is top-left, x increases right, y increases down. Terminal results are `agent_0_win`, `agent_1_win`, or `draw`. Invalid replay actions are the exact token `INVALID`.

The model architecture and ONNX opset/runtime remain implementation choices and are intentionally not frozen by this contract; each checkpoint records its exact architecture/version and opset/runtime declaration.
