#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)), '..');
const source = resolve(root, 'apps/viewer/search/territory.rs');
const temp = await mkdtemp(resolve(tmpdir(), 'circuit-search-bench-'));
const wasmPath = resolve(temp, 'planner.wasm');
const compiled = spawnSync('rustc', [
  '--crate-type', 'cdylib', '--target', 'wasm32-unknown-unknown',
  '--edition=2021', '-C', 'opt-level=z', '-C', 'lto', '-C', 'codegen-units=1',
  '-C', 'strip=symbols', source, '-o', wasmPath,
], { cwd: root, encoding: 'utf8' });
if (compiled.status !== 0) throw new Error(compiled.stderr || compiled.stdout || 'WASM compile failed');

const wasmBytes = await readFile(wasmPath);
const { instance } = await WebAssembly.instantiate(wasmBytes);
const e = instance.exports;
const input = Number(e.circuit_input_ptr());
const inputLen = Number(e.circuit_input_len());
const output = Number(e.circuit_output_ptr());
assert.equal(inputLen, 46);

let rng = 0x6d2b79f5;
function randomInt(limit) {
  rng = (Math.imul(rng ^ (rng >>> 15), 1 | rng) + 0x9e3779b9) | 0;
  let value = Math.imul(rng ^ (rng >>> 7), 61 | rng) ^ rng;
  value ^= value >>> 14;
  return (value >>> 0) % limit;
}

const fixtures = [];
for (let index = 0; index < 40; index++) {
  const heads = [36 + (index % 5), 107 - (index % 7)];
  const occupied = new Set(heads);
  const occupiedCount = 18 + randomInt(38);
  while (occupied.size < occupiedCount) occupied.add(randomInt(144));
  fixtures.push({ heads, occupied: [...occupied], headings: [randomInt(4), randomInt(4)] });
}

function plan(fixture) {
  const bytes = new Uint8Array(e.memory.buffer, input, inputLen);
  bytes.fill(0);
  bytes[0] = 12; bytes[1] = 12;
  bytes[2] = fixture.heads[0]; bytes[3] = fixture.heads[1];
  bytes[4] = fixture.headings[0]; bytes[5] = fixture.headings[1];
  bytes[7] = 3; bytes[8] = 3;
  new DataView(e.memory.buffer).setUint32(input + 10, 56, true);
  for (const cell of fixture.occupied) bytes[14 + (cell >> 3)] |= 1 << (cell & 7);
  assert.equal(e.circuit_plan(), 0);
  const values = new Float64Array(e.memory.buffer, output, 15);
  const action = values[11] | 0;
  assert.ok(action === -1 || (values[9] & (1 << action)) !== 0,
    `planner returned illegal action ${action} with mask ${values[9]} for ${JSON.stringify(fixture)}`);
  return values[12];
}

try {
  for (const fixture of fixtures) for (let i = 0; i < 10; i++) plan(fixture);
  const samples = [];
  let completedDepthSum = 0;
  for (let repeat = 0; repeat < 50; repeat++) {
    for (const fixture of fixtures) {
      const start = process.hrtime.bigint();
      completedDepthSum += plan(fixture);
      samples.push(Number(process.hrtime.bigint() - start) / 1e6);
    }
  }
  samples.sort((a, b) => a - b);
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const percentile = (p) => samples[Math.ceil(p * samples.length) - 1];
  console.log(JSON.stringify({
    runtime: `Node ${process.version} V8 ${process.versions.v8}`,
    engine: 'WebAssembly on V8 (headless Node; not a browser timing)',
    board: '12x12', max_depth: 3, edge_budget: 56, fixtures: fixtures.length,
    measured_decisions: samples.length,
    mean_ms: mean, p50_ms: percentile(0.50), p95_ms: percentile(0.95),
    p99_ms: percentile(0.99), completed_depth_mean: completedDepthSum / samples.length,
  }, null, 2));
} finally {
  await rm(temp, { recursive: true, force: true });
}
