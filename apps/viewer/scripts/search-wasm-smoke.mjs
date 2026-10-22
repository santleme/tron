#!/usr/bin/env node
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { step } from '../engine/engine.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const rust = resolve(root, 'apps/viewer/search/territory.rs');
const temp = await mkdtemp(resolve(tmpdir(), 'circuit-search-wasm-'));
const wasmPath = resolve(temp, 'planner.wasm');

const fixtures = [
  { name: 'open-4x4', width: 4, height: 4, heads: [5, 10], headings: [1, 3], occupied: [5, 10], depth: 2, budget: 512 },
  { name: 'obstacles-5x4', width: 5, height: 4, heads: [6, 13], headings: [1, 3], occupied: [1, 6, 8, 13, 18], depth: 2, budget: 512 },
  { name: 'separated-5x3', width: 5, height: 3, heads: [6, 8], headings: [1, 3], occupied: [2, 6, 7, 8, 12], depth: 2, budget: 512 },
  { name: 'trapped-player-0', width: 5, height: 4, heads: [6, 18], headings: [1, 3], occupied: [1, 5, 6, 7, 11, 18], depth: 1, budget: 64 },
  { name: 'insufficient-root-budget', width: 4, height: 3, heads: [1, 7], headings: [3, 1], occupied: [1, 7], depth: 1, budget: 1 },
];

let randomState = 0x6d2b79f5;
function randomInt(limit) {
  randomState = (Math.imul(randomState ^ (randomState >>> 15), 1 | randomState) + 0x9e3779b9) | 0;
  let value = Math.imul(randomState ^ (randomState >>> 7), 61 | randomState) ^ randomState;
  value = value ^ (value >>> 14);
  return (value >>> 0) % limit;
}
for (let index = 0; index < 48; index++) {
  const width = 4 + randomInt(3);
  const height = 4 + randomInt(3);
  const size = width * height;
  const heads = [randomInt(size), randomInt(size)];
  while (heads[1] === heads[0]) heads[1] = randomInt(size);
  const occupied = [...heads];
  for (let cell = 0; cell < size; cell++) {
    if (!heads.includes(cell) && randomInt(4) === 0) occupied.push(cell);
  }
  fixtures.push({
    name: `generated-${String(index).padStart(2, '0')}`,
    width, height, heads, occupied,
    headings: [randomInt(4), randomInt(4)],
    depth: 1 + randomInt(3), budget: 96,
  });
}

const py = String.raw`
import json, sys
from training.search.bitboard import CompactState, action_mask
from training.search.planners import fixed_depth_pair
rows = json.load(sys.stdin)
out = []
for f in rows:
    occupied = sum(1 << cell for cell in f['occupied'])
    s = CompactState(f['width'], f['height'], tuple(f['heads']), tuple(f['headings']), occupied)
    result0, result1 = fixed_depth_pair(
        s, max_depth=f['depth'], edge_budget=f['budget'], seed0=0, seed1=0)
    out.append({'masks': [action_mask(s, 0), action_mask(s, 1)], 'value': result0.value,
                'policies': [result0.policy, result1.policy],
                'depth': result0.completed_depth, 'edges': result0.transition_edges,
                'leaves': result0.leaf_evaluations})
print(json.dumps(out))
`;

function assertCommand(command, args, description) {
  const result = spawnSync(command, args, { cwd: root, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${description} failed:\n${result.stderr || result.stdout}`);
  return result.stdout;
}

try {
  assertCommand('rustc', ['--crate-type', 'cdylib', '--target', 'wasm32-unknown-unknown', '--edition=2021', '--cfg', 'circuit_parity', '--check-cfg=cfg(circuit_parity)', '-O', rust, '-o', wasmPath], 'WASM build');
  const oracleRun = spawnSync('python3', ['-c', py], { cwd: root, encoding: 'utf8', input: JSON.stringify(fixtures) });
  if (oracleRun.status !== 0) throw new Error(`Python oracle failed:\n${oracleRun.stderr || oracleRun.stdout}`);
  const oracle = JSON.parse(oracleRun.stdout);
  const { instance } = await WebAssembly.instantiate(await readFile(wasmPath));
  const e = instance.exports;
  const input = Number(e.circuit_input_ptr());
  const output = Number(e.circuit_output_ptr());
  const transitionOutput = Number(e.circuit_transition_ptr());
  const inputLen = Number(e.circuit_input_len());
  assert.equal(inputLen, 46, 'ABI input packet length');
  assert.equal(e.circuit_transition_len(), 14, 'transition parity output length');

  fixtures.forEach((f, index) => {
    const bytes = new Uint8Array(e.memory.buffer, input, inputLen);
    bytes.fill(0);
    bytes[0] = f.width; bytes[1] = f.height;
    bytes[2] = f.heads[0]; bytes[3] = f.heads[1];
    bytes[4] = f.headings[0]; bytes[5] = f.headings[1];
    bytes[6] = 0; bytes[7] = 3; bytes[8] = f.depth;
    new DataView(e.memory.buffer).setUint32(input + 10, f.budget, true);
    for (const cell of f.occupied) bytes[14 + (cell >> 3)] |= 1 << (cell & 7);
    assert.equal(e.circuit_plan(), 0, `${f.name}: planner ABI status`);
    const values = new Float64Array(e.memory.buffer, output, 15);
    const mask0 = values[9] | 0; const mask1 = values[10] | 0;
    assert.deepEqual([mask0, mask1], oracle[index].masks, `${f.name}: Python action-mask parity`);
    assert.ok(Math.abs(values[0] - oracle[index].value) < 1e-7,
      `${f.name}: value mismatch WASM=${values[0]} Python=${oracle[index].value}`);
    const action = values[11] | 0;
    assert.ok(action === -1 || (mask0 & (1 << action)) !== 0, `${f.name}: returned illegal action ${action}`);
    assert.equal(values[12], oracle[index].depth, `${f.name}: completed-depth parity`);
    assert.equal(values[13], oracle[index].edges, `${f.name}: transition-edge parity`);
    assert.equal(values[14], oracle[index].leaves, `${f.name}: leaf-evaluation parity`);
    for (let player = 0; player < 2; player++) {
      for (let direction = 0; direction < 4; direction++) {
        const outputIndex = 1 + player * 4 + direction;
        assert.ok(Math.abs(values[outputIndex] - oracle[index].policies[player][direction]) < 1e-7,
          `${f.name}: player ${player} policy ${direction} differs`);
      }
    }
    if (f.name === 'insufficient-root-budget') {
      assert.equal(values[12], 0, 'undersized budget must report no completed search depth');
      assert.equal(values[1 + action], 0.5, 'fallback policy should be uniform over player 0 legal moves');
    }
    if (index < 4 || f.name === 'insufficient-root-budget') console.log(`${f.name}: masks=${mask0}/${mask1} value=${values[0].toFixed(6)} depth=${values[12]} action=${action}`);
  });

  function arenaState(f) {
    const point = (cell) => ({ x: cell % f.width, y: Math.floor(cell / f.width) });
    const [head0, head1] = f.heads;
    const extraTrail = f.occupied.filter((cell) => cell !== head0 && cell !== head1).map(point);
    const result = f.result === 'draw' ? 'draw' : f.result === 'agent_0_win' ? 'agent_0_win' : f.result === 'agent_1_win' ? 'agent_1_win' : null;
    return {
      width: f.width, height: f.height, tick: 7, result,
      agents: [
        { position: point(head0), heading: f.headings[0], trail: [...extraTrail, point(head0)], dead: Boolean(f.dead?.[0]) },
        { position: point(head1), heading: f.headings[1], trail: [point(head1)], dead: Boolean(f.dead?.[1]) },
      ],
    };
  }
  function writeState(f, s) {
    const bytes = new Uint8Array(e.memory.buffer, input, inputLen);
    bytes.fill(0);
    bytes[0] = s.width; bytes[1] = s.height;
    for (let player = 0; player < 2; player++) {
      const agent = s.agents[player];
      bytes[2 + player] = agent.position.y * s.width + agent.position.x;
      bytes[4 + player] = agent.heading;
      if (!agent.dead) bytes[7] |= 1 << player;
      for (const cell of agent.trail) {
        const index = cell.y * s.width + cell.x;
        bytes[14 + (index >> 3)] |= 1 << (index & 7);
      }
    }
    bytes[6] = s.result === 'agent_0_win' ? 1 : s.result === 'agent_1_win' ? 2 : s.result === 'draw' ? 3 : 0;
  }
  function occupiedCells(s) {
    const occupied = new Set();
    for (const agent of s.agents) for (const p of agent.trail) occupied.add(p.y * s.width + p.x);
    return occupied;
  }
  const transitionFixtures = fixtures.map((f) => ({ ...f, result: null, dead: null }));
  transitionFixtures.push(
    { name: 'already-terminal-draw', width: 4, height: 3, heads: [4, 7], headings: [1, 3], occupied: [4, 7], result: 'draw', dead: [true, true] },
    { name: 'already-terminal-win', width: 4, height: 3, heads: [4, 7], headings: [1, 3], occupied: [4, 7], result: 'agent_0_win', dead: [false, true] },
  );
  const jointActions = [0, 1, 2, 3, 4]; // 4 is the canonical engine's INVALID proposal.
  let transitionsChecked = 0;
  for (const f of transitionFixtures) {
    const initial = arenaState(f);
    for (const action0 of jointActions) for (const action1 of jointActions) {
      const actions = [action0 === 4 ? 'INVALID' : action0, action1 === 4 ? 'INVALID' : action1];
      const expected = step(initial, actions);
      writeState(f, initial);
      assert.equal(e.circuit_transition(action0, action1), 0, `${f.name}: Rust transition ABI status`);
      const actual = new Uint32Array(e.memory.buffer, transitionOutput, 14);
      const result = expected.state.result === null ? 0 : expected.state.result === 'agent_0_win' ? 1 : expected.state.result === 'agent_1_win' ? 2 : 3;
      assert.equal(actual[0], result, `${f.name} ${actions}: result parity`);
      assert.equal(actual[1], Number(!expected.state.agents[0].dead) | (Number(!expected.state.agents[1].dead) << 1), `${f.name} ${actions}: alive parity`);
      expected.state.agents.forEach((agent, player) => {
        assert.equal(actual[2 + player], agent.position.y * expected.state.width + agent.position.x, `${f.name} ${actions}: player ${player} head parity`);
        assert.equal(actual[4 + player], agent.heading, `${f.name} ${actions}: player ${player} heading parity`);
      });
      const actualOccupied = new Set();
      for (let cell = 0; cell < f.width * f.height; cell++) {
        const word = BigInt(actual[6 + Math.floor(cell / 64) * 2]) | (BigInt(actual[7 + Math.floor(cell / 64) * 2]) << 32n);
        if ((word & (1n << BigInt(cell % 64))) !== 0n) actualOccupied.add(cell);
      }
      assert.deepEqual([...actualOccupied].sort((a, b) => a - b), [...occupiedCells(expected.state)].sort((a, b) => a - b), `${f.name} ${actions}: occupancy parity`);
      transitionsChecked++;
    }
  }
  console.log(`WASM planner parity passed (${fixtures.length} deterministic states); canonical TS/Rust transition parity passed (${transitionsChecked} joint transitions).`);
} finally {
  await rm(temp, { recursive: true, force: true });
}
