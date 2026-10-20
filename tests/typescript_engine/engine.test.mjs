import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';
import { ACTIONS, Pcg32, canonicalJson, chooseAction, createSeededState, deriveSeed, digestObject, observe, replayActions, step } from '../../apps/viewer/engine/index.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const fixture = async (name) => JSON.parse(await readFile(resolve(root, 'shared/fixtures', name), 'utf8'));

test('action order and deterministic argmax select lowest index on ties', () => {
  assert.deepEqual(ACTIONS, ['UP', 'RIGHT', 'DOWN', 'LEFT']);
  assert.equal(chooseAction([2, 7, 7, 1]), 1);
  assert.throws(() => chooseAction([1, Infinity, 0, 0]), /four finite/);
});

test('SHA-256 domain separated PCG streams match all published seed vectors', async () => {
  const vectors = (await fixture('seed-vectors-1.json')).vectors;
  for (const vector of vectors) {
    const seed = await deriveSeed(vector.master_seed, vector.label, vector.indices);
    assert.equal(seed.initstate.toString(), vector.initstate, vector.label);
    assert.equal(seed.initseq.toString(), vector.initseq, vector.label);
    const rng = new Pcg32(seed);
    assert.deepEqual(vector.pcg32, vector.pcg32.map(() => rng.nextUint32()));
  }
});

test('seeded initialization derives distinct positions and independent headings reproducibly', async () => {
  const a = await createSeededState({ width: 8, height: 6, masterSeed: '0', arenaIndex: 0 });
  const b = await createSeededState({ width: 8, height: 6, masterSeed: '0', arenaIndex: 0 });
  assert.deepEqual(a, b);
  assert.notDeepEqual(a.state.agents[0].position, a.state.agents[1].position);
  assert.deepEqual([observe(a.state, 0), observe(a.state, 1)], [observe(b.state, 0), observe(b.state, 1)]);
  await assert.rejects(createSeededState({ width: 1, height: 1, masterSeed: '0', arenaIndex: 0 }), /at least 2/);
});

test('seat-relative observation is row-major 12-channel [height][width] tensor', () => {
  const state = { width: 5, height: 4, tick: 0, result: null, agents: [
    { position: { x: 1, y: 1 }, heading: 1, trail: [{ x: 2, y: 1 }, { x: 1, y: 1 }], dead: false },
    { position: { x: 3, y: 2 }, heading: 3, trail: [{ x: 3, y: 2 }], dead: false },
  ] };
  const obs = observe(state, 0);
  assert.equal(obs.length, 4); assert.equal(obs[0].length, 5); assert.equal(obs[0][0].length, 12);
  assert.equal(obs[1][1][0], 1); assert.equal(obs[1][1][2], 1); assert.equal(obs[1][2][2], 1); assert.equal(obs[2][3][1], 1);
  assert.equal(obs[1][1][5], 1); assert.equal(obs[2][3][11], 1);
  const reverse = observe(state, 1);
  assert.equal(reverse[1][1][1], 1); assert.equal(reverse[1][2][3], 1); assert.equal(reverse[1][1][9], 1);
});

test('simultaneous transitions match every shared collision vector', async () => {
  const data = await fixture('collision-transitions-1.json');
  for (const vector of data.cases) {
    const positions = vector.positions.map(([x, y]) => ({ x, y }));
    const headings = vector.headings.map(name => ACTIONS.indexOf(name));
    const trails = positions.map(p => [{ ...p }]);
    for (const [x, y] of vector.occupied_cells ?? []) trails[0].unshift({ x, y });
    const state = { width: vector.width, height: vector.height, tick: 0, result: null, agents: positions.map((position, i) => ({ position, heading: headings[i], trail: trails[i], dead: false })) };
    const { state: next, record } = step(state, vector.actions);
    assert.deepEqual(record.collision_causes, vector.expected_causes, vector.id);
    assert.equal(next.result, vector.expected_result, vector.id);
  }
});

test('terminal step does not advance or change the match', () => {
  const state = { width: 5, height: 5, tick: 1, result: 'agent_1_win', agents: [
    { position: { x: 0, y: 0 }, heading: 0, trail: [{ x: 0, y: 0 }], dead: true },
    { position: { x: 4, y: 4 }, heading: 2, trail: [{ x: 4, y: 4 }], dead: false },
  ] };
  const transition = step(state, [0, 2]);
  assert.equal(transition.state, state);
  assert.equal(transition.state.tick, 1);
  assert.deepEqual(transition.record.collision_causes, [[], []]);
  assert.throws(() => replayActions(state, [[0, 2]]), /after the terminal result/);
});

test('seeded initialization, observations and multi-tick transitions match Python reference when available', async (t) => {
  const probe = spawnSync('python3', ['--version'], { encoding: 'utf8' });
  if (probe.error || probe.status !== 0) return t.skip('python3 unavailable');
  const actions = [[1, 3], [0, 2], [1, 0], [2, 3], [0, 1]];
  const script = String.raw`
import json, sys
from training.arena.engine import observe, step
from training.arena.seeds import PCG32, stream_set
from training.arena.engine import initial_state
payload=json.load(sys.stdin)
width,height,seed,arena=payload['width'],payload['height'],int(payload['seed']),payload['arena']
streams=stream_set(seed,arena)
state=initial_state(width,height,PCG32.from_seed(streams[0]),tuple(PCG32.from_seed(s) for s in streams[1:]))
def serialize(s):
 return {'width':s.width,'height':s.height,'tick':s.tick,'result':s.result,'agents':[{'position':{'x':a.position[0],'y':a.position[1]},'heading':a.heading,'trail':[{'x':x,'y':y} for x,y in a.trail],'dead':not a.alive} for a in s.agents]}
out={'state':serialize(state),'observations':[observe(state,0),observe(state,1)],'steps':[]}
for joint in payload['actions']:
 tr=step(state,joint); state=tr.state
 out['steps'].append({'state':serialize(state),'causes':[list(c) for c in tr.causes], 'observations':[observe(state,0),observe(state,1)]})
print(json.dumps(out,separators=(',',':')))
`;
  const width = 9, height = 7, seed = '18446744073709551615', arena = 4;
  const result = spawnSync('python3', ['-c', script], { cwd: root, input: JSON.stringify({ width, height, seed, arena, actions }), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const expected = JSON.parse(result.stdout);
  const seeded = await createSeededState({ width, height, masterSeed: seed, arenaIndex: arena });
  assert.deepEqual(seeded.state, expected.state);
  assert.deepEqual([observe(seeded.state, 0), observe(seeded.state, 1)], expected.observations);
  let state = seeded.state;
  for (const [index, joint] of actions.entries()) {
    if (state.result !== null) break;
    const transition = step(state, joint);
    assert.deepEqual(transition.state, expected.steps[index].state, `state after tick ${index}`);
    assert.deepEqual(transition.record.collision_causes, expected.steps[index].causes, `causes after tick ${index}`);
    assert.deepEqual([observe(transition.state, 0), observe(transition.state, 1)], expected.steps[index].observations, `observations after tick ${index}`);
    state = transition.state;
  }
});

test('valid replay fixture digest and canonical JSON contract are reproducible', async () => {
  const replay = await fixture('lc-replay-1.valid.json');
  assert.equal(await digestObject(replay, 'replay_sha256'), replay.replay_sha256);
  const initial = replay.initial_state;
  const state = { width: initial.width, height: initial.height, tick: 0, result: null, agents: initial.positions.map((position, i) => ({ position, heading: ACTIONS.indexOf(initial.headings[i]), trail: initial.trails[i], dead: false })) };
  const rebuilt = replayActions(state, replay.ticks.map(tick => tick.joint_actions));
  assert.deepEqual(rebuilt.records.map(record => record.collision_causes), replay.ticks.map(tick => tick.collision_causes));
  assert.deepEqual(rebuilt.records.map(record => record.terminal_causes), replay.ticks.map(tick => tick.terminal_causes));
  assert.equal(rebuilt.result, replay.result);
  assert.equal(rebuilt.records.length, replay.tick_count);
  const regeneratedReplay = { ...replay, ticks: rebuilt.records };
  assert.equal(await digestObject(regeneratedReplay, 'replay_sha256'), replay.replay_sha256);
  const canonicalFixture = await fixture('canonical-digest-1.json');
  for (const vector of canonicalFixture.vectors) {
    assert.equal(canonicalJson(vector.document), vector.canonical_utf8);
    assert.equal(await digestObject({ ...vector.document, sha256: vector.sha256 }, 'sha256'), vector.sha256);
  }
});
