import test from 'node:test';
import assert from 'node:assert/strict';
import { Pcg32 } from '../engine/prng.ts';
import { sampleLegalAction } from '../src/searchPolicy.ts';

const seed = { initstate: 123n, initseq: 456n };

test('samples only legal actions and repeats from the same seed', () => {
  const first = new Pcg32(seed);
  const second = new Pcg32(seed);
  const actions = Array.from({ length: 64 }, () => sampleLegalAction([0.5, 0.5, 0, 0], 0b0011, first));
  const repeated = Array.from({ length: 64 }, () => sampleLegalAction([0.5, 0.5, 0, 0], 0b0011, second));
  assert.deepEqual(actions, repeated);
  assert.deepEqual(new Set(actions), new Set([0, 1]));
});

test('renormalizes over legal actions and falls back safely for empty weights', () => {
  assert.equal(sampleLegalAction([0.99, 0.01, 0, 0], 0b0010, new Pcg32(seed)), 1);
  assert.equal(sampleLegalAction([0, 0, 0, 0], 0b1100, new Pcg32(seed)), 2);
  assert.equal(sampleLegalAction([0, 0, 0, 0], 0, new Pcg32(seed)), 'INVALID');
});
