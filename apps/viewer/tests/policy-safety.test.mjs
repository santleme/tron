import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseImmediatelySafeAction, chooseLegalAction } from '../engine/engine.ts';

test('the optional safety mask avoids immediate wall and trail collisions', () => {
  const state = {
    width: 3,
    height: 3,
    tick: 0,
    result: null,
    agents: [
      { position: { x: 0, y: 0 }, heading: 1, trail: [{ x: 0, y: 0 }], dead: false },
      { position: { x: 2, y: 2 }, heading: 3, trail: [{ x: 2, y: 2 }], dead: false },
    ],
  };

  assert.equal(chooseLegalAction([9, 2, 1, 100], 1), 0);
  assert.equal(chooseImmediatelySafeAction([9, 2, 1, 100], state, 0), 1);
});

test('the safety mask falls back to the best non-reverse action when trapped', () => {
  const state = {
    width: 2,
    height: 2,
    tick: 0,
    result: null,
    agents: [
      { position: { x: 0, y: 0 }, heading: 1, trail: [{ x: 0, y: 0 }], dead: false },
      { position: { x: 0, y: 1 }, heading: 3,
        trail: [{ x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }], dead: false },
    ],
  };

  assert.equal(chooseImmediatelySafeAction([0, 9, 4, 100], state, 0), 1);
});
