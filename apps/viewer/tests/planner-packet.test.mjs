import assert from 'node:assert/strict';
import test from 'node:test';
import { writePlannerInput } from '../src/plannerPacket.ts';

test('WASM packet preserves arena heads, headings, alive flags, trails and budget', () => {
  const state = {
    width: 4, height: 3, tick: 0, result: null,
    agents: [
      { position: { x: 1, y: 1 }, heading: 1, trail: [{ x: 1, y: 1 }, { x: 2, y: 1 }], dead: false },
      { position: { x: 3, y: 2 }, heading: 3, trail: [{ x: 3, y: 2 }], dead: true },
    ],
  };
  const packet = new Uint8Array(46);
  writePlannerInput(state, packet);

  assert.deepEqual(Array.from(packet.slice(0, 9)), [4, 3, 5, 11, 1, 3, 0, 1, 3]);
  assert.equal(new DataView(packet.buffer).getUint32(10, true), 56);
  assert.equal(packet[14], (1 << 5) | (1 << 6));
  assert.equal(packet[15], 1 << 3);
});

test('WASM packet encodes terminal results and rejects unsupported board sizes', () => {
  const state = {
    width: 2, height: 2, tick: 1, result: 'draw',
    agents: [
      { position: { x: 0, y: 0 }, heading: 0, trail: [], dead: true },
      { position: { x: 1, y: 1 }, heading: 2, trail: [], dead: true },
    ],
  };
  const packet = new Uint8Array(46);
  writePlannerInput(state, packet, 2, 32);
  assert.equal(packet[6], 3);
  assert.equal(packet[7], 0);
  assert.equal(packet[8], 2);
  assert.equal(new DataView(packet.buffer).getUint32(10, true), 32);
  assert.throws(() => writePlannerInput(state, new Uint8Array(45)), /46-byte|supports/);
});
