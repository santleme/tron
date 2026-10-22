import type { ArenaState } from '../engine/types.ts';

const PACKET_BYTES = 46;

export function writePlannerInput(
  state: ArenaState,
  bytes: Uint8Array,
  depth = 3,
  edgeBudget = 56,
): void {
  if (bytes.length !== PACKET_BYTES || state.width * state.height > 256) {
    throw new Error('Fixed-depth WASM planner supports boards of at most 256 cells');
  }
  bytes.fill(0);
  bytes[0] = state.width;
  bytes[1] = state.height;
  for (let player = 0; player < 2; player++) {
    const agent = state.agents[player];
    bytes[2 + player] = agent.position.y * state.width + agent.position.x;
    bytes[4 + player] = agent.heading;
    if (!agent.dead) bytes[7] |= 1 << player;
    for (const point of agent.trail) markOccupied(point.x, point.y);
    markOccupied(agent.position.x, agent.position.y);
  }
  if (state.result === 'agent_0_win') bytes[6] = 1;
  else if (state.result === 'agent_1_win') bytes[6] = 2;
  else if (state.result === 'draw') bytes[6] = 3;
  bytes[8] = depth;
  new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    .setUint32(10, edgeBudget, true);

  function markOccupied(x: number, y: number) {
    const cell = y * state.width + x;
    bytes[14 + (cell >> 3)] |= 1 << (cell & 7);
  }
}
