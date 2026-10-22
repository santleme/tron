import wasmUrl from './generated/fixed_depth.wasm?url';
import { writePlannerInput } from './plannerPacket.ts';
import type { ArenaState, JointAction } from '../engine/types.ts';
import type { Pcg32 } from '../engine/prng.ts';
import { sampleLegalAction } from './searchPolicy.ts';

type PlannerExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  circuit_input_ptr(): number;
  circuit_input_len(): number;
  circuit_output_ptr(): number;
  circuit_plan(): number;
};

let planner: Promise<PlannerExports> | undefined;

async function loadPlanner(): Promise<PlannerExports> {
  planner ??= (async () => {
    const response = await fetch(wasmUrl);
    if (!response.ok) throw new Error(`Could not load the fixed-depth WASM planner (${response.status})`);
    const { instance } = await WebAssembly.instantiate(await response.arrayBuffer());
    const exports = instance.exports as PlannerExports;
    for (const name of ['memory', 'circuit_input_ptr', 'circuit_input_len',
      'circuit_output_ptr', 'circuit_plan']) {
      if (!(name in exports)) throw new Error(`Fixed-depth WASM planner is missing ${name}`);
    }
    return exports;
  })();
  return planner;
}

/** Search once and sample both players from their equilibrium policies. */
export async function fixedDepthActions(
  state: ArenaState,
  rngs: readonly [Pcg32, Pcg32],
): Promise<[JointAction, JointAction]> {
  const wasm = await loadPlanner();
  const bytes = new Uint8Array(wasm.memory.buffer, wasm.circuit_input_ptr(), wasm.circuit_input_len());
  writePlannerInput(state, bytes);

  if (wasm.circuit_plan() !== 0) throw new Error('Fixed-depth WASM planner rejected the arena state');
  const output = new Float64Array(wasm.memory.buffer, wasm.circuit_output_ptr(), 15);
  return [0, 1].map((seat) => {
    const policy = Array.from(output.slice(1 + seat * 4, 5 + seat * 4));
    return sampleLegalAction(policy, output[9 + seat] | 0, rngs[seat]);
  }) as [JointAction, JointAction];
}
