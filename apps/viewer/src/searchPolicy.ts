import type { JointAction } from '../engine/types.ts';
import type { Pcg32 } from '../engine/prng.ts';

export function sampleLegalAction(
  probabilities: readonly number[],
  legalMask: number,
  rng: Pick<Pcg32, 'nextUint32'>,
): JointAction {
  if ((legalMask & 0b1111) === 0) return 'INVALID';
  if (probabilities.length !== 4) throw new TypeError('search policy must contain four action probabilities');

  const legal: number[] = [];
  let total = 0;
  for (let action = 0; action < 4; action++) {
    const probability = probabilities[action];
    if (!Number.isFinite(probability) || probability < 0) throw new TypeError('search policy probabilities must be finite and non-negative');
    if ((legalMask & (1 << action)) !== 0) {
      legal.push(action);
      total += probability;
    }
  }
  if (total <= 0) return legal[0] as JointAction;

  let target = (rng.nextUint32() / 0x1_0000_0000) * total;
  for (const action of legal) {
    target -= probabilities[action];
    if (target < 0) return action as JointAction;
  }
  return legal.at(-1) as JointAction;
}
