const MASK64 = (1n << 64n) - 1n;
const MULTIPLIER = 6364136223846793005n;
const encoder = new TextEncoder();

export function parseSeed(seed: string | bigint): bigint {
  const value = typeof seed === "bigint" ? seed : /^(0|[1-9][0-9]*)$/.test(seed) ? BigInt(seed) : -1n;
  if (value < 0n || value > MASK64) throw new RangeError("seed must be an unsigned 64-bit integer");
  return value;
}

function le64(value: bigint): Uint8Array {
  const out = new Uint8Array(8);
  let v = value;
  for (let i = 0; i < 8; i += 1) { out[i] = Number(v & 0xffn); v >>= 8n; }
  return out;
}

function readLe64(bytes: Uint8Array, offset: number): bigint {
  let out = 0n;
  for (let i = 7; i >= 0; i -= 1) out = (out << 8n) | BigInt(bytes[offset + i]);
  return out;
}

export interface PcgSeed { initstate: bigint; initseq: bigint }

export async function deriveSeed(masterSeed: string | bigint, label: "episode" | "agent/0" | "agent/1", indices: readonly number[]): Promise<PcgSeed> {
  const seed = parseSeed(masterSeed);
  const pieces: Uint8Array[] = [encoder.encode(`lc-arena/seed/v1/${label}`), new Uint8Array([0]), le64(seed)];
  for (const index of indices) {
    if (!Number.isSafeInteger(index) || index < 0) throw new RangeError("stream indices must be non-negative safe integers");
    pieces.push(le64(BigInt(index)));
  }
  const bytes = new Uint8Array(pieces.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const part of pieces) { bytes.set(part, offset); offset += part.length; }
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return { initstate: readLe64(digest, 0), initseq: readLe64(digest, 8) };
}

export class Pcg32 {
  private state = 0n;
  private readonly increment: bigint;

  constructor({ initstate, initseq }: PcgSeed) {
    this.increment = ((initseq << 1n) | 1n) & MASK64;
    this.nextUint32();
    this.state = (this.state + initstate) & MASK64;
    this.nextUint32();
  }

  nextUint32(): number {
    const old = this.state;
    this.state = (old * MULTIPLIER + this.increment) & MASK64;
    const xorshifted = Number((((old >> 18n) ^ old) >> 27n) & 0xffffffffn) >>> 0;
    const rotation = Number(old >> 59n) & 31;
    return ((xorshifted >>> rotation) | (xorshifted << ((-rotation) & 31))) >>> 0;
  }

  /** Uniform integer in [0, bound), using rejection rather than modulo bias. */
  nextBounded(bound: number): number {
    if (!Number.isSafeInteger(bound) || bound < 1 || bound > 0x100000000) throw new RangeError("bound must be an integer in [1, 2^32]");
    const threshold = (0x100000000 - bound) % bound;
    for (;;) { const value = this.nextUint32(); if (value >= threshold) return value % bound; }
  }
}

export async function deriveStreams(masterSeed: string | bigint, arenaIndex: number): Promise<{ episode: PcgSeed; agents: [PcgSeed, PcgSeed] }> {
  if (!Number.isSafeInteger(arenaIndex) || arenaIndex < 0) throw new RangeError("arenaIndex must be a non-negative safe integer");
  return {
    episode: await deriveSeed(masterSeed, "episode", [arenaIndex]),
    agents: [await deriveSeed(masterSeed, "agent/0", [arenaIndex, 0]), await deriveSeed(masterSeed, "agent/1", [arenaIndex, 1])],
  };
}
