import { step } from "./engine.ts";
import type { ArenaState, JointAction, TickRecord } from "./types.ts";

const encoder = new TextEncoder();

function compareCodePoints(a: string, b: string): number {
  const aa = Array.from(a, c => c.codePointAt(0)!);
  const bb = Array.from(b, c => c.codePointAt(0)!);
  for (let i = 0; i < Math.min(aa.length, bb.length); i += 1) if (aa[i] !== bb[i]) return aa[i] - bb[i];
  return aa.length - bb.length;
}

function validateCanonical(value: unknown): void {
  if (typeof value === "string") {
    if (value.normalize("NFC") !== value) throw new TypeError("canonical JSON strings must be NFC-normalized");
  } else if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError("canonical JSON numbers must be finite");
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) throw new RangeError("canonical JSON integers must be in the interoperable range");
  } else if (Array.isArray(value)) value.forEach(validateCanonical);
  else if (value && typeof value === "object") {
    for (const [key, child] of Object.entries(value)) { validateCanonical(key); validateCanonical(child); }
  } else if (value !== null && typeof value !== "boolean") throw new TypeError("unsupported canonical JSON value");
}

function canonical(value: unknown): string {
  validateCanonical(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => compareCodePoints(a, b));
    return `{${entries.map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}`;
  }
  const out = JSON.stringify(value);
  if (out === undefined) throw new TypeError("unsupported canonical JSON value");
  return out;
}

export function canonicalJson(value: unknown): string { return canonical(value); }

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes.slice().buffer as ArrayBuffer));
  return Array.from(digest, byte => byte.toString(16).padStart(2, "0")).join("");
}

export async function digestObject<T extends Record<string, unknown>>(value: T, digestField: keyof T): Promise<string> {
  const { [digestField]: digest, ...content } = value;
  void digest;
  return sha256Hex(encoder.encode(canonicalJson(content)));
}

export async function verifyObjectDigest(value: Record<string, unknown>, digestField: string): Promise<boolean> {
  const expected = value[digestField];
  if (typeof expected !== "string" || !/^[0-9a-f]{64}$/.test(expected)) return false;
  return (await digestObject(value, digestField)) === expected;
}

/** Rebuilds every recorded frame from joint actions; fails closed on a tick mismatch. */
export function replayActions(initial: ArenaState, actions: readonly (readonly [JointAction, JointAction])[]): { states: ArenaState[]; records: TickRecord[]; result: ArenaState["result"] } {
  let state = initial;
  const states = [state];
  const records: TickRecord[] = [];
  for (const joint of actions) {
    if (state.result !== null) throw new Error("replay contains actions after the terminal result");
    const transition = step(state, joint);
    state = transition.state;
    states.push(state);
    records.push(transition.record);
  }
  return { states, records, result: state.result };
}
