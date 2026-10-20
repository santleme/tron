import { ACTIONS, type Action, type ArenaState, type CollisionCause, type JointAction, type Position, type TickRecord, type Transition } from "./types.ts";
import { deriveStreams, Pcg32, type PcgSeed, parseSeed } from "./prng.ts";

const DELTAS: readonly Position[] = [{ x: 0, y: -1 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: -1, y: 0 }];
const key = ({ x, y }: Position) => `${x},${y}`;
const copyPosition = ({ x, y }: Position): Position => ({ x, y });

export function validateState(state: ArenaState): void {
  if (!Number.isSafeInteger(state.width) || !Number.isSafeInteger(state.height) || state.width < 2 || state.height < 2) throw new RangeError("board dimensions must be at least 2");
  if (state.agents.length !== 2) throw new TypeError("exactly two agents are required");
  for (const agent of state.agents) {
    if (!Number.isInteger(agent.heading) || agent.heading < 0 || agent.heading > 3) throw new RangeError("heading must be an action index in [0, 3]");
    if (agent.trail.length === 0 || key(agent.trail.at(-1)!) !== key(agent.position)) throw new TypeError("each agent trail must end at its current head");
  }
}

export function observe(state: ArenaState, agentIndex: 0 | 1): number[][][] {
  validateState(state);
  const opponent = (1 - agentIndex) as 0 | 1;
  const grid = Array.from({ length: state.height }, () => Array.from({ length: state.width }, () => Array<number>(12).fill(0)));
  for (const [i, agent] of state.agents.entries()) {
    const trail = i === agentIndex ? 2 : 3;
    const head = i === agentIndex ? 0 : 1;
    for (const cell of agent.trail) grid[cell.y][cell.x][trail] = 1;
    grid[agent.position.y][agent.position.x][head] = 1;
    grid[agent.position.y][agent.position.x][4 + (i === agentIndex ? 0 : 4) + agent.heading] = 1;
  }
  // Keep explicit local binding to make the seat-relative channel mapping apparent.
  void opponent;
  return grid;
}

export function chooseAction(logits: readonly number[]): Action {
  if (logits.length !== 4 || logits.some(v => !Number.isFinite(v))) throw new TypeError("policy must return four finite action scores");
  let best: Action = 0;
  for (let i = 1; i < 4; i += 1) if (logits[i] > logits[best]) best = i as Action;
  return best;
}

function hasCause(causes: CollisionCause[], cause: CollisionCause): void { if (!causes.includes(cause)) causes.push(cause); }
function inside(p: Position, s: ArenaState): boolean { return p.x >= 0 && p.x < s.width && p.y >= 0 && p.y < s.height; }
function resultFor(causes: [CollisionCause[], CollisionCause[]]): ArenaState["result"] {
  const dead0 = causes[0].length > 0, dead1 = causes[1].length > 0;
  return dead0 && dead1 ? "draw" : dead0 ? "agent_1_win" : dead1 ? "agent_0_win" : null;
}

export function step(state: ArenaState, actions: readonly [JointAction, JointAction]): Transition {
  validateState(state);
  if (state.result !== null) return { state, record: { tick: state.tick, joint_actions: [...actions] as [JointAction, JointAction], collision_causes: [[], []], terminal_causes: [] } };
  const starts = state.agents.map(a => copyPosition(a.position)) as [Position, Position];
  const destinations: [Position | null, Position | null] = state.agents.map((agent, i) => {
    const action = actions[i];
    if (typeof action !== "number" || !Number.isInteger(action) || action < 0 || action >= ACTIONS.length || agent.dead) return null;
    if (((action + 2) % 4) === agent.heading) return null;
    const direction = action as Action;
    const delta = DELTAS[direction];
    return { x: agent.position.x + delta.x, y: agent.position.y + delta.y };
  }) as [Position | null, Position | null];
  const causes: [CollisionCause[], CollisionCause[]] = [[], []];
  for (let i = 0; i < 2; i += 1) {
    const action = actions[i];
    if (state.agents[i].dead || action === "INVALID" || typeof action !== "number" || !Number.isInteger(action) || action < 0 || action > 3) hasCause(causes[i], "invalid_action");
    else if (((action + 2) % 4) === state.agents[i].heading) hasCause(causes[i], "reverse_action");
    const dest = destinations[i];
    if (dest === null) continue;
    if (!inside(dest, state)) hasCause(causes[i], "out_of_bounds");
    else {
      for (let j = 0; j < 2; j += 1) {
        if (key(dest) === key(starts[j])) hasCause(causes[i], "occupied_head");
        else if (state.agents[j].trail.some(p => key(p) === key(dest))) hasCause(causes[i], "occupied_trail");
      }
    }
  }
  if (destinations[0] !== null && destinations[1] !== null && key(destinations[0]) === key(destinations[1])) { hasCause(causes[0], "same_destination"); hasCause(causes[1], "same_destination"); }
  if (destinations[0] !== null && destinations[1] !== null && key(destinations[0]) === key(starts[1]) && key(destinations[1]) === key(starts[0])) { hasCause(causes[0], "head_swap"); hasCause(causes[1], "head_swap"); }

  const result = resultFor(causes);
  const agents = state.agents.map((agent, i) => {
    if (causes[i].length || agent.dead) return { ...agent, position: copyPosition(agent.position), trail: agent.trail.map(copyPosition), dead: true };
    const destination = destinations[i]!;
    return { position: copyPosition(destination), heading: actions[i] as Action, trail: [...agent.trail.map(copyPosition), copyPosition(destination)], dead: false };
  }) as ArenaState["agents"];
  const nextState: ArenaState = { width: state.width, height: state.height, agents, tick: state.tick + 1, result };
  const terminal_causes = result === null ? [] : causes.map((list, i) => list.length ? `agent_${i}_dead` : "").filter(Boolean);
  const record: TickRecord = { tick: state.tick, joint_actions: [...actions] as [JointAction, JointAction], collision_causes: causes, terminal_causes };
  return { state: nextState, record };
}

export async function createSeededState(options: { width: number; height: number; masterSeed: string | bigint; arenaIndex: number }): Promise<{ state: ArenaState; streams: { episode: PcgSeed; agents: [PcgSeed, PcgSeed] }; master_seed: string; arena_index: number }> {
  const { width, height, arenaIndex } = options;
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 2 || height < 2 || width * height < 2) throw new RangeError("board dimensions must be at least 2");
  const master = parseSeed(options.masterSeed);
  const streams = await deriveStreams(master, arenaIndex);
  const episodeRng = new Pcg32(streams.episode);
  const ids: [number, number] = [episodeRng.nextBounded(width * height), 0];
  do { ids[1] = episodeRng.nextBounded(width * height); } while (ids[1] === ids[0]);
  const coords = ids.map(id => ({ x: id % width, y: Math.floor(id / width) })) as [Position, Position];
  const headings = streams.agents.map(seed => new Pcg32(seed).nextBounded(4)) as [Action, Action];
  const state: ArenaState = { width, height, tick: 0, result: null, agents: [0, 1].map(i => ({ position: coords[i], heading: headings[i], trail: [copyPosition(coords[i])], dead: false })) as ArenaState["agents"] };
  return { state, streams, master_seed: master.toString(), arena_index: arenaIndex };
}

export function actionNames(): readonly string[] { return ACTIONS; }
