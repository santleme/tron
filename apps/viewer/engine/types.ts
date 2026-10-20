export const ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"] as const;
export type ActionName = (typeof ACTIONS)[number];
export type Action = 0 | 1 | 2 | 3;
export type JointAction = Action | "INVALID";
export type Result = "agent_0_win" | "agent_1_win" | "draw";
export type CollisionCause = "out_of_bounds" | "occupied_head" | "occupied_trail" | "invalid_action" | "reverse_action" | "same_destination" | "head_swap";
export interface Position { x: number; y: number }
export interface AgentState { position: Position; heading: Action; trail: Position[]; dead: boolean }
export interface ArenaState { width: number; height: number; agents: [AgentState, AgentState]; tick: number; result: Result | null }
export interface TickRecord { tick: number; joint_actions: [JointAction, JointAction]; collision_causes: [CollisionCause[], CollisionCause[]]; terminal_causes: string[] }
export interface Transition { state: ArenaState; record: TickRecord }
