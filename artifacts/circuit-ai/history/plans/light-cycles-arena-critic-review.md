# Critic Review — Light Cycles AI Arena RALPLAN revision 1

**Verdict: APPROVE**

## Justification

The revised RALPLAN, PRD, and test specification are mutually consistent and actionable for the next supervised execution phase. They preserve the clarified personal, local-first, self-play MVP; require downloaded-checkpoint browser inference and fresh local matches; defer human play, tournaments, and Grid society simulation; and treat Kaggle submission as an explicit user-invoked operation. The selected two-runtime architecture has a real, acknowledged parity cost and concrete gates rather than hiding that risk. No planning blocker found.

## Review findings

- **Principles/options/recommendation:** The principles prioritize browser-local trained-checkpoint use, deterministic artifacts, throughput, and safe external side effects. Options A and B are distinguishable on the decisive tradeoff: Python-only replay viewing minimizes semantic/runtime duplication but fails the requested browser-local inference behavior; Python reference/training plus TypeScript/ONNX browser runtime satisfies that behavior while adding parity and export risks. The plan fairly states those costs and chooses B for the user's stated product requirement; it does not claim B is simpler or universally superior.
- **Requirements and scope:** The specification and interview support a personal training/inspection tool, two-agent survival self-play, metrics/replays, Kaggle submit-watch-download automation, GPU PPO updates, and throughput-first selection. The plan keeps the speculative Grid society as a later idea and excludes the original three-mode expansion from MVP. It does not assume both T4s must be occupied.
- **Acceptance and verification:** Acceptance is observable: deterministic collision and seed fixtures; Python/TypeScript golden parity; local offline ONNX inference producing a fresh match/replay; optimizer update/device evidence; fixed held-out evaluation with seat swaps, opponent identities, W/D/L, uncertainty, survival/collision metrics and per-game artifacts; mocked Kaggle side effects; and benchmark records/configuration rationale. Separating artifact/report completeness from a guaranteed PPO win-rate gain is honest and testable; no unsupported learning claim is required.
- **Risk and sequencing:** The plan gates broad work on shared contract fixtures and ONNX compatibility, Python rules before TS parity, then training/export, browser inference, Kaggle integration, and measurement. It identifies parity drift, ONNX support, PPO learning uncertainty, GPU starvation/dual-GPU overhead, mutable Kaggle constraints, partial downloads, and scope creep with corresponding mitigations and stop conditions.
- **Kaggle safety/docs caveats:** Only the explicit `submit` command with kernel, source revision, and `--confirm` may push; status/download are explicit and bounded, tests/examples default to dry-run/mocks, and artifact promotion is staged, validated, atomic, and no-overwrite. The plan forbids credentials or a real remote run in this delivery. It cites official Kaggle CLI/auth/notebook/GPU-use and PyTorch DDP/tuning references, and correctly treats quotas, runtime, CLI versions, and actual T4 availability as mutable/external evidence rather than established facts.
- **Required consensus checks:** Principle/option consistency passes; alternative treatment is sufficient for this bounded architecture decision; risk and verification are specific and evidence-oriented. No deliberate-mode pre-mortem expansion is required for this short review; the risk list and fail/stop gates provide a credible implementation pre-mortem.

## Non-blocking implementation gates

Carry these already identified details into the first contract milestone before source lanes diverge: select a single shared schema/fixture location; freeze observation tensor shape/orientation and supported model/opset/runtime; pin canonical numeric serialization and digest exclusions to avoid self-reference. Confirm Kaggle version-to-CLI reference mapping and validate downloaded run identity at implementation time. These are explicitly assigned to the contract/compatibility gate and do not prevent the approved plan from proceeding.

## Stop condition

Critic gate is complete: Planner/Ultragoal may proceed under the approved revision, preserving the stated contract-first gates. This review did not implement source code, use Kaggle credentials, or launch a Kaggle job.
