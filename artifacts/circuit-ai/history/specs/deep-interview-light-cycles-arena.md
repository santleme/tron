# Requirements — Light Cycles AI Arena (MVP)

## Intent
Build a personal, extensible Light Cycles-style game where the user can observe neural agents learning through self-play. Train in Kaggle and inspect/use the resulting checkpoints locally. The goal is both the game and a visible, evidence-backed training experiment—not a multi-user product.

## MVP scope
- A deterministic grid-based survival arena with simultaneous movement and collision resolution; one life per program/agent.
- Initially two AI agents per match, using self-play.
- Local web interface, with a visually compelling digital-grid/Light Cycles aesthetic (glowing trails and cinematic presentation are part of the seed request).
- Observe self-play and trained checkpoint behavior locally.
- Training metrics and replay artifacts are first-class outputs. Include checkpoint evaluation across training progress, with measurable win/survival/collision signals and replays that let the user inspect play.
- Reproducible Kaggle training plus local Kaggle automation to submit/start a run, monitor it, and download resulting checkpoints/metrics into a locally consumable run/artifact format.
- PPO network updates must execute on GPU in Kaggle. Vectorized headless arenas and throughput measurement are desired; choose one or two T4s by measured efficiency, not by a requirement that both are always occupied.

## Later / out of MVP
- Human-versus-AI play.
- Full checkpoint tournament mode.
- A wider simulation of The Grid as a society, including social/citizen/campaign/RPG systems; the user may revisit if the arena proves interesting.

## Explicit non-goals / boundaries
- No multi-user accounts, hosted shared service, or social platform: this is a personal project.
- Do not execute a Kaggle job or use the user's Kaggle credentials as part of this delivery; provide user-invoked automation and safe credential configuration guidance.
- Do not force both T4 GPUs into each run if that reduces training throughput.
- Do not expand MVP into society simulation.

## Acceptance criteria to operationalize in planning
1. A local browser experience can run deterministic matches and display self-play without training-time rendering.
2. Collision/survival outcomes are reproducible under a fixed seed and simultaneous moves are resolved consistently.
3. Kaggle training updates PPO parameters on GPU; logs/checkpoints are portable and loadable by the local experience.
4. The training artifact includes metrics and replays; held-out/checkpoint evaluation can demonstrate progress against earlier snapshots and a simple baseline.
5. A user-invoked automation workflow can submit/launch a Kaggle run, report its status, and download expected outputs; tests must not need account credentials or launch a real Kaggle job.
6. A short benchmark compares efficient configurations and records arenas/steps per second and GPU use; choose throughput over nominal two-GPU occupancy.
7. Future human play, full tournaments, and society simulation remain separately extensible rather than blocking the MVP.

## Decision boundaries
- User grants freedom to choose architecture and implementation.
- Preserve the above product scope; planning may select implementation details based on maintainability and measured throughput.
- The user explicitly chose self-play-first and throughput-first, not three-mode completeness or mandatory dual-GPU use.

## Technical facts to confirm during planning
- Current Kaggle CLI/notebook APIs, output/download behavior, and GPU runtime constraints require current official-documentation confirmation.
- The user's statement that Kaggle T4 x2 is available is an input constraint, not yet independently verified in this environment.

## Interview status
Standard deep interview completed below the 0.20 ambiguity threshold after an explicit pressure pass. See `.omx/interviews/light-cycles-arena-20261019T211321Z.md` and context `.omx/context/a-proposta-construir-uma-arena-de-light-cycles-e-20261019T210425Z.md`.
