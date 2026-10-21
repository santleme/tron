# G008 Ralplan — Architect Review, Cycle 4 / Iteration 3

**Verdict: ITERATE**  
**Reviewed plan:** `.omx/plans/g008-competitive-intelligence-ralplan-draft.md`  
**Review scope:** cycle-4 re-review. No implementation performed.

## Findings

The prior findings are fixed: G007 legacy results are explicitly non-paired; the benchmark uses five paired full-update repeats; snapshots and traces are retained; downstream work is blocked on the benchmark artifact; and the exact roots and numeric gates are preserved.

One required reproducibility issue remains: checkpoint resume has not been tested for equivalence to an uninterrupted run, including NumPy RNG state. The draft defines the save/resume boundary at line 17 and records RNG IDs/URIs at line 43, but its test list at line 45 does not compare resumed execution with uninterrupted execution. In `training/ppo/train.py:313-317`, initialization seeds Python and Torch but not NumPy; `training/ppo/train.py:321-337` restores Python and Torch states but not NumPy.

Before approval, persist and restore Python, NumPy, Torch CPU, and every CUDA RNG state, along with environment-root derivation and schedule position. Add a test that, after resume, collection matches the uninterrupted run for the next environment seeds, observations, actions, transitions, and relevant PPO state under the supported deterministic contract.

## Stop condition / handoff

Architect review remains open. Address the resume-state and equivalence-test gap, then request re-review. This is review evidence only; no product code or plan was changed by this review.
