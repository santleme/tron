# G008 next cycle proposal

**State:** proposal only; not registered or authorized for execution.
**Current campaign:** `g008-competitive-20261021-v2`; all registered work is consumed.
**Goal:** improve the policy while preserving an independent, unused final evaluation and the existing promotion criteria.

## Evidence already available

- The v2 final candidate recorded 18 wins, 24 draws, and 470 losses over 512 games against `open-space`. It failed the 60% observed-win promotion criterion by a wide margin.
- The paired strategy comparison selected the all-latest baseline. The mixed opponent curriculum scored a mean paired delta of -0.037435 versus baseline; its 95% interval was [-0.099109, 0.024239], and it regressed in all three seed pairs. Repeating the same mixed curriculum is not justified by this result.
- The selected pair-2 baseline scored 61.23% against `g007-step-300` when draws count as half a win, but its observed win rate was 58.01% (297/512); the open-space final observed win rate was 3.52%. These are different opponents, and the validation score is not the final promotion metric. The detailed split and diagnostic limits are recorded in [G008 validation failure analysis](G008_VALIDATION_FAILURE_ANALYSIS.md).
- The six retained v2 validation reports classify 1,084 of 1,138 losses (95.3%) as having a locally safe terminal action under a one-step counterfactual that holds the observed opponent action fixed. `occupied_trail` caused 957/1,138 candidate losses (84.1%). This supports investigating action safety and space preservation, but does not prove that those alternatives would win against a reacting opponent.
- Existing older v15 diagnostics reconstructed 244 losses and found 584 locally dominated decisions with zero observation parity errors. The source report had no per-tick action trace, so the diagnosis reconciles outcomes and durations rather than the original action trajectory. Treat it as a hypothesis source, not a direct diagnosis of the v2 final checkpoint.
- Training logs show that entropy declined from about 1.10 near initialization to 0.17 for the baseline and 0.11 for the mixed seed-pair-1 runs by update 1000. This confirms convergence toward lower-entropy policies; by itself it does not establish either healthy convergence or harmful collapse.

The registry is authoritative for exact per-run outputs, checkpoints, and hashes. This proposal summarizes the already-recorded results and does not authorize more analysis against the consumed final holdout.

## Working hypothesis

The current training distribution and selection opponent do not reward the survival and space-preservation behavior needed against a strong open-space policy. The v2 mixed opponent pool failed to improve even the existing step-300 validation score, while the selected baseline still transferred poorly to the final opponent. The first next-cycle objective is to measure and reduce this distribution gap using train and validation data only.

## Proposed experiment sequence after new authorization

1. **Freeze a new campaign.** Review the exact trainer, engine, source manifest, seed plan, train/validation rosters, compute budget, and workload-specific authorization. Freeze a new protected final roster; never reuse the consumed v2 final roster.
2. **Use existing train/validation evidence first.** Use the already-retained per-game v2 validation reports and their hash-linked diagnostics in [G008 validation failure analysis](G008_VALIDATION_FAILURE_ANALYSIS.md). Do not repeat that analysis or run another replay sweep. Focus any future train/validation comparison on safe-action selection and space preservation; the available one-step counterfactuals do not establish strategic superiority.
3. **Compare a small paired set of training arms.** Keep one unchanged baseline. Compare it with (a) a more diverse non-final opponent curriculum and (b) one narrowly specified survival/space reward adjustment. Use identical paired training roots, run lengths, and validation schedules. Do not expose the protected final policy, roster, or labels to training or selection.
4. **Select only on validation.** Require a predeclared paired improvement, no seed-pair regression beyond the registered tolerance, and stable behavior across the selected validation opponent mix. Preserve every run, checkpoint, report, and replay, including rejected arms.
5. **Freeze one candidate.** Bind the selected checkpoint, package tree, source manifest, config, trainer/engine hashes, and final roster before final evaluation starts.
6. **Run one independent final evaluation on Kaggle.** Use a new protected roster with at least 512 games against the exact approved `open-space` policy. Promote only if the registry validator and final report both pass at least 60% observed wins. A failure remains a failed result and does not authorize tuning against that roster.

## Authorization gate

No training, simulation, benchmark, replay sweep, or evaluation is authorized by this document. Before the next Kaggle operation, the project needs a fresh campaign review and registry registration, workload-specific authorization, remaining compute-budget check, explicit resource caps, and live Kaggle preflight. Account quota or an old authorization does not satisfy these requirements.
