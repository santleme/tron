# Process review: OMX and direct execution

**Question:** did this project progress better with OMX or without it?

## Verdict

For the latest train-and-evaluate task, direct execution was the better delivery path: it produced a completed Kaggle continuation, a 512-game comparison, and a checkpoint sweep without routing ordinary exploratory work through repeated OMX planning and review cycles.

That is not evidence that OMX made the model worse. There was no controlled comparison, and the direct run reused the G008 trainer, seed schedule, evaluator, and contracts built during the OMX-heavy period. The fairest conclusion is **hybrid**: use direct execution for bounded exploratory work; keep the useful reproducibility contracts and reserve the full registry and approval path for official promotion or final holdout claims.

## Measured project outcomes

| Evidence | Result | What it establishes |
| --- | --- | --- |
| G007 `step-300` in `RESULTS.md` | 58/2/4 vs initialization; 1/4/59 vs `open-space`; 1,200 optimizer updates | The project learned against its own initialization, but remained weak against its stronger heuristic opponent. |
| G008 frozen T4 baseline (`g008/download-v9`) | Median 1,276 joint transitions/s, against a 10,000/s target | A measured baseline, not an optimization win; rollout dominated optimizer time. |
| Exploratory G008 `step-3` vs G007 `step-300` | 33W/18D/461L (6.45%) over 512 games | A very weak short-run starting checkpoint. |
| Exploratory G008 `step-236` vs the same G007 checkpoint and 512 seeds | 217W/51D/244L (42.38%); seat results 106W and 111W | A substantial improvement on this fixed suite. It is still below 50% wins and is not a promotion result. |
| Intermediate checkpoint sweep | Step 236 had the highest observed win count (217/512); progress was non-monotonic | Large gains appeared early, then results moved around 38–42% from steps 128–236. This suite is diagnostic only. |

In the paired step-3 to step-236 comparison, 185 games changed from loss to win, 44 from loss to draw, and 7 from win to loss. Mean game length rose from 5.75 to 12.82 ticks. This supports improvement against this opponent and seed set; it does not establish broad strength or generalization.

## What OMX contributed

The OMX period produced real engineering: split-aware schedules, provenance contracts, frozen benchmark configuration, append-only experiment history, Kaggle recovery handling, and tests that caught seed, registry, SDK metadata, and lifecycle defects. Those are useful when results must be reproducible and promotion decisions auditable.

The execution record also shows process cost and state drift. The repository currently has 67 files under `.omx/plans/` and 36 under `.omx/evidence/`. The Ultragoal ledger has 18 records, including 12 resumes and four steering changes, but no completion event. Its active goal is still the original Stage 0; G008 recovery goals remain marked pending despite subsequent Kaggle work. The original submission cap was also superseded after being exceeded. These records were not a reliable live progress tracker.

## What the evidence cannot say

- There is no controlled counterfactual proving the same work would have been completed sooner or produced a better model without OMX.
- The latest continuation was direct in orchestration, but depended on G008 code and contracts created earlier; it was not a clean “no OMX” project.
- The exploratory evaluator records outcomes and terminal reasons but not action/state traces. It cannot explain why the remaining 244 games were lost.
- The training metrics omit reward, entropy, KL, and separated policy/value losses. A falling aggregate loss does not prove the policy stopped improving.
- The 42.38% result is not Kaggle competition performance, a registered benchmark, or a promotion claim.

## What to improve next

1. Keep two lanes. Run exploratory train/evaluate loops directly with a concise config, quota check, and saved artifacts. Use registry, lock, attestation, and holdout gates when making official benchmark or promotion claims.
2. Keep one short, current progress record. Update it when work completes; do not create another review document for the same unchanged question.
3. Use native agents for independent work slices and reviews, then integrate once. Do not serialize ordinary execution through multiple plan-review roles.
4. Improve experiment observability: record reward/return, entropy, KL, policy/value loss, and episode length. Preserve periodic checkpoints and evaluate them on a diagnostic suite, then use fresh seeds for finalist selection.
5. The next training experiment should test one change at a time. The completed run used the latest self-play opponent for all arenas and no snapshot or heuristic opponents. A bounded opponent-mixture experiment is a plausible hypothesis, not a proven fix; compare it against step 236 on fresh seeds and keep the final-only opponent isolated.

## Artifacts

- G007 and G008 baseline history: `artifacts/circuit-ai/RESULTS.md`
- Step-3 and step-236 reports: `artifacts/circuit-ai/evaluations/g008-v15-exploratory/` and `artifacts/circuit-ai/evaluations/g008-v15-continuation-236/`
- The 11-checkpoint exploratory sweep: `artifacts/circuit-ai/evaluations/g008-v15-continuation-236/checkpoint-sweep/`
- Relevant OMX plans: `.omx/plans/g008-competitive-intelligence-plan.md` and `.omx/plans/g008-kaggle-recovery-plan.md`
