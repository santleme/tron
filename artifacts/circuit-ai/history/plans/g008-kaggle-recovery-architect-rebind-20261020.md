# G008 Kaggle Recovery — Architect Rebind Review

- **Session:** `01a15941-194a-7c70-b7ef-ecbf54f67dba`
- **Reviewed plan:** `.omx/plans/g008-kaggle-recovery-plan.md`
- **Plan SHA-256:** `0358c3c00ffa438a3bd033939687827774da5f04ba54746c075eb4680b384d72`
- **Planner artifact:** `.omx/plans/g008-kaggle-recovery-planner-rebind-20261020.md`
- **Planner artifact SHA-256:** `a79355fe04e3c06192ec56e127169725dfdd65ef57e502f5883722a8a00547bd`
- **Verdict:** **APPROVE (planning only; Critic is next).**

I verified the plan's current bytes against the requested SHA-256 and read the current-session Planner rebind. The Planner's recommendation to preserve these exact bytes is architecturally sound. This rebind does not change the approved plan, authorize registry mutation, or authorize a Kaggle operation.

## Architectural assessment

The plan keeps the key boundaries intact: benchmark and training are two frozen workload registrations; benchmark evidence is performance/parity-only; the benchmark artifact is linked to a terminal attempt and immediately precedes lock closure; training runs only after lock; and no-count retries remain bounded by cumulative compute, quota, output, wall-time and protected-reserve constraints. It also explicitly preserves the original seed/provenance/holdout/final-evaluation contracts and prevents a failed 10,000 transitions/s gate from silently canceling the separately requested bounded training run (`g008-kaggle-recovery-plan.md:14, 38, 60, 99-114, 118-133`).

The paired experiment is properly designed to compare identical initial model, optimizer, and RNG states with frozen inputs, deterministic transition parity, shared run identity, and alternating order before timing; the candidate selection rule requires parity plus both the median and per-pair gain thresholds (`g008-kaggle-recovery-plan.md:92, 101-102`). That separates optimization evidence from strategic training outcomes and avoids letting benchmark artifacts leak into evaluation.

The recovery amendment is the narrow architectural seam for the historical count-cap exception: the plan scopes it to exactly two named registrations, preserves append-only prior history and the ordinary campaign cap, and keeps resource limits authoritative (`g008-kaggle-recovery-plan.md:4, 38, 77, 121-130`). The existing final Architect and Critic records bind the same plan hash and describe planning-only approval with implementation and fresh-preflight gates still outstanding (`g008-kaggle-recovery-architect-review-final.md`, `g008-kaggle-recovery-critic-review-final.md`).

### Backend identity boundary

The canonical backend descriptor and digest described in the current recovery evidence are acceptable as a **metadata-only execution contract**: store the exact descriptor and canonical digest together inside `recovery_amendment.reservation_metadata`, independently recompute and compare the digest before registry append, and bind that same digest to every corresponding frozen workload variant. Do not represent it as a hash of Kaggle's actual container or runtime image. The actual Kaggle image, Python/PyTorch/CUDA versions, and observed accelerator remain separate runtime facts captured from each remote artifact, as the plan requires (`g008-kaggle-recovery-plan.md:101, 121, 124`). This interpretation prevents a pre-run contract hash from overstating what it proves while keeping the backend field stable across the paired variants.

## Steelman antithesis

The strongest objection is that allowing retries without a count ceiling can create unbounded remote submission traffic if failures consume little or no accelerator quota; likewise, the backend contract digest cannot prove that Kaggle ran the declared image or software stack. These are real limits: compute-only accounting does not bound every rejected or near-zero-cost push, and a declared contract is not remote attestation. A planning approval could therefore appear broader than the evidence it will eventually produce.

## Tradeoff tension

Removing the count ceiling honors the user's explicit recovery instruction and avoids stopping a valid frozen workload solely because of an arbitrary attempt number. It reduces a simple operational bound, so safety depends more heavily on exact version assignment, single-writer controls, terminal reconciliation, live policy/quota checks, immutable package identity, cumulative resource reservations, and hard stops on ambiguous status. Separately, freezing a canonical backend contract makes paired comparisons auditable but cannot guarantee runtime image identity before Kaggle executes.

## Synthesis

Retain the two-registration, count-free retry policy under the exact per-attempt and cumulative limits in the plan; every retry must reuse frozen bytes, wait for terminal reconciliation, and pass the same current preflight. Keep the backend digest explicitly scoped to declared provisioning/accelerator/dependency semantics, and use captured remote runtime facts to describe what actually executed. The plan already places those controls at the correct boundaries; neither limitation requires changing its approved bytes. Any mismatch in exact version, exclusive-writer state, runtime/package identity, quota, output, or registry ancestry remains a stop condition.

## Gate disposition

**Approve the exact plan for planning only.** This is not execution authority. Before registry or Kaggle work, the leader still needs completed sequential Critic approval and the bound handoff, passing implementation/lifecycle/package validations, a fresh exact-kernel/quota reconciliation, and the plan's actual-preflight no-outside-writer confirmation.
