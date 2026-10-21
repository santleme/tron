# G008 Ralplan — Critic Review

**Verdict: APPROVE**  
**Reviewed plan:** `.omx/plans/g008-competitive-intelligence-ralplan-draft.md`  
**Requirements:** `.omx/specs/g008-competitive-intelligence.md`  
**Context/interview:** `.omx/context/g008-competitive-intelligence-20261020T002306Z.md`; `.omx/interviews/g008-competitive-intelligence-20261020T002610Z.md`  
**Architect review:** Cycle-3 final approval, as relayed by the Ralplan leader. The on-disk Architect review artifact is an earlier ITERATE and is superseded for this sequential review.

## Validation summary

- The plan keeps the exact `open-space` opponent and its games/data out of training, imitation, validation, and checkpoint selection; it explicitly includes it in the locked final suite. Split manifests, policy/data hashes, disjoint seeds/game conditions, validation-only promotion inputs, and fail-closed missing/overlap checks make this boundary operational.
- Checkpoint promotion is specified as paired per-opponent W/D/L scoring on common seed-seat conditions, seed-cluster bootstrap, one-sided decline testing with Holm-Bonferroni correction, a predeclared validation family, and explicit no-promotion behavior for missing coverage. Deterministic synthetic tests cover regression, ties, clustering, multiplicity, and incomplete pairs.
- The benchmark locks the G007 principal workload and numerator, distinguishes joint two-agent transitions from agent decisions, defines timing boundaries and warmups/repeats, and requires the median of five unprofiled runs to reach 10,000 joint transitions/s. The principal config hash (`0ec65b7978e26ecb13b893e76ec358d48c41428db96b0a60dd810f9f24f0e484`) matches the G007 benchmark artifact. The reference runtime is specified as Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, one Tesla T4; actual runtime must be recorded and a mismatch cannot be claimed as a gate pass.
- Optimization is bounded to a finite profile/low-risk/NumPy/at-most-one-Numba ladder, with oracle parity and repeatable end-to-end gains required. Rust/PyO3 and DDP are not assumed; no UI league, paid service, or billing change is in scope.
- Kaggle work has explicit GPU-hour, reserve, per-batch, wall-time, submission, and artifact-size caps plus checkpoint/run recovery rules. If any hard throughput, per-opponent win-rate, regression, or ONNX gate is missed, G008 remains **not approved** and reports measured results rather than redefining success.

## Deliberate review

- **Principle/option consistency:** Pass. Staged profiling is aligned with the measured rollout bottleneck; parity and end-to-end evidence precede backend adoption.
- **Alternatives depth:** Pass. The staged compiled candidate, immediate Rust/PyO3 port, and Python/NumPy fallback are compared with relevant performance, integration, and semantic-risk tradeoffs.
- **Risks and verification:** Pass. Leakage, validation overfit, simulator drift, quota exhaustion, statistical misapplication, and ONNX compatibility have concrete checks or stop rules.
- **Pre-mortem / tests:** Pass. A failed run cannot be promoted through missing provenance, incomplete paired coverage, parity mismatch, or final-test tuning. The plan calls for split/leakage, differential, synthetic bootstrap/multiplicity, schema compatibility, benchmark, and real browser-match validation.

## Conclusion

No material blockers found. Proceed to the approved G008 execution handoff and Stage 0 freeze. This review approves the plan, not the eventual results: all user-defined gates remain mandatory, and missing any gate requires reporting G008 as not approved.
