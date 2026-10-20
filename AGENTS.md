# Project Agent Rules

## Parallel work

- For every project task, spawn at least two AI agents and parallelize independent analysis, implementation, or review slices. Assign each agent a distinct, bounded deliverable; keep integration and final verification in the primary agent. For small tasks, parallelize a second-opinion review or independent verification. If native agent support or available slots prevent this, respect the runtime limit and report the constraint rather than fabricating delegation.
- Use the native agent role that fits each slice. Do not delegate two agents to edit the same files at once; identify file ownership and reconcile their findings before completing the task.

## Compute and training

- Run all model training and genuinely compute-heavy workloads on Kaggle. This includes target-scale throughput benchmarks, long-running simulations, and GPU jobs. Treat Kaggle as the project's canonical training and heavy-compute environment.
- Use local or CI environments for code changes, static checks, deterministic unit tests, and lightweight correctness checks that do not train a model or claim hardware throughput. Keep correctness smoke checks separate from training runs.
- Before a Kaggle run, check the approved workload, remaining compute budget, and submission limits. Follow `tools/kaggle/README.md`, including its explicit `--confirm` submission gate; do not bypass the project's submission lifecycle. Record the Kaggle image, Python/PyTorch/CUDA versions, GPU model, run configuration, and output evidence. Never report local CPU measurements as Kaggle/T4 performance.
- If Kaggle is unavailable or the approved budget/limits prevent the requested training, do not silently move training to local hardware. Complete independent non-training work and report the blocker and evidence needed to resume.
