# PPO self-play and fixed evaluation

The canonical simulator is `training.arena`; training uses a shared actor/value
MLP and runs the two seats against the latest copy of that policy. The PPO
optimizer consumes batched arena observations on the selected PyTorch device.
The headless reference engine remains Python/CPU work; CUDA use is recorded as
the number of actual optimizer minibatch updates on CUDA, not inferred from
device availability. The metrics record also includes software, device,
optimizer update count, CPU core count, wall elapsed time, update-only
environment steps/s, and per-update rollout, optimizer, checkpoint, and
end-to-end latency. The primary throughput metric excludes checkpoint saves;
`wall_environment_steps_per_second` includes in-loop checkpoint I/O. PPO loss
scalars stay on-device until an update ends to avoid per-minibatch CUDA syncs.
Each update also records sample-weighted policy loss, value loss, entropy,
approximate KL, and clip fraction. Rollout reward and completed-game W/D/L,
terminal reward, reward return, and duration are split by sampled opponent
(`latest`, `snapshot:step-0`, or each heuristic style). Durations and episode
returns include full games that may span multiple rollouts; the record also
counts unfinished partial games at rollout end.

## Local reproducible smoke

Python 3.10+ is recommended. In an isolated local environment install the
compatible dependency ranges with:

```bash
python -m pip install -r training/requirements.txt
python -m training.ppo.train --output artifacts/smoke --no-onnx \
  --config /tmp/lc-smoke.json
```

For a tiny CPU smoke, `/tmp/lc-smoke.json` can contain:
`{"width":6,"height":6,"arena_count":2,"rollout_steps":8,"total_updates":1,"update_epochs":1,"minibatch_size":16,"hidden_size":16,"device":"cpu","checkpoint_every":1}`.
Omit `--no-onnx` to export ONNX (opset 17) and emit paired checkpoint
metadata plus `manifest.json`. `total_updates` is a global target, not an
additional count on resume: resuming step 4 with `total_updates: 10` runs six
more updates. A checkpoint already at or past the requested target is rejected.
`--resume <checkpoint.pt>` restores model, optimizer and recorded RNG state;
pass the same architecture dimensions and original run directory (which must
contain the immutable step-0 checkpoint). Native checkpoints carry the
optimizer and current global update counter.

Self-play defaults to latest-vs-latest with no historical opponent inference
overhead. To include a frozen initial-version opponent, set
`snapshot_opponent_probability` to a value in `[0,1]` in the JSON config, e.g.
`0.25` samples the step-0 checkpoint as opponent for roughly one quarter of
arenas in each rollout. Sampling is per arena/rollout from the checkpointed
Python RNG state; the current policy is trained and seat assignments alternate
to balance seat exposure. Per-update records and the run manifest record the
opponent sampling mode, probability, snapshot checkpoint ID, and sampled arena
counts. This is a bounded initial-snapshot mixture, not a league.

`TrainConfig.action_mask_version` freezes immediate-move semantics into both
checkpoint metadata and ONNX. Version 1 masks only reversal. Version 2 also
masks a wall or occupied-trail destination when at least one forward safe move
exists; if trapped, it keeps the finite v1 forward distribution. This mask
cannot predict simultaneous collisions with the opponent. Compare versions as
separate registered candidates with identical training and validation settings.

G008 may opt into a heuristic-opponent fraction only with a frozen
`g008-train` environment schedule. Its built-in training styles (`cruise`,
`local-mobility`, and `interceptor`) use simple local rules and never call the
flood-fill `open-space` policy. G008 training also rejects the legacy G007
evaluator, which would otherwise consult the protected open-space holdout; use
the split-aware validation path for checkpoint selection and reserve the exact
open-space policy for final evaluation only.

## G008 failure diagnosis

Registered G008 evaluation classifies each loss during its original 512-game
run on Kaggle. The compact `g008-evaluation.json` contains per-loss terminal
cause, legal moves before the fatal action, reachable area, and whether a
one-step alternative survives while holding the opponent's observed action
fixed. Its summary aggregates categories by cause and seat. The evaluator keeps
at most eight representative replays and never recreates hundreds of matches
locally. One-step counterfactuals are diagnostic signals; they do not prove that
the opponent would make the same move after a different action or that the
alternative guarantees a win.

The optional `training.eval.g008_failure_diagnostics` command summarizes an
already-produced report. It does not simulate games or create replay files.

## Throughput sweep

Use the user-invoked sweep to compare at most five candidate
`arena_count`/`rollout_steps`/`minibatch_size` combinations on one selected
device. It runs a fixed 1–10 update count per candidate (default 3), writes
each trial's native checkpoints/training metrics (evaluation and ONNX export
are omitted to avoid unmeasured work) plus a machine-readable
`benchmark-results.json`. Each candidate first runs one discarded warmup
update, then three measured repeats at the selected fixed update count. It
selects the highest median update-only environment steps/s (median PPO update
latency breaks ties). The report includes raw repeats, medians, wall-inclusive
steps/s as a diagnostic, separate rollout and optimizer durations and their
ratio, actual device and optimizer/CUDA update counts plus the selection
rationale. At most 20 short
training jobs run in a sweep; use a new/empty output directory to avoid
overwriting previous evidence.

```bash
cat > /tmp/lc-benchmark.json <<'EOF'
[
  {"arena_count": 32, "rollout_steps": 64, "minibatch_size": 256},
  {"arena_count": 64, "rollout_steps": 64, "minibatch_size": 512}
]
EOF
python -m training.ppo.benchmark --configs /tmp/lc-benchmark.json \
  --output artifacts/benchmark --device auto --updates 3
```

All candidates use the same device and update count; the runner is explicitly
single-device because DDP/multi-GPU execution is not implemented. It does not
compare one vs two GPUs or use occupancy as a selection criterion. The sweep
measures rollout+PPO update throughput, excluding checkpoint I/O from its
selection metric, and reports wall-inclusive throughput separately; its
short-run result is a local configuration comparison, not a universal speedup
claim.

Run the fixed evaluation from a pair of native checkpoints after training:

```bash
python -m training.eval.lc_eval --target artifacts/run/checkpoints/step-10.pt \
  --step-0 artifacts/run/checkpoints/step-0.pt --output artifacts/run/evaluation \
  --run-id my-run --checkpoint-id step-10
```

This writes one `lc-replay/1` replay for every held-out seed/seat and matchup,
plus an `lc-eval/1` report with W/D/L, win-rate Wilson 95% interval, survival
statistics, collision causes, and all seed/seat records. Win rate is wins / 64;
draws are separate from wins. Evaluation does not claim
that a model improved.

## Kaggle CUDA workloads

Registered G008 benchmark, training, validation, and final-evaluation packages
are built with `tools.kaggle/cli.py`. The package freezes the run config and
source, selects `cuda:0`, records visible GPUs separately from the device used,
and reconciles downloaded reports and checkpoints against the registry. Kaggle
can generate the match replays and classify losses inside the original
simulation; bulk replay files do not need to be copied to the local machine.

Use the registered runner for G008 experiments. A Kaggle kernel showing
`COMPLETE` is not by itself a reconciled experiment result; its exact version,
report, manifest, and checkpoint hashes must match the local registry.

For an unregistered smoke run, in the selected Kaggle notebook image first
record `python --version`,
`python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")'`
and `nvidia-smi`. Do not pip-reinstall PyTorch if the notebook image already
has a compatible CUDA build. Run a tiny job with `device: "cuda"`, one update,
two or more arenas, and export enabled; inspect `training-metrics.json` and
confirm `device.device == "cuda"`, `optimizer_updates > 0`, and
`cuda_optimizer_updates == optimizer_updates`. Multi-GPU/DDP comparison is not
implemented. This smoke path is separate from registered G008 results.

In the selected Kaggle notebook image, first record `python --version`,
`python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")'`
and `nvidia-smi`. Do not pip-reinstall PyTorch if the notebook image already
has a compatible CUDA build. Run a tiny job with `device: "cuda"`, one update,
two or more arenas, and export enabled; inspect `training-metrics.json` and
confirm `device.device == "cuda"`, `optimizer_updates > 0`, and
`cuda_optimizer_updates == optimizer_updates`. Then run the bounded throughput
sweep on that selected single device and record environment steps/s, end-to-end
update latency, rollout duration, optimizer duration, their ratio, and CUDA
update counts. Multi-GPU/DDP comparison is not implemented. This repository has not
submitted a Kaggle job or verified remote hardware; T4 allocation, quotas and
image package versions are external checks.

Kaggle should use `training/requirements-kaggle.txt` only if ONNX/schema
dependencies are missing; it intentionally omits PyTorch to preserve the
notebook image's CUDA-enabled build. The run manifest records the actual
Python, PyTorch, CUDA, exporter and opset versions, so inspect that metadata
instead of assuming a specific Kaggle image version.

The default workspace Python has no `torch`, `onnx`, or CUDA. CPU PPO smoke,
ONNX export/checker, run bundle, and browser WASM playback were independently
verified in an isolated temporary environment using PyTorch 2.7.1+cpu and ONNX
1.23.2. Kaggle jobs have also run; use the experiment registry to distinguish
completed but rejected artifacts and exploratory runs from fully reconciled
CUDA evidence.
