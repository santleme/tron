# Kaggle v6 recovery / retry diagnosis — 2026-10-20

## Exact execution state

- Kernel handle: `pascalcoimbra/circuit-ai/6`, source revision
  `g008-r2-173f821e2695d5d4`.
- Rechecked status after the user requested stop/retry: `KernelWorkerStatus.QUEUED`.
- `kaggle kernels logs pascalcoimbra/circuit-ai/6` returns no log content.
- Fresh `kaggle quota`: GPU 2.23h used, 27.77h remaining; no quota exhaustion.
- No other owned kernels were reported running by the read-only status sweep.
- `kaggle kernels output .../6` returned no files. `kaggle kernels files .../6`
  listed artifacts timestamped 2026-10-19 23:27 UTC (the pre-G008 G007 run),
  so those files are not accepted as v6 output or recovered G008 evidence.

## Stop attempt and retry boundary

- The installed Kaggle CLI exposes no `kernels cancel` command. The public
  Kaggle SDK has `cancel_kernel_session(kernel_session_id)`; an exact cancel
  request for session/version `6` was attempted after rechecking the queued
  state and returned HTTP 403: `Permission 'kernelSessions.cancel' was denied`.
  The same kernel remains queued. The SDK/endpoint is described in
  [Kaggle CLI issue #1172](https://github.com/Kaggle/kaggle-cli/issues/1172),
  but the account's current API authorization does not permit it.
- Deleting `pascalcoimbra/circuit-ai` would delete the existing kernel and its
  G007 artifacts, not just stop v6, so it was not used.
- Version 6 is the sixth pushed version on this kernel; the accepted G008 plan
  caps total submissions at six. A v7 retry would violate that cap. The current
  live v6 was not restarted or replaced.

## What failed / did not fail

- The submitted job has not reached worker startup; therefore there is no Python
  traceback, CUDA result, throughput result, or learning result to diagnose.
- A concrete submission omission was found: generated `kernel-metadata.json`
  has `enable_gpu: true` but `machine_shape: ""`. Current Kaggle metadata docs
  identify `machine_shape` as the accelerator selector and name
  `NvidiaTeslaT4` for T4 ×2. This omission means the requested accelerator was
  not explicit; it is a plausible configuration failure, but queue status and
  empty logs do not prove it caused this queued state.
- A prior local `prepare` attempt recursively copied its own temporary directory
  because the destination was nested under `training/`. That packaging bug was
  fixed by excluding `.kaggle.prepare-*`; the final v6 package prepared and
  submitted successfully, and the full local suite passed (127 passed, 4 skipped).
- This is currently a Kaggle queued-session/cancellation limitation, not a
  demonstrated trainer or CUDA failure. No new submission was attempted because
  the exact session could not be cancelled and the six-submission cap is reached.
