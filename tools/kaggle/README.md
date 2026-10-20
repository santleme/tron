# Explicit Kaggle lifecycle

All operations are local CLI invocations; importing the viewer, building the
app, or running tests never contacts Kaggle. Kaggle authentication remains the
official CLI's responsibility and is only accessed when a real command runs.
Install/configure Kaggle CLI using its official instructions. Never place
credentials in this repository or output artifacts.

## Prepare and submit

`prepare` assembles a deterministic, self-contained source tree from `training/`
and `schemas/`; it does not import Kaggle credentials or contact Kaggle. By
default the output is `training/kaggle/`. Use `--output-dir` to select a
different package location. A config file can select an artifact run ID and
PPO settings; otherwise it uses `kaggle-run` with `device: auto` (resolved to
CUDA only). Kaggle training is GPU-required: an explicit CPU config or a
session without CUDA fails before dependency installation/training. Local
development still supports CPU through `training.ppo.train`. Example config:

```json
{"run_id":"kaggle-run-r1","train_config":{"device":"auto","arena_count":32,"total_updates":10}}
```

The generated `run_kaggle.py` writes run artifacts under
`/kaggle/working/arena-output/<run_id>`, which Kaggle collects as kernel output.
It reads the image's existing PyTorch/CUDA versions first, requires CUDA, and
never installs PyTorch. If ONNX/schema support packages are absent, it installs only
`training/requirements-kaggle.txt` with `pip --no-deps`, then verifies the
preinstalled PyTorch/CUDA version did not change. Kaggle internet is enabled
only to permit that optional package install; no external data sources are
configured.

```bash
python3 -m tools.kaggle kaggle prepare --kernel-id owner/light-cycles \
  --source-revision local-r7 --run-config training/kaggle-run-config.example.json
```

The example config is checked in and valid as-is; copy it and edit `run_id`
and `train_config` for each run if desired. `prepare` never overwrites it.

The generated package contains Kaggle's `kernel-metadata.json` with `id`
equal to the exact `owner/slug` and a `.arena-source-revision` marker. The
revision is intentionally user-set: this greenfield source may be a Git SHA
or an explicit local revision label. Validate the exact source without network
access before submission:

```bash
python3 -m tools.kaggle kaggle submit --kernel-id owner/light-cycles \
  --source-revision local-r7 --source-dir training/kaggle --dry-run
```

Only the following explicit command crosses the submission boundary:

```bash
python3 -m tools.kaggle kaggle submit --kernel-id owner/light-cycles \
  --source-revision local-r7 --source-dir training/kaggle --confirm
```

It verifies metadata and the revision marker, checks that the selected kernel
is not already running, then calls `kaggle kernels push -p <source>`. A missing
`--confirm`, mismatch, running target, or dry-run makes no remote mutation.
Kaggle CLI output is not copied into an artifact or augmented with auth data.
Record the version reported by Kaggle for later explicit selection.

## Observe and download

`status` and `download` require an exact kernel and version. Polling is bounded
to 30 attempts at 10 seconds by default and can be adjusted within safe limits.
The adapter uses version-qualified Kaggle kernel IDs, e.g. `owner/slug/5`.

```bash
python3 -m tools.kaggle kaggle status --kernel-id owner/light-cycles \
  --version 5 --max-attempts 30 --interval-seconds 10
python3 -m tools.kaggle kaggle download --kernel-id owner/light-cycles \
  --version 5 --run-id run-2026-01 --output-dir artifacts/run-2026-01
```

Download starts only after the selected version reports success. Kaggle output
is written into a temporary sibling staging directory; `manifest.json`, run
and evaluation schemas/digests, all manifest-listed file hashes, and linked
replay schemas/digests/identities are validated before an atomic rename. An
existing destination is never overwritten. Invalid/partial staging is removed
on failure; no incomplete final run directory is promoted. `jsonschema` is
required for artifact validation (included in `training/requirements-kaggle.txt`).

The adapter calls the supported `kaggle kernels push`, `status`, and `output`
commands via argument arrays (no shell). Tests inject a fake process and never
invoke the Kaggle CLI, credentials, or network. Actual submission/download is
external and deliberately not exercised as part of development.
