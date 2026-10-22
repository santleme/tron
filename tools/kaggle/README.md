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
CUDA only). PPO Kaggle training is GPU-required: an explicit CPU config or a
session without CUDA fails before dependency installation/training. A separate
`simultaneous_search` mode runs tournaments on CPU without GPU or internet
access. Local development still supports CPU through `training.ppo.train`.
Example PPO config:

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
equal to the exact `owner/slug`, a `.arena-source-revision` marker, and a
`source-manifest.json`. The manifest lists exact staged paths and SHA-256
values, Kaggle requirements and metadata, package-builder and runner-template
hashes, a Git commit when available, dirty-tree status, and the explicit
source revision label. Its digest is included in runtime evidence; the
registered package tree hash binds the complete submitted kernel.
Historical packages remain verifiable by their recorded package hashes even
when they predate readable source manifests. Validate the exact source without
network access before submission:

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
invoke the Kaggle CLI, credentials, or network.

## Simultaneous-search tournaments

An isolated run config with `run_id` and `simultaneous_search` stages only the
canonical arena and `training/search`. Opponent and canonical-engine bindings
remain checked. Exploratory runs use Kaggle's actual notebook session, disk,
and compute quotas, and set both `enable_gpu` and `enable_internet` to false
because these search workloads are CPU-bound. Tournament downloads use the
search-specific output validator, which checks report identity and digest,
listed file hashes, matchup roster, game identities, and summaries. They do
not enter the PPO run artifact or G008 registry.

Optional `agent_edge_budgets` calibrates logical work per planner, while
`matchups` selects a unique subset from the permitted planner/heuristic roster.
Use measured decision latency to set these budgets for CPU-matched follow-up
comparisons; the report continues to record actual decision and process time.

Optional `workers` selects worker processes (default 4, capped at runtime by
the container's CPU affinity and cgroup quota). Independent games run in separate
processes because the planners are CPU-bound Python; results are collected in
the same matchup/seed/seat order and each seed's two seat-swapped games remain
an atomic scheduling batch. The report records the effective worker count.
This improves throughput without changing planner decisions or canonical game
transitions; Kaggle measurements are still required before claiming a speedup.

### Fresh fixed-depth versus open-space assessment

Generate a unique run ID and 256-seed plan from the frozen candidate manifest,
then prepare and submit the isolated 512-game assessment. It uses the exact
canonical open-space policy code, both candidate seats for each seed, the
native build of the same Rust source as the browser WASM, and replay validation
against `lc-engine/1`. Its result is nonofficial and never enters G008.

```bash
assessment_dir="artifacts/circuit-ai/experimental/independent-open-space/assessment-$(date +%s)"
python3 -m training.search.create_independent_eval_config --output "$assessment_dir"
run_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$assessment_dir/run-config.json")"
python3 -m tools.kaggle kaggle prepare --kernel-id owner/independent-open-space \
  --source-revision "$run_id" --run-config "$assessment_dir/run-config.json" \
  --output-dir "$assessment_dir/package"
```

Submit with the generated source revision and explicit `--confirm`; then use
version-qualified `status` and `download`. The output validator checks all 512
games, seat-swapped seed pairs, the Wilson 95% interval, candidate and opponent
hashes, source manifest, and replay digests/transitions.

Optional `native_territory: true` compiles the dependency-free Rust articulation
count kernel locally while preparing the package, then embeds that exact shared
library in the source manifest. Kaggle remains offline and does not need a Rust
toolchain. The Python path remains available if the native library cannot load;
the run report records whether it loaded successfully. Only enable it after a
same-seed Kaggle calibration confirms exact replay parity and end-to-end gain.

Optional `native_fixed_depth: true` builds the viewer's Rust simultaneous
planner as a native shared library during local preparation and adds explicit
`native-fixed-depth` matchups to the bounded roster. It requires local `rustc`,
and Kaggle still runs CPU-only/offline. The run report records the flag; include
the native agent only in selected matchups so baseline Python comparisons remain
available in the same host session.

An explicitly pinned Policy-Value checkpoint can be selected with
`policy_value_checkpoint` and compared against the registered planners. The
checkpoint is mounted from a private Kaggle dataset, checked against the
SHA-256 allowlist in `training/search/tournament.py`, and loaded on CPU through
the matching v2 model implementation. The search package remains offline and
GPU-disabled. The model is a standalone policy opponent; its value head is not
used by the planner in this matchup.

### Frozen v15 step-236 opponent

The permitted experimental opponent `v15-step236` is bound to the private Kaggle
dataset `pascalcoimbra/circuit-ai-v15-step236-experimental`. `prepare` verifies
the local checkpoint against its pinned SHA-256 before writing the kernel
metadata, but does not copy the checkpoint into the kernel source or its embedded
source archive. This keeps the generated Python source below Kaggle's source
size limit. The kernel mounts the dataset at
`/kaggle/input/circuit-ai-v15-step236-experimental/`; the runtime verifies the
mounted checkpoint bytes against the same pinned digest before inference.

Create/update the private dataset separately with Kaggle's official CLI. Its
directory should contain the file `step-236.pt` and a `dataset-metadata.json`
whose `id` is `pascalcoimbra/circuit-ai-v15-step236-experimental` and whose
`isPrivate` field is `true`. Before upload, compare the file's SHA-256 with the
pin in `training/search/tournament.py` and record the resulting dataset version
in the experiment manifest. For a new dataset directory, Kaggle's CLI can
generate the metadata template; then create it privately:

```bash
kaggle datasets init -p /path/to/v15-step236-dataset
# Place the verified step-236.pt in that directory and set the exact id/private fields.
kaggle datasets create -p /path/to/v15-step236-dataset
```

For an existing dataset, use `kaggle datasets version -p
/path/to/v15-step236-dataset -m "refresh frozen v15 step-236 artifact"` after
the same hash and metadata checks. Kaggle CLI defaults newly created datasets
to private unless `--public` is supplied. Record the resulting dataset version
with the experiment evidence; `dataset_sources` binds the dataset handle, while
the runtime SHA check binds the checkpoint bytes. Dataset creation/versioning
is a separate explicit Kaggle operation; preparing a kernel only records the
source binding.

## Policy-Value distillation experiments

Policy-Value work uses two isolated, non-G008 run-config modes. They cannot be
combined with PPO, search tournaments, benchmarks, or each other. Dataset
generation mounts only `training/arena` and `training/search`, disables GPU and
internet because teacher generation is CPU search, and sizes work to Kaggle's
actual 20 GB working storage and 12-hour notebook session. Boards and work
budgets follow the planner ABI and storage needs. Run IDs and seed manifests
are generated automatically; protected evaluation partitions remain isolated.

Example CPU/offline teacher-dataset config:

```json
{
  "run_id": "pv-distill-dev-01",
  "policy_value_data": {
    "experiment_id": "pv-distill-dev-01",
    "seeds": [7000000000000001, 7000000000000002],
    "width": 12,
    "height": 12,
    "edge_budget": 56,
    "max_depth": 3,
    "heuristic_styles": ["cruise", "local-mobility", "interceptor"],
    "max_wall_seconds": 3600
  }
}
```

The downloaded `examples.npz` is paired with `manifest.json`; both include
SHA-256 bindings and development-only provenance. Upload that output as a
private Kaggle dataset, then pin its exact owner/slug, dataset filename, NPZ
hash, manifest filename/hash, and experiment ID in a separate training config.
The training package stages only `training/search/train_policy_value.py`, its
hashing protocol, and the Kaggle ONNX requirements. It enables a T4 GPU and
uses only the explicitly mounted private dataset; the runtime verifies the
manifest and NPZ hashes before training and respects available disk, session,
and free Kaggle GPU quota. CUDA absence is a hard failure with
no CPU fallback. Checkpoints, ONNX, report, and all hashes are validated before
`download` atomically promotes output, and an existing destination is never
overwritten. Submission still goes through the exact `prepare` / `submit
--confirm` / versioned `status` / `download` lifecycle above.

Example GPU training config (replace both hashes with the exact downloaded
dataset bytes):

```json
{
  "run_id": "pv-train-dev-01",
  "policy_value_train": {
    "dataset_source": "pascalcoimbra/circuit-ai-distill-dev-01",
    "dataset_experiment_id": "pv-distill-dev-01",
    "dataset_file": "examples.npz",
    "dataset_manifest_file": "manifest.json",
    "dataset_sha256": "<64 lowercase hex characters>",
    "dataset_manifest_sha256": "<64 lowercase hex characters>",
    "epochs": 20,
    "batch_size": 256,
    "hidden_size": 128,
    "learning_rate": 0.0003,
    "value_weight": 1.0,
    "validation_fraction": 0.1,
    "seed": 20261022
  }
}
```

`policy_value_data` and `policy_value_train` have dedicated download validators;
their artifacts do not enter PPO run validation or any official registry.
