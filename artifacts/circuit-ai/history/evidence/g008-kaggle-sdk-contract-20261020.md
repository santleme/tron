# Kaggle CLI and SDK contract evidence — 2026-10-20

Read-only inspection of the pinned official PyPI source distributions. No Kaggle
API call or kernel mutation was made. Retrieval/inspection date: 2026-10-20 UTC.

| Distribution | Version | Source archive SHA-256 | Official source |
|---|---:|---|---|
| Kaggle CLI | 2.2.4 | `28a58f6c1b9f46ce2afdfb439e0c4e91219e818b377ea4254f4e479456d8cb78` | [PyPI](https://pypi.org/project/kaggle/2.2.4/), [sdist](https://files.pythonhosted.org/packages/8d/22/9e4d5015aad7c7794af098e3def53a26a7deae3ee2e348c129bc2446260e/kaggle-2.2.4.tar.gz) |
| Kaggle SDK | 0.1.35 | `7373c46e7546dc983e1cd87fb7555906f799869b6f02a747a3121c89edc420b8` | [PyPI](https://pypi.org/project/kagglesdk/0.1.35/), [sdist](https://files.pythonhosted.org/packages/47/3f/2b7b46e8c17046261b138f22c5b7421bbbb85cf6a8f69f4aa43fe983eb1a/kagglesdk-0.1.35.tar.gz) |

## Observed contracts in the pinned bytes

- CLI `kernels push --timeout N` forwards integer `N`; SDK request field is
  `session_timeout_seconds` (wire field `sessionTimeoutSeconds`). `kernels_push`
  returns `ApiSaveKernelResponse.version_number` (`versionNumber`), which is the
  assignment evidence. Missing/zero assignment is not success.
- SDK `get_kernel` metadata has optional `current_version_number`. Request the
  exact owner and slug and validate a positive version; the public source does
  not prove server semantics beyond the generated field shape.
- CLI `kernels list` rendering omits `current_version_number`; it is not an
  assignment or current-version authority.
- CLI `kernels status owner/slug/version` ignores the parsed version when it
  builds the status request: it sets only owner and slug, not `version_label`.
  It queries latest status. Use it only with exact-kernel current-version
  metadata immediately before and after each poll plus the runtime exclusive
  writer controls and actual-preflight no-outside-writer confirmation.
- SDK documentation says `session_timeout_seconds` terminates the session after
  that many seconds of runtime and must be below Kaggle's global maximum. The
  pinned source does not state a numeric global maximum or whether queue time is
  included. CLI serializes only truthy timeout values; zero/omission is invalid
  for a registered attempt. Project bounds remain 3,600s benchmark / 7,200s
  training, subject to fail-closed positive finite integer validation.
- A timeout is not proof that compute stopped. Keep the full reservation until
  recognized terminal status is observed.

## Source locations

CLI source: `src/kaggle/api/kaggle_api_extended.py` around 6319–6333,
6447–6494, and 6778–6796; CLI parser around 1312–1317. SDK source:
`kagglesdk/kernels/types/kernels_api_service.py` around 2015–2017, 2325–2339,
2675–2702, 2731, and 2812–2823. Hashes above were checked against the official
PyPI records and downloaded archive bytes. These are implementation facts from
the pinned clients; they do not guarantee server behavior for version-selective
fields.

## Required adapter tests before registration

1. Assert registered timeout is a positive finite integer within the workload
   cap and is forwarded unchanged through the project push adapter into
   `session_timeout_seconds` for either workload kind.
2. Assert one push per attempt ID; persist a durable dispatch marker immediately
   before calling the SDK; accept only the exact positive response assignment,
   corroborated by exact-kernel metadata before polling.
3. Assert `get_kernel` is called with exact owner/slug, metadata is bracketed
   around every latest-status poll, only enumerated terminal statuses resolve,
   and version drift or unknown/ambiguous responses leave the attempt pending.
4. Assert no status/set/list-difference inference establishes assigned version;
   crash/lost-response after dispatch keeps the full reservation and blocks
   subsequent events until exact reconciliation.
5. Assert timeout deadlines are monotonic timeout plus at most ten minutes of
   reconciliation slack, and timeout does not resolve an attempt without
   terminal evidence.

The numeric Kaggle global timeout maximum is not established by these artifacts;
do not invent a value. Enforce the lower, reviewed project caps and fail closed
on any server rejection or contradictory response.
