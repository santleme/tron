# G008 Kaggle kernel timeout contract

**Checked:** 2026-10-20. This note records official client/API documentation, not a Kaggle submission or an empirical test of server-side validation.

## Official contract

- Kaggle's [Notebook technical specifications](https://www.kaggle.com/docs/notebooks#technical-specifications) state a maximum execution time of **12 hours for CPU and GPU notebook sessions** and **9 hours for TPU sessions**. They also say Save & Run All must finish within those limits. The planned kernel uses a GPU, so the documented maximum is 12 hours (**43,200 seconds**).
- The official [`kagglesdk` v0.1.37 request type](https://github.com/Kaggle/kagglesdk/blob/ae99c895ac3d5cfb80436be8e426c0bc658c5892/kagglesdk/kernels/types/kernels_api_service.py#L2324-L2339), released 2026-08-06, documents `session_timeout_seconds` as terminating the kernel after the supplied runtime and says the value “must be lower than the global maximum.” Its generated setter checks that the value is an integer but does not define a numeric maximum.
- The official [Kaggle CLI `kernels push` reference](https://github.com/Kaggle/kaggle-cli/blob/6894864ff4095cf98e9e36d3a694ff5fc9ba3818/docs/kernels.md#L286-L310) describes `--timeout <SECONDS>` as maximum run time in seconds. In the corresponding [CLI parser](https://github.com/Kaggle/kaggle-cli/blob/6894864ff4095cf98e9e36d3a694ff5fc9ba3818/src/kaggle/cli.py#L1417-L1424), the value is parsed as an integer; the [push implementation](https://github.com/Kaggle/kaggle-cli/blob/6894864ff4095cf98e9e36d3a694ff5fc9ba3818/src/kaggle/api/kaggle_api_extended.py#L7270-L7280) assigns it to `request.session_timeout_seconds` without a local numeric range check. The source commit is dated 2026-10-16; the latest released CLI inspected was v2.2.4, released 2026-07-23.

## G008 timeout assessment

| Workload | Requested timeout | Comparison to GPU maximum | Assessment |
| --- | ---: | ---: | --- |
| Paired benchmark | 3,600 s (1 h) | 3,600 < 43,200 | Within documented bound |
| PPO training | 7,200 s (2 h) | 7,200 < 43,200 | Within documented bound |

Both requested values satisfy the SDK's strict “lower than the global maximum” condition and are below Kaggle's documented 12-hour CPU/GPU execution maximum.

## Caveat

The CLI and generated SDK do not encode a numeric range for this field. Kaggle's public docs establish the notebook execution maximum, but do not expose the `save_kernel` endpoint's backend validator implementation. Thus the field-specific server validation has **not** been empirically tested here; this assessment is based on the public global runtime limit and the official client's strict-below contract. No authenticated endpoint was queried and no kernel was submitted for this note.
