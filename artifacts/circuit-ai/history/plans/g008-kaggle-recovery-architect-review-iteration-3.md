# G008 Kaggle recovery — Architect review, iteration 3

**Verdict: ITERATE**

The count-free retry rule is appropriately bounded. Clarify three audit details before approval:

1. A post-lock `retry_attempt` must be validated against the original registration and prior attempt: same workload kind, run UUID, config, seed/schedule, roots/opponents and bounds; prior attempt terminal; unique attempt identity; artifact/status linked to the exact attempt. The original registration never changes. Only source/package hashes may differ in a corrected-package event, which must include a patch hash and deterministic validation evidence. The validator must reject any other changed field.
2. Define the final pre-lock anchor chronology. Any early prefix snapshot is provisional. After all benchmark retries, write the immutable `benchmark_artifact` anchored to the exact prior registry head and exact successful attempt; immediately append `registration_closed` using that same head/index. The resulting lock hash then anchors post-lock attestation and training evidence.
3. The report must include successful attempt ID, Kaggle version/status and exact source/package hashes. `benchmark_artifact` validation must compare those fields with that attempt's `retry_attempt` event; post-lock attestation must bind the unchanged report hash and original prefix anchors to the lock.

**Antithesis:** Append-only event hashes and artifact hashes already make the record tamper-evident.
**Tension:** Hashes prove bytes, but do not define attempt eligibility, permitted field changes or the final prefix boundary.
**Synthesis:** Keep count-free, cumulative-budgeted retries and explicitly validate event chronology, frozen registration fields and the successful attempt/report binding.

This is planning evidence only; no implementation, tests, training or Kaggle submission was authorized.
