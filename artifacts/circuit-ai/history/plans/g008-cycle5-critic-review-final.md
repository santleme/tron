# G008 Cycle-5 — Critic review, final bytes

**Verdict: APPROVE (planning only).** Reviewed exact hashes:

- Plan: `0a9e1ebccae2a793fadf531fb99013a9d19ff832a5880de9ee15204b477ca81a`
- Test spec: `e3b8efef4b441d694883a856164ed3d25f2b278f2bd7ca9ee30dfb8097b3ba6c`
- Cycle-5 draft: `9150f2bb8cddf326d860752fd0e1352c9d6b55b2aadaae60f55eef2dab23cafe`
- Seed plan `/3`: `c6b16733a045e94e7310183e8be8a62cf4327b5d2bdc3eed40bb9d552ce17593`

The seed plan preserves train/imitation/validation/final partitions and holdouts, separates imitation initial/reset domains, and binds imitation run keys to canonical UUIDs. Test-spec lines 18–20 require exact successor SHA pinning and rejection of `/1`, incomplete `/2`, wrong version/hash and missing/stale bindings at applicable consumers. Lines 26–28 specify terminal-status gating, pre-lock benchmark-only order, successful artifact anchors, immediate artifact→lock adjacency, post-lock training, and attestation before downstream use.

The ordinary G008 six-submission cap remains. The separate recovery authority scopes the no-count-ceiling exception to exactly two named recovery registrations and preserves the cap for other campaigns.

**Strongest counterargument:** a pre-lock artifact cannot bind the future lock at creation. Immutable content/prefix anchors, immediate artifact-to-lock adjacency, and separate post-lock attestation address this limitation.

No code, tests, or Kaggle operations were performed.
