# External spatial checkpoints for search tournaments

These are exact checkpoint copies from the separate spatial-imitation worktree,
staged only for isolated experimental tournaments. The source worktree and its
files remain untouched. Each checkpoint is bound by SHA-256 in
`training/search/tournament.py` and copied into a private, CPU-only Kaggle
kernel package only when selected explicitly in that run's config.

- `dagger1-selected-20261021`: selected failure-focused DAgger1 incumbent,
  checkpoint SHA-256 `f01978c5ba9edd4f7d544a91115f3739d22c543731d90d3cdc4ea0d96f9a96af`.
- `spatial-e10-selected-20261021`: selected earlier spatial CNN baseline,
  checkpoint SHA-256 `e10d44869ed2a9959655ffb0cbacddcfa422a8e06baa31efde697e05c63a481c`.

Both were selected on development heuristics and are permitted only for
development comparisons. They are not official candidates; no evaluation
seeds or training datasets are included here.
