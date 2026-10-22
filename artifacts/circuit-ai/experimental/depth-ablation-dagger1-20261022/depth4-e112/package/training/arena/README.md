# Python canonical arena v1

The engine has no runtime dependencies and is deliberately headless. `step`
resolves a joint action against one immutable pre-tick state and returns a
`Transition` with per-agent causes and destinations. Observations follow
`schemas/README.md`'s row-major `[height][width][12]` layout.

`initialize_episode(width, height, master_seed, arena_index)` derives the three
v1 streams from `training.arena.seeds`: the episode stream chooses two distinct
row-major cell IDs uniformly via rejection-sampled `next_bounded(width*height)`
(agent 0 first, retry repeats for agent 1); independent `agent/0` and `agent/1`
streams each choose one heading via `next_bounded(4)`. There are no adjacency
exclusions. This exact initialization order is part of Python/TypeScript parity.

`VectorArena` owns N independent states and seed triplets. Its pure-Python batch
step is a headless vector runner (not NumPy/GPU vectorization); it applies the
same scalar rules independently and intentionally introduces no dependency
until throughput measurements justify one.
