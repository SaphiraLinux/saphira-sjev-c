# Multithreaded trainer (`--threads N`)

Trainer only. Model maths, objective, file format, CLI behaviour (apart from
the new flag) and the inference path are unchanged.

## Design (pthreads, no OpenMP)

Per-batch fork-join with persistent per-worker gradient buffers:

- `train_loop(..., nthreads, ...)` creates one `sjev_thread_pool` holding
  `nthreads` private `sjev_grad` buffers, allocated once per run.
- Each batch `[start,end)` is split into contiguous chunks (first workers
  take the remainder; deterministic). `sjev_pool_run_batch` spawns N threads
  (also for N=1 — same code path), each running the existing `forward_one` /
  `backward_one` sequentially over its disjoint examples into its private
  grad plus a private loss sum.
- Main thread joins all workers, then reduces worker grads in stable
  worker-index order with plain float loops (no atomics, no locks), sums
  losses in index order, and runs the existing `grad_scale(1/blen)` →
  `grad_clip(1.0)` → `adam_step` exactly once.
- Model weights are read-only in workers; the single AdamW update happens
  after the join. `forward_cache` temporaries are malloc'd per example and
  never shared. Worker arg structs are tiny and read-mostly; gradient
  buffers are separate multi-hundred-KB calloc regions, so there is no
  false sharing.
- Reduction is deterministic for a fixed thread count. 1-thread vs N-thread
  runs differ only by FP32 summation order (see `tests/test_threads.sh`
  for the parity procedure).

Known simplifications (measured, not assumed): threads are spawned/joined
per batch (negligible next to batch compute); static contiguous chunking
can stall fast workers behind stragglers when examples vary in cost;
per-example malloc/free is unchanged from the serial trainer.

## CLI

`sjev train ... --threads N` (default 1, valid 1..1024, kebab-case like
`--batch-size`). Invalid values fail cleanly before loading data.
`sjev grad-check ... [--threads N]` runs the identical pool path over the
whole file as one batch (logits use a separate deterministic forward-only
pass, so they are thread-independent by construction).

## Memory per worker (W=64, R=64, C=256)

Persistent per worker: one full gradient buffer =
257·W + C·W + 4·W + 3·R·W floats = 45,376 floats ≈ **177 KiB**.
Transient workspace is malloc'd/freed per example (unchanged); with T
threads up to T examples are in flight (~0.5–1 MB each at these dims).

## Benchmark (i9-13900K)

On the 13th-gen hybrid reference machine (i9-13900K: 8 P-cores with SMT,
16 E-cores), P-core SMT pairs are CPUs 0–15 and E-cores are CPUs 16–31.
See `results/` for the public scaling table and
`examples/compare_threads.sh` to reproduce thread comparison.
Scaling is sublinear past ~4 threads on this memory-streaming workload
(bandwidth-bound, SMT adds nothing) and static chunking idles fast workers
behind variable-cost examples. A shared work queue would be the next step
if scaling matters more.
