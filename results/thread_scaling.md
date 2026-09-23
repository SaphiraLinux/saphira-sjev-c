# Thread scaling (public synthetic set, 32,000 examples, i9-13900K)

Unpinned `--threads` 1..32 plus pinned affinity sets (P SMT pairs CPUs
0–15, E-cores 16–31). Full table: `thread_scaling.tsv`.

Fastest overall: **24 unpinned workers — 1.69 s, 6.85×**. Best explicitly
pinned physical-core topology: **8P + 16E — 1.77 s, 6.52×** (Linux
scheduling the 24 workers did slightly better on that run than the
explicit mask). 32 threads (with P-SMT) regress to 5.81×; 16 E-cores
(5.37×) beat 16 P-logical threads (4.48×). Validation NLL is identical
to ~1e-4 across every thread count.
Scaling is sublinear past ~4 threads: memory-streaming workload plus
static-chunk stragglers (see THREADING.md).
