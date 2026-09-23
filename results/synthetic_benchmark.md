# Synthetic benchmark (public data, W64/R64/C192/O32, 8 epochs, seed 7)

Dataset: `tools/make_synthetic.py --train 4000 --validation 500 --test 500
--seed 7` (hashes in `synthetic_benchmark.json`). Test top1 **0.992**,
top3 **1.000**, NLL 0.0148. Shuffled control: top1 **0.298** (≈ chance for
3–4 options), NLL 6.41 — the model depends on the query.
