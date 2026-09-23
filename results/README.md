# Reproducible public benchmark outputs

Every run here uses only redistributable or generated data — never private
corpora. Machine-readable tables are TSV; each run also has a short summary.

- `synthetic_benchmark.tsv` / `.json`: 4000/500/500 synthetic set
  (see `tests/data/README.md` for the generator command and hashes),
  W64/R64/C192/O32, 8 epochs, batch 64, seed 7, threads 8.
- `thread_scaling.tsv` / `.md`: same config, `--threads` 1..32 plus
  i9-13900K affinity sets (P-core SMT pairs are CPUs 0-15, E-cores 16-31).

No `.sjev` model binaries are committed (`*.sjev` is git-ignored);
`model_sha256` lets anyone verify a local rerun bit-for-bit against the
recorded training recipe.
