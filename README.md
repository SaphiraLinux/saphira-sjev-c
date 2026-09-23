# SJEV

SJEV is a small, self-contained C implementation of a one-pass option
scorer: given a text context and a list of text options, it returns one
probability per option. See `ARCHITECTURE.md` for the model specification
and `FORMAT.md` for the standalone `.sjev` binary model format.

The native build and runtime need only a C11 compiler, libc, libm and
pthreads. There is no Python, PyTorch, CUDA, or BLAS dependency for
building, training, or running SJEV itself. Python (standard library only)
is used for dataset/test tooling: `tools/` generators, `tests/` checks and
the optional `tools/export_model.py` checkpoint-conversion script, never
for the C program.

Branches: `Master` is the original single-thread baseline. The
`multithreaded` branch adds the pthreads trainer (`--threads N`);
`--threads 1` reproduces the single-worker path there.

## Layout
```
ARCHITECTURE.md   model specification
FORMAT.md         .sjev binary format
THREADING.md      multithreaded trainer design and scaling notes
README.md         this file
LICENSE           Business Source License 1.1
NOTICE            authorship / acknowledgement notice
Makefile          top-level driver (delegates to src/)
src/              native C implementation (sjev.c model.c math.c ...)
tests/            shell test suite + committed fixtures (tests/data/)
examples/         runnable quickstart / train / eval / thread comparison
tools/            dataset preparation/generation utilities
results/          reproducible public benchmark outputs
```

## Build
make
./src/sjev --help
Compiler defaults (CC ?= cc, CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic, link -lm -pthread) are musl-compatible and use plain POSIX plus libm.

## Tests
make check
Runs `tests/run_tests.sh`: build, CLI help/error paths, init/save/load,
prediction determinism (probabilities sum to one), tiny train with falling
loss, eval metrics, fixed-seed reproducibility, `--threads 1 vs 2/4/8`
parity, malformed-model rejection, and a Master-vs-branch parity exercise
via a temporary git worktree. Sanitizer legs: `make asan`, `make ubsan`,
`make tsan` (each rebuilds `src/sjev` under that sanitizer).

## Data format
One JSON object per line with `context` (string), `options` (array >=2 non-empty strings), `label` (zero-based index):
{"context": "The customer needs a refund.", "options": ["refund", "sales", "technical support"], "label": 0}

Public datasets: `tools/make_synthetic.py` generates synthetic train/val/test
(no external source; see `tests/data/README.md` for provenance of every
committed fixture). `tools/prepare_wikispeedia.py`,
`tools/prepare_dictionary.py` and `tools/prepare_thesaurus.py` build
full-size sets from user-supplied public/upstream sources (never committed).

## Train
./src/sjev train TRAIN.jsonl --validation VALID.jsonl --output MODEL.sjev
Flags (defaults): --width 64 --rank 64 --context-tokens 192 --option-tokens 32 --epochs 8 --batch-size 64 --learning-rate 0.002 --weight-decay 1e-4 --seed 7 --threads 1
Keeps best-validation-loss checkpoint. --init-model MODEL.sjev continues with fresh optimiser state.
`--threads N` (1..1024) parallelises examples within each batch with private
per-worker gradients, one deterministic reduction, one global clip and one
AdamW step; weights are read-only in workers. See THREADING.md.

## Query
./src/sjev predict MODEL.sjev --context "Choose the exact badge amber badger." --option "azure crane" --option "amber badger" --option "gold heron"
Output JSON [{option, probability, logit}], probs sum to one. Inference time on stderr.

## Evaluate
./src/sjev eval MODEL.sjev TEST.jsonl
Prints test_nll, top1, top3, ece, example count, timing as JSON.

## Examples
sh examples/quickstart.sh
Builds, generates data, trains, evaluates and predicts end to end.
sh examples/compare_threads.sh [threads...]
Trains the same dataset/seed under several thread counts for comparison.

## Checkpoint conversion (optional)
tools/export_model.py converts TinyScorer .pt (torch.save({"config": ..., "state_dict": ...}), encoder="tiny") to .sjev:
python3 tools/export_model.py runs/model.pt model.sjev
Requires PyTorch; the C program never does.

## Licence
SJEV is licensed under the Business Source License 1.1 — see LICENSE. See NOTICE for authorship and acknowledgement notice.
