# SJEV

SJEV is a small, self-contained C implementation of a one-pass option
scorer: given a text context and a list of text options, it returns one
probability per option. See `ARCHITECTURE.md` for the model specification
and `FORMAT.md` for the standalone `.sjev` binary model format.

The native build and runtime need only a C11 compiler and libm. There is
no Python, PyTorch, CUDA, or BLAS dependency for building, training, or
running SJEV itself. PyTorch is required only for the optional
`export_model.py` checkpoint-conversion script, never for the C program.

The initial public release does not include the internal research test
and corpus-generation tooling. Independent hermetic tests will be added
separately.

## Layout

```text
ARCHITECTURE.md   model specification
FORMAT.md         .sjev binary format
README.md         this file
LICENSE           Business Source License 1.1
NOTICE            authorship / acknowledgement notice
Makefile          build rules
sjev.c            CLI: predict, train, eval, grad-check
model.c / model.h model load/save/init, inference
math.c / math.h   layernorm, matmuls, softmax
token.c / token.h byte tokenisation
data.c / data.h   JSONL dataset loading
train.c / train.h forward/backward, Adam, training loop
export_model.py   optional: convert a TinyScorer .pt checkpoint to .sjev
```

## Build

```sh
make
./sjev --help
```

The compiler defaults (`CC ?= cc`, `CFLAGS ?= -O2 -std=c11 -Wall -Wextra
-Wpedantic`, link `-lm`) are musl-compatible and use plain POSIX plus
libm.

## Data format

One JSON object per line with `context` (string), `options` (array of at
least two non-empty strings), and `label` (zero-based index of the
correct option):

```json
{"context": "The customer needs a refund.", "options": ["refund", "sales", "technical support"], "label": 0}
```

## Train

```sh
./sjev train TRAIN.jsonl --validation VALID.jsonl --output MODEL.sjev
```

Useful flags (defaults shown):

```text
--width 64 --rank 64 --context-tokens 192 --option-tokens 32
--epochs 8 --batch-size 64 --learning-rate 0.002
--weight-decay 1e-4 --seed 7
```

Training keeps the checkpoint with the best validation loss and writes it
to `--output`. To continue from an existing model with fresh optimiser
state: `--init-model MODEL.sjev`.

## Query

```sh
./sjev predict MODEL.sjev --context "Choose the exact badge amber badger." \
  --option "azure crane" --option "amber badger" --option "gold heron"
```

Output is JSON with one `{option, probability, logit}` entry per option;
probabilities sum to one. Inference time is reported on stderr.

## Evaluate

```sh
./sjev eval MODEL.sjev TEST.jsonl
```

Prints `test_nll`, `top1`, `top3`, `ece`, example count, and timing as
JSON.

## Checkpoint conversion (optional)

`export_model.py` converts a TinyScorer `.pt` checkpoint
(`torch.save({"config": ..., "state_dict": ...})`, `encoder="tiny"`) to
the `.sjev` format. This is the only script that needs PyTorch:

```sh
python3 export_model.py runs/model.pt model.sjev
```

## Licence

SJEV is licensed under the Business Source License 1.1 — see `LICENSE`.
See `NOTICE` for the authorship and acknowledgement notice.
