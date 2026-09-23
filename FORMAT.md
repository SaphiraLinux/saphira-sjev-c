# sjev binary model format — Saphira Jev v1

Standalone format for TinyScorer. No PyTorch at runtime.

## General

- Magic: 4 bytes ASCII `SJEV` (`0x53 0x4A 0x45 0x56`)
- Byte order: little-endian for all multi-byte integers and FP32.
- FP32: IEEE-754 binary32, little-endian.
- Alignment: no padding beyond the header's fixed 32 bytes; tensors are tightly packed.
- Tensors are row-major, C-contiguous, matching `torch` contiguous layout.

## Header (32 bytes, little-endian)

| offset | size | type   | name            | description |
|--------|------|--------|-----------------|-------------|
| 0      | 4    | char[4]| magic           | `SJEV` |
| 4      | 4    | uint32 | version         | `1` (uint32 LE) |
| 8      | 4    | uint32 | width           | `W`  |
| 12     | 4    | uint32 | rank            | `R`  |
| 16     | 4    | uint32 | context_tokens  | `C` max bytes for context truncation |
| 20     | 4    | uint32 | option_tokens   | `O` max bytes per option |
| 24     | 4    | uint32 | reserved0       | `0` |
| 28     | 4    | uint32 | reserved1       | `0` |

Valid ranges (loader rejects otherwise): `1 ≤ W,R ≤ 4096`, `1 ≤ C,O ≤ 8192`, `version == 1`, magic exact. The two reserved fields must be zero (future flags). Header is 32 bytes regardless of future version bump.

## Tensor payload (immediately after header)

Sizes derived from header; all `float32 LE`:

1. `embedding` `[257][W]` — `257*W` floats, row 0 is padding (`padding_idx=0`). Row `i` = `embedding.weight[i*W + w]`.
2. `position` `[C][W]` — `C*W` floats, row `p` = `position.weight[p*W + w]`.
3. `context_norm_weight` `[W]` floats — `head.context_norm.weight`
4. `context_norm_bias` `[W]` floats — `head.context_norm.bias`
5. `option_norm_weight` `[W]` floats — `head.option_norm.weight`
6. `option_norm_bias` `[W]` floats — `head.option_norm.bias`
7. `query_weight` `[R][W]` — `R*W` floats, PyTorch `head.query.weight` layout `[R, W]` row-major: `query_weight[r*W + w]`.
8. `key_weight` `[R][W]` — `head.key.weight` same layout.
9. `value_weight` `[R][W]` — `head.value.weight` same layout.

Total file size:
```
32 + 4 * (257*W + C*W + W + W + W + W + R*W + R*W + R*W)
= 32 + 4 * (W*(257 + C + 3*R) + 4*W)
= 32 + 4 * W * (261 + C + 3*R)
```
Examples:
- `W=64,R=64,C=256,O=32` → `32 + 4*45376 = 181536` bytes (as in `runs/wikispeedia-256-32.pt` conversion).
- `W=32,R=32,C=192,O=32` → `32 + 4*W*(261+C+3R)` correspondingly.

`O` does not affect payload size; it controls tokenisation truncation only. The same binary works with any `O` because option pooling divides by actual length. `O` is nonetheless stored to document the intended tokenizer pairing.

## Ordering rationale

Order matches the logical forward pass and the `state_dict` enumeration: embedding/position first (lookup), then norms, then QKV projections. No transposition is applied during export; the C loader uses the same `[R, W]` row-major as written.

## Integrity

- Loader checks file size equals exactly header-predicted size; extra trailing bytes or truncation are rejected as corrupt.
- No checksum field in v1; corruption is detected via size mismatch and subsequent size checks.
- Floats are not validated beyond being readable; NaN/Inf in weights is accepted (will propagate to logits).

## Export

`src/export_model.py` reads a `torch.save({"config":..., "state_dict":...})` checkpoint produced by `jevlike/train.py` and writes this format. It rejects `encoder != "tiny"` and verifies all nine tensors are present with expected shapes.

## Versioning

`version=1` is the only valid version. Future versions must be negotiated by bumping this field; v1 loaders reject `version != 1`.
