# TinyScorer native forward path — exact specification

SJEV is an independent C implementation informed by the TinyScorer
architecture demonstrated by the Jevlike research project
(https://github.com/vinnylarouge/jevlike): `jevlike/model.py`
(TinyScorer + AttentionHead) and `jevlike/data.py` (ByteCollator / `_bytes`).
No Jevlike source code is included in this distribution.

## Parameter inventory (TinyScorer only)

Pytorch `state_dict` keys saved by `jevlike/train.py`:

| key | shape | note |
|-----|-------|------|
| `embedding.weight` | `[257, width]` | `padding_idx=0`, 257 = 256 byte values + 1 padding row |
| `position.weight` | `[context_tokens, width]` | `nn.Embedding(context_tokens, width)`, no padding |
| `head.context_norm.weight` | `[width]` | LayerNorm gamma |
| `head.context_norm.bias` | `[width]` | LayerNorm beta |
| `head.option_norm.weight` | `[width]` | |
| `head.option_norm.bias` | `[width]` | |
| `head.query.weight` | `[rank, width]` | `nn.Linear(width, rank, bias=False)` |
| `head.key.weight` | `[rank, width]` | |
| `head.value.weight` | `[rank, width]` | |

No other tensors. `encoder == "tiny"` path never writes encoder weights other than above.

## Tensor shapes for a single batch element

Let
- `W = width`
- `R = rank`
- `L = context sequence length` (actual bytes after truncation, 0 < L ≤ context_tokens)
- `N = number of options` (≥2)
- `K_i = token length of option i` (0 < K_i ≤ option_tokens, truncated per-option)

Byte tokenisation (`jevlike/data.py:_bytes`):
```
def _bytes(text, length):
    return [byte + 1 for byte in text.encode("utf-8", errors="replace")[:length]]
```
- UTF-8 encode with `errors="replace"`, truncate the raw byte string to `length` bytes, then map each byte `b → b+1`.
- `0` reserved for padding; never emitted by `_bytes` but stored as row 0 of embedding.
- Truncation is by bytes, not characters; a multi-byte codepoint split mid-sequence leaves the partial bytes as-is.

Collator (`ByteCollator`, `_tensor_batch`):
- `context_ids: [B, L_max]` padded with `0`
- `context_mask: [B, L_max]` bool `context_ids != 0`
- `option_ids: [B, N_max, K_max]` padded with `0`
- `option_token_mask: [B, N_max, K_max]` bool
- `option_mask: [B, N_max]` bool (which N slots are present)

Single-example inference collapses batch dim `B=1`; native CLI processes one example at a time and therefore `L = L_max`, `N = N_max`, `K_max = max_i K_i` but per-option mean-pool ignores padding via mask.

## Forward

Given one example `b=0`:

1. **Context embedding + position**
   ```
   pos = arange(L)                         # [L]
   ctx_emb = embedding[context_ids[0]]      # [L, W]
   pos_emb = position[pos]                  # [L, W]
   ctx = ctx_emb + pos_emb                  # [L, W]  float32
   ```

2. **Option embedding + mean-pool**
   ```
   opt_tok = embedding[option_ids[0]]       # [N, K_max, W]
   w = option_token_mask[0].unsqueeze(-1)   # [N, K_max, 1]  0/1 float
   # masked sum then masked mean:
   opt = (opt_tok * w).sum(dim=1) / w.sum(dim=1).clamp_min(1)  # [N, W]
   ```
   i.e. per-option `opt[n,w] = ( Σ_{k < K_n} emb[ id[n,k], w ]) / K_n`. `clamp_min(1)` guards zero-length options (spec requires ≥1 but implemented).

3. **LayerNorm** (both branches `eps=1e-5`, `elementwise_affine=True`, PyTorch default unbiased=False)
   ```
   c = context_norm(ctx.float())   # [L, W]
   o = option_norm(opt.float())    # [N, W]

   # per row r, per dim d:
   mean = sum_d x[r,d] / W
   var  = sum_d (x[r,d]-mean)^2 / W
   y[r,d] = (x[r,d]-mean) / sqrt(var + 1e-5) * weight[d] + bias[d]
   ```

4. **Q / K / V** (bias-free linear)
   ```
   Wq, Wk, Wv : [R, W]   (PyTorch layout [out, in])
   Q = o @ Wq.T   # [N, R]
   K = c @ Wk.T   # [L, R]
   V = c @ Wv.T   # [L, R]
   ```

5. **Scaled dot-product attention (per option query)**
   ```
   scores_raw[n, l] = dot(Q[n], K[l]) / sqrt(R)   # [N, L]
   # masking: where context_mask is False, scores = finfo(float32).min (-3.4028235e38)
   # single-example L == true length => no masking
   scale = sqrt(R)
   ```

6. **Softmax over context**
   ```
   attn[n, l] = softmax_l(scores_raw[n, :])   # [N, L], softmax after masking
   softmax numerically stable: max-subtract, exp, renormalise. float32.
   ```

7. **Attended context**
   ```
   A[n, r] = Σ_l attn[n,l] * V[l, r]   # [N, R]
   ```

8. **Scoring (dot, scaled again)**
   ```
   logits[n] = dot(Q[n], A[n]) / sqrt(R)   # [N]
   # masking: where option_mask False => finfo.min. single-example => none.
   ```

9. **Option softmax**
   ```
   prob[n] = softmax_n(logits)   # [N]
   ```

References:
- `jevlike/model.py:12-43` AttentionHead
- `jevlike/model.py:46-63` TinyScorer.forward
- `jevlike/data.py:49-62` ByteCollator

## Numerical details relevant to C parity

- All activations `float32` (`float` in C, IEEE-754 binary32). `context.float()` / `options.float()` are no-ops when input is already float32.
- LayerNorm epsilon `1e-5` (PyTorch default).
- `torch.finfo(float32).min = -3.4028234663852886e38`. Used as `-inf` surrogate for masked positions.
- Softmax and `exp` use libm float32; tolerance target `1e-5` for probs/logits.
- `sqrt(R)` computed as `math.sqrt(rank)` (Python float64) then used in float32 division; C `sqrtf((float)rank)` matches within 1 ULP for rank 64.

## Checkpoint config fields (src reads them for format)

`torch.save({"config": config, "state_dict": trainable_state})` where config:
```
encoder       : "tiny"
hf_model      : str (unused for tiny)
width         : int
rank          : int
context_tokens: int
option_tokens : int
```
These widths are written to the binary header and drive tensor sizes; the C maths must not hard-code them.

## What is NOT in milestone one

- Vision path (`jevlike/vision.py` DoomScorerV2) – different stem/positions/reads.
- Training (cross-entropy, AdamW, grad clip).
- HuggingFace encoder path.
- Batched inference beyond `B=1`.
