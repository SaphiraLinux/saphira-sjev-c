#pragma once
#include <stddef.h>

/* LayerNorm per-row: in [rows*width] -> out [rows*width]
 * eps = 1e-5, weight/bias [width]
 */
void sjev_layernorm(const float *in, float *out, int rows, int width,
                    const float *weight, const float *bias);

/* Matmul helpers: all float32 plain loops */
void sjev_mat_q(const float *norm_opt, const float *Wq, float *Q, int N, int W, int R);
void sjev_mat_kv(const float *norm_ctx, const float *W, float *KV, int L, int width, int R);

/* Softmax rows: inout scores [rows*cols] -> attention [rows*cols] (or probs) */
void sjev_softmax_rows(const float *in, float *out, int rows, int cols);
void sjev_softmax_1d(const float *in, float *out, int n);
