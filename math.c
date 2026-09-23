/*
 * math.c — SJEV layernorm, matmuls and softmax helpers.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the Business Source License 1.1 — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 * Developed for SHAMPOO, the Shared Human-Agent-Model Platform for
 * Orchestration and Operations (https://shampoo.op2.uk/).
 */
#include "math.h"
#include <math.h>

void sjev_layernorm(const float *in, float *out, int rows, int width,
                    const float *weight, const float *bias) {
    const double eps = 1e-5;
    for (int r = 0; r < rows; r++) {
        const float *src = in + (size_t)r * width;
        float *dst = out + (size_t)r * width;
        double sum = 0.0;
        for (int w = 0; w < width; w++) sum += (double)src[w];
        double mean = sum / (double)width;
        double var_sum = 0.0;
        for (int w = 0; w < width; w++) {
            double d = (double)src[w] - mean;
            var_sum += d * d;
        }
        double var = var_sum / (double)width;
        float inv = (float)(1.0 / sqrt(var + eps));
        float mean_f = (float)mean;
        for (int w = 0; w < width; w++) {
            float norm = (src[w] - mean_f) * inv;
            dst[w] = norm * weight[w] + bias[w];
        }
    }
}

void sjev_mat_q(const float *norm_opt, const float *Wq, float *Q, int N, int W, int R) {
    for (int n = 0; n < N; n++) {
        const float *opt = norm_opt + (size_t)n * W;
        float *q = Q + (size_t)n * R;
        for (int r = 0; r < R; r++) {
            const float *wr = Wq + (size_t)r * W;
            double acc = 0.0;
            for (int w = 0; w < W; w++) acc += (double)opt[w] * (double)wr[w];
            q[r] = (float)acc;
        }
    }
}

void sjev_mat_kv(const float *norm_ctx, const float *W, float *KV, int L, int width, int R) {
    for (int l = 0; l < L; l++) {
        const float *ctx = norm_ctx + (size_t)l * width;
        float *kv = KV + (size_t)l * R;
        for (int r = 0; r < R; r++) {
            const float *wr = W + (size_t)r * width;
            double acc = 0.0;
            for (int w = 0; w < width; w++) acc += (double)ctx[w] * (double)wr[w];
            kv[r] = (float)acc;
        }
    }
}

void sjev_softmax_rows(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *src = in + (size_t)r * cols;
        float *dst = out + (size_t)r * cols;
        float maxv = src[0];
        for (int c = 1; c < cols; c++) if (src[c] > maxv) maxv = src[c];
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float e = expf(src[c] - maxv);
            dst[c] = e;
            sum += e;
        }
        float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
        for (int c = 0; c < cols; c++) dst[c] *= inv;
    }
}

void sjev_softmax_1d(const float *in, float *out, int n) {
    if (n <= 0) return;
    float maxv = in[0];
    for (int i = 1; i < n; i++) if (in[i] > maxv) maxv = in[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = expf(in[i] - maxv);
        out[i] = e;
        sum += e;
    }
    float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
    for (int i = 0; i < n; i++) out[i] *= inv;
}
