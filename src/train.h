/*
 * train.h — SJEV training structures and interface.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the Business Source License 1.1 — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 * Developed for SHAMPOO, the Shared Human-Agent-Model Platform for
 * Orchestration and Operations (https://shampoo.op2.uk/).
 */
#pragma once
#include "model.h"
#include "data.h"
#include <stdint.h>
#include <stddef.h>

struct sjev_grad {
    float *embedding; // [257*W]
    float *position;  // [C*W]
    float *ln_c_w, *ln_c_b;
    float *ln_o_w, *ln_o_b;
    float *Wq, *Wk, *Wv;
};

struct adam_state {
    float *m_emb, *v_emb;
    float *m_pos, *v_pos;
    float *m_lc_w, *v_lc_w;
    float *m_lc_b, *v_lc_b;
    float *m_lo_w, *v_lo_w;
    float *m_lo_b, *v_lo_b;
    float *m_Wq, *v_Wq;
    float *m_Wk, *v_Wk;
    float *m_Wv, *v_Wv;
    int64_t t; // step
    float beta1, beta2, eps, wd, lr;
};

struct forward_cache {
    int Lc;
    int N;
    int label;
    size_t *opt_lens; // N
    uint32_t *ctx_ids; // Lc
    uint32_t **opt_ids; // N
    float *ctx_emb; // Lc*W
    float *opt_pool; // N*W
    float *ctx_norm;
    float *opt_norm;
    float *Q; // N*R
    float *K; // Lc*R
    float *V;
    float *scores; // N*Lc
    float *attn; // N*Lc
    float *attended; // N*R
    float *logits; // N
    float *probs; // N
    float loss;
    // layernorm caches needed for backward: mean/inv/xhat could recompute, but store for speed
    // we keep ctx_emb and opt_pool to recompute; don't need extra
};

int grad_alloc(const struct sjev_model *m, struct sjev_grad *g, char *err, size_t err_len);
void grad_free(const struct sjev_model *m, struct sjev_grad *g);
void grad_zero(const struct sjev_model *m, struct sjev_grad *g);
void grad_scale(const struct sjev_model *m, struct sjev_grad *g, float scale);
float grad_global_norm(const struct sjev_model *m, const struct sjev_grad *g);
void grad_clip(const struct sjev_model *m, struct sjev_grad *g, float max_norm);

int adam_alloc(const struct sjev_model *m, struct adam_state *s, float lr, float beta1, float beta2, float eps, float wd);
void adam_free(const struct sjev_model *m, struct adam_state *s);
void adam_step(struct sjev_model *m, const struct sjev_grad *g, struct adam_state *s);

int forward_one(const struct sjev_model *m, const struct example *ex, struct forward_cache *c, char *err, size_t err_len);
void cache_free(struct forward_cache *c);
int backward_one(const struct sjev_model *m, const struct forward_cache *c, struct sjev_grad *grad, char *err, size_t err_len);

// helpers for loss / metrics
float cache_loss(const struct forward_cache *c);
int eval_dataset(const struct sjev_model *m, const struct dataset *ds, float *out_nll, float *out_top1, float *out_top3, char *err, size_t err_len);

/* Multithreaded batch trainer (pthreads, no OpenMP).
 * Workers process disjoint examples of one batch into private gradient
 * buffers; the main thread reduces in worker-index order, then clips/scales
 * once and applies one AdamW step. Model weights are read-only in workers.
 * Reduction is deterministic for a fixed thread count (plain FP loops, no
 * atomics); 1-thread vs N-thread results differ only by FP32 summation order.
 */
struct sjev_thread_pool;
struct sjev_thread_pool *sjev_pool_create(const struct sjev_model *m, int nthreads, char *err, size_t err_len);
void sjev_pool_free(const struct sjev_model *m, struct sjev_thread_pool *p);
/* Run indices[start..end) through forward/backward, accumulating the mean
 * gradient basis into grad_out and the summed loss into *loss_sum.
 * grad_out is fully overwritten (zeroed then reduced). */
int sjev_pool_run_batch(struct sjev_thread_pool *p, const struct sjev_model *m,
                        const struct dataset *ds, const size_t *indices,
                        size_t start, size_t end,
                        struct sjev_grad *grad_out, double *loss_sum,
                        char *err, size_t err_len);

int train_loop(const struct sjev_model *model_init, const struct dataset *train, const struct dataset *valid,
               int epochs, int batch_size, int nthreads, struct adam_state *adam, const char *output_path,
               char *err, size_t err_len);
