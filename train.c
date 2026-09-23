#define _POSIX_C_SOURCE 200809L
#include "train.h"
#include "token.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

// Helpers for layernorm backward inlined in backward_one

int grad_alloc(const struct sjev_model *m, struct sjev_grad *g, char *err, size_t err_len) {
    if(!m||!g) return -1;
    memset(g,0,sizeof(*g));
    size_t emb_n=(size_t)257*m->width;
    size_t pos_n=(size_t)m->context_tokens*m->width;
    size_t norm_n=m->width;
    size_t qkv_n=(size_t)m->rank*m->width;
    g->embedding=(float*)calloc(emb_n, sizeof(float));
    g->position=(float*)calloc(pos_n, sizeof(float));
    g->ln_c_w=(float*)calloc(norm_n, sizeof(float));
    g->ln_c_b=(float*)calloc(norm_n, sizeof(float));
    g->ln_o_w=(float*)calloc(norm_n, sizeof(float));
    g->ln_o_b=(float*)calloc(norm_n, sizeof(float));
    g->Wq=(float*)calloc(qkv_n, sizeof(float));
    g->Wk=(float*)calloc(qkv_n, sizeof(float));
    g->Wv=(float*)calloc(qkv_n, sizeof(float));
    if(!g->embedding||!g->position||!g->ln_c_w||!g->ln_c_b||!g->ln_o_w||!g->ln_o_b||!g->Wq||!g->Wk||!g->Wv){
        if(err) snprintf(err,err_len,"oom grad_alloc");
        grad_free(m,g);
        return -1;
    }
    return 0;
}
void grad_free(const struct sjev_model *m, struct sjev_grad *g) {
    (void)m;
    if(!g) return;
    free(g->embedding); free(g->position); free(g->ln_c_w); free(g->ln_c_b); free(g->ln_o_w); free(g->ln_o_b); free(g->Wq); free(g->Wk); free(g->Wv);
    memset(g,0,sizeof(*g));
}
void grad_zero(const struct sjev_model *m, struct sjev_grad *g) {
    size_t emb_n=(size_t)257*m->width;
    size_t pos_n=(size_t)m->context_tokens*m->width;
    size_t norm_n=m->width;
    size_t qkv_n=(size_t)m->rank*m->width;
    memset(g->embedding,0,emb_n*sizeof(float));
    memset(g->position,0,pos_n*sizeof(float));
    memset(g->ln_c_w,0,norm_n*sizeof(float));
    memset(g->ln_c_b,0,norm_n*sizeof(float));
    memset(g->ln_o_w,0,norm_n*sizeof(float));
    memset(g->ln_o_b,0,norm_n*sizeof(float));
    memset(g->Wq,0,qkv_n*sizeof(float));
    memset(g->Wk,0,qkv_n*sizeof(float));
    memset(g->Wv,0,qkv_n*sizeof(float));
}
void grad_scale(const struct sjev_model *m, struct sjev_grad *g, float scale) {
    size_t emb_n=(size_t)257*m->width;
    size_t pos_n=(size_t)m->context_tokens*m->width;
    size_t norm_n=m->width;
    size_t qkv_n=(size_t)m->rank*m->width;
    for(size_t i=0;i<emb_n;i++) g->embedding[i]*=scale;
    for(size_t i=0;i<pos_n;i++) g->position[i]*=scale;
    for(size_t i=0;i<norm_n;i++){ g->ln_c_w[i]*=scale; g->ln_c_b[i]*=scale; g->ln_o_w[i]*=scale; g->ln_o_b[i]*=scale; }
    for(size_t i=0;i<qkv_n;i++){ g->Wq[i]*=scale; g->Wk[i]*=scale; g->Wv[i]*=scale; }
}
float grad_global_norm(const struct sjev_model *m, const struct sjev_grad *g) {
    double sum=0;
    size_t emb_n=(size_t)257*m->width;
    size_t pos_n=(size_t)m->context_tokens*m->width;
    size_t norm_n=m->width;
    size_t qkv_n=(size_t)m->rank*m->width;
    for(size_t i=0;i<emb_n;i++){ double v=g->embedding[i]; sum+=v*v; }
    for(size_t i=0;i<pos_n;i++){ double v=g->position[i]; sum+=v*v; }
    for(size_t i=0;i<norm_n;i++){ double v=g->ln_c_w[i]; sum+=v*v; v=g->ln_c_b[i]; sum+=v*v; v=g->ln_o_w[i]; sum+=v*v; v=g->ln_o_b[i]; sum+=v*v; }
    for(size_t i=0;i<qkv_n;i++){ double v=g->Wq[i]; sum+=v*v; v=g->Wk[i]; sum+=v*v; v=g->Wv[i]; sum+=v*v; }
    return (float)sqrt(sum);
}
void grad_clip(const struct sjev_model *m, struct sjev_grad *g, float max_norm) {
    float norm = grad_global_norm(m,g);
    if (norm > max_norm && norm > 0) {
        float scale = max_norm / norm;
        grad_scale(m,g,scale);
    }
}

int adam_alloc(const struct sjev_model *m, struct adam_state *s, float lr, float beta1, float beta2, float eps, float wd) {
    memset(s,0,sizeof(*s));
    s->lr=lr; s->beta1=beta1; s->beta2=beta2; s->eps=eps; s->wd=wd; s->t=0;
    size_t emb_n=(size_t)257*m->width;
    size_t pos_n=(size_t)m->context_tokens*m->width;
    size_t norm_n=m->width;
    size_t qkv_n=(size_t)m->rank*m->width;
    s->m_emb=(float*)calloc(emb_n,sizeof(float)); s->v_emb=(float*)calloc(emb_n,sizeof(float));
    s->m_pos=(float*)calloc(pos_n,sizeof(float)); s->v_pos=(float*)calloc(pos_n,sizeof(float));
    s->m_lc_w=(float*)calloc(norm_n,sizeof(float)); s->v_lc_w=(float*)calloc(norm_n,sizeof(float));
    s->m_lc_b=(float*)calloc(norm_n,sizeof(float)); s->v_lc_b=(float*)calloc(norm_n,sizeof(float));
    s->m_lo_w=(float*)calloc(norm_n,sizeof(float)); s->v_lo_w=(float*)calloc(norm_n,sizeof(float));
    s->m_lo_b=(float*)calloc(norm_n,sizeof(float)); s->v_lo_b=(float*)calloc(norm_n,sizeof(float));
    s->m_Wq=(float*)calloc(qkv_n,sizeof(float)); s->v_Wq=(float*)calloc(qkv_n,sizeof(float));
    s->m_Wk=(float*)calloc(qkv_n,sizeof(float)); s->v_Wk=(float*)calloc(qkv_n,sizeof(float));
    s->m_Wv=(float*)calloc(qkv_n,sizeof(float)); s->v_Wv=(float*)calloc(qkv_n,sizeof(float));
    if(!s->m_emb||!s->v_emb||!s->m_pos||!s->v_pos||!s->m_lc_w||!s->v_lc_w||!s->m_lc_b||!s->v_lc_b||!s->m_lo_w||!s->v_lo_w||!s->m_lo_b||!s->v_lo_b||!s->m_Wq||!s->v_Wq||!s->m_Wk||!s->v_Wk||!s->m_Wv||!s->v_Wv){
        adam_free(m,s); return -1;
    }
    return 0;
}
void adam_free(const struct sjev_model *m, struct adam_state *s) {
    (void)m;
    free(s->m_emb); free(s->v_emb);
    free(s->m_pos); free(s->v_pos);
    free(s->m_lc_w); free(s->v_lc_w);
    free(s->m_lc_b); free(s->v_lc_b);
    free(s->m_lo_w); free(s->v_lo_w);
    free(s->m_lo_b); free(s->v_lo_b);
    free(s->m_Wq); free(s->v_Wq);
    free(s->m_Wk); free(s->v_Wk);
    free(s->m_Wv); free(s->v_Wv);
    memset(s,0,sizeof(*s));
}

static void adam_update_array(float *param, float *m, float *v, const float *grad, size_t n, int64_t t, float lr, float beta1, float beta2, float eps, float wd, int skip_first_W) {
    double beta1_t = pow((double)beta1, (double)t);
    double beta2_t = pow((double)beta2, (double)t);
    double inv1 = 1.0 / (1.0 - beta1_t);
    double inv2 = 1.0 / (1.0 - beta2_t);
    for(size_t i=0;i<n;i++){
        if (skip_first_W && i < (size_t)skip_first_W) continue;
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        double m_hat = m[i] * inv1;
        double v_hat = v[i] * inv2;
        double upd = m_hat / (sqrt(v_hat) + eps);
        double wd_term = wd * param[i];
        param[i] = (float)(param[i] - lr * (upd + wd_term));
    }
}

void adam_step(struct sjev_model *m, const struct sjev_grad *g, struct adam_state *s) {
    s->t++;
    int64_t t = s->t;
    // zero grad for embedding row 0 already zero, but skip its update
    adam_update_array(m->embedding, s->m_emb, s->v_emb, g->embedding, (size_t)257*m->width, t, s->lr, s->beta1, s->beta2, s->eps, s->wd, (int)m->width);
    adam_update_array(m->position, s->m_pos, s->v_pos, g->position, (size_t)m->context_tokens*m->width, t, s->lr, s->beta1, s->beta2, s->eps, s->wd, 0);
    adam_update_array(m->ln_c_w, s->m_lc_w, s->v_lc_w, g->ln_c_w, m->width, t, s->lr, s->beta1, s->beta2, s->eps, s->wd, 0);
    adam_update_array(m->ln_c_b, s->m_lc_b, s->v_lc_b, g->ln_c_b, m->width, t, s->lr, s->beta1, s->beta2, s->eps, s->wd, 0);
    adam_update_array(m->ln_o_w, s->m_lo_w, s->v_lo_w, g->ln_o_w, m->width, t, s->lr, s->beta1, s->beta2, s->eps, s->wd, 0);
    adam_update_array(m->ln_o_b, s->m_lo_b, s->v_lo_b, g->ln_o_b, m->width, t, s->lr, s->beta1, s->beta2, s->eps, s->wd, 0);
    adam_update_array(m->Wq, s->m_Wq, s->v_Wq, g->Wq, (size_t)m->rank*m->width, t, s->lr, s->beta1, s->beta2, s->eps, s->wd, 0);
    adam_update_array(m->Wk, s->m_Wk, s->v_Wk, g->Wk, (size_t)m->rank*m->width, t, s->lr, s->beta1, s->beta2, s->eps, s->wd, 0);
    adam_update_array(m->Wv, s->m_Wv, s->v_Wv, g->Wv, (size_t)m->rank*m->width, t, s->lr, s->beta1, s->beta2, s->eps, s->wd, 0);
    // ensure padding row stays zero
    for(uint32_t w=0;w<m->width;w++) m->embedding[w]=0.0f;
}

int forward_one(const struct sjev_model *m, const struct example *ex, struct forward_cache *c, char *err, size_t err_len) {
    memset(c,0,sizeof(*c));
    uint32_t W=m->width, R=m->rank;
    int Lc = (int)sjev_truncated_len(ex->context, m->context_tokens);
    if (Lc==0) { if(err) snprintf(err,err_len,"empty context"); return -1; }
    int N = ex->n_options;
    c->Lc=Lc; c->N=N; c->label=ex->label;
    c->opt_lens=(size_t*)malloc(N*sizeof(size_t));
    c->ctx_ids=(uint32_t*)malloc(Lc*sizeof(uint32_t));
    c->opt_ids=(uint32_t**)malloc(N*sizeof(uint32_t*));
    if(!c->opt_lens||!c->ctx_ids||!c->opt_ids){ if(err) snprintf(err,err_len,"oom"); cache_free(c); return -1; }
    for(int i=0;i<N;i++) c->opt_ids[i]=NULL;
    sjev_tokenize(ex->context, m->context_tokens, c->ctx_ids);
    for(int i=0;i<N;i++){
        size_t l = sjev_truncated_len(ex->options[i], m->option_tokens);
        c->opt_lens[i]=l;
        if(l>0){
            c->opt_ids[i]=(uint32_t*)malloc(l*sizeof(uint32_t));
            if(!c->opt_ids[i]){ if(err) snprintf(err,err_len,"oom"); cache_free(c); return -1; }
            sjev_tokenize(ex->options[i], m->option_tokens, c->opt_ids[i]);
        }
    }
    // allocate temporaries
    c->ctx_emb=(float*)malloc((size_t)Lc*W*sizeof(float));
    c->opt_pool=(float*)malloc((size_t)N*W*sizeof(float));
    c->ctx_norm=(float*)malloc((size_t)Lc*W*sizeof(float));
    c->opt_norm=(float*)malloc((size_t)N*W*sizeof(float));
    c->Q=(float*)malloc((size_t)N*R*sizeof(float));
    c->K=(float*)malloc((size_t)Lc*R*sizeof(float));
    c->V=(float*)malloc((size_t)Lc*R*sizeof(float));
    c->scores=(float*)malloc((size_t)N*Lc*sizeof(float));
    c->attn=(float*)malloc((size_t)N*Lc*sizeof(float));
    c->attended=(float*)malloc((size_t)N*R*sizeof(float));
    c->logits=(float*)malloc(N*sizeof(float));
    c->probs=(float*)malloc(N*sizeof(float));
    if(!c->ctx_emb||!c->opt_pool||!c->ctx_norm||!c->opt_norm||!c->Q||!c->K||!c->V||!c->scores||!c->attn||!c->attended||!c->logits||!c->probs){
        if(err) { snprintf(err,err_len,"oom forward"); }
        cache_free(c); return -1;
    }
    // 1 ctx_emb
    for(int i=0;i<Lc;i++){
        uint32_t tid=c->ctx_ids[i];
        const float *emb=m->embedding + (size_t)tid*W;
        const float *pos=m->position + (size_t)i*W;
        float *dst=c->ctx_emb + (size_t)i*W;
        for(uint32_t w=0;w<W;w++) dst[w]=emb[w]+pos[w];
    }
    // 2 opt_pool
    for(int n=0;n<N;n++){
        float *dst=c->opt_pool + (size_t)n*W;
        for(uint32_t w=0;w<W;w++) dst[w]=0;
        size_t ln=c->opt_lens[n];
        if(ln==0) continue;
        for(size_t k=0;k<ln;k++){
            uint32_t tid=c->opt_ids[n][k];
            const float *emb=m->embedding + (size_t)tid*W;
            for(uint32_t w=0;w<W;w++) dst[w]+=emb[w];
        }
        float inv=1.0f/(float)ln;
        for(uint32_t w=0;w<W;w++) dst[w]*=inv;
    }
    // 3 layernorm
    sjev_layernorm(c->ctx_emb, c->ctx_norm, Lc, W, m->ln_c_w, m->ln_c_b);
    sjev_layernorm(c->opt_pool, c->opt_norm, N, W, m->ln_o_w, m->ln_o_b);
    // 4 QKV
    sjev_mat_q(c->opt_norm, m->Wq, c->Q, N, W, R);
    sjev_mat_kv(c->ctx_norm, m->Wk, c->K, Lc, W, R);
    sjev_mat_kv(c->ctx_norm, m->Wv, c->V, Lc, W, R);
    float inv_sqrt = 1.0f / sqrtf((float)R);
    // 5 scores
    for(int n=0;n<N;n++){
        const float *q=c->Q + (size_t)n*R;
        float *sc=c->scores + (size_t)n*Lc;
        for(int l=0;l<Lc;l++){
            const float *k=c->K + (size_t)l*R;
            double dot=0;
            for(uint32_t r=0;r<R;r++) dot += (double)q[r]*k[r];
            sc[l]=(float)(dot * inv_sqrt);
        }
    }
    // 6 attn
    sjev_softmax_rows(c->scores, c->attn, N, Lc);
    // 7 attended
    for(int n=0;n<N;n++){
        float *dst=c->attended + (size_t)n*R;
        for(uint32_t r=0;r<R;r++) dst[r]=0;
        const float *a=c->attn + (size_t)n*Lc;
        for(int l=0;l<Lc;l++){
            float w=a[l];
            const float *v=c->V + (size_t)l*R;
            for(uint32_t r=0;r<R;r++) dst[r]+=w*v[r];
        }
    }
    // 8 logits
    for(int n=0;n<N;n++){
        const float *q=c->Q + (size_t)n*R;
        const float *av=c->attended + (size_t)n*R;
        double dot=0;
        for(uint32_t r=0;r<R;r++) dot += (double)q[r]*av[r];
        c->logits[n]=(float)(dot * inv_sqrt);
    }
    // 9 probs
    sjev_softmax_1d(c->logits, c->probs, N);
    // loss
    float p = c->probs[ex->label];
    if (p < 1e-12f) p=1e-12f;
    c->loss = -logf(p);
    return 0;
}

void cache_free(struct forward_cache *c){
    if(!c) return;
    free(c->opt_lens);
    free(c->ctx_ids);
    if(c->opt_ids){ for(int i=0;i<c->N;i++) free(c->opt_ids[i]); free(c->opt_ids); }
    free(c->ctx_emb); free(c->opt_pool); free(c->ctx_norm); free(c->opt_norm);
    free(c->Q); free(c->K); free(c->V); free(c->scores); free(c->attn); free(c->attended); free(c->logits); free(c->probs);
    memset(c,0,sizeof(*c));
}

int backward_one(const struct sjev_model *m, const struct forward_cache *c, struct sjev_grad *grad, char *err, size_t err_len) {
    uint32_t W=m->width, R=m->rank;
    int Lc=c->Lc, N=c->N;
    // dlogits = probs - one_hot for mean loss (not yet divided by batch); caller will divide after
    float *dlogits = (float*)malloc(N*sizeof(float));
    if(!dlogits){ if(err) snprintf(err,err_len,"oom dlogits"); return -1; }
    for(int n=0;n<N;n++) dlogits[n]=c->probs[n];
    dlogits[c->label] -= 1.0f;
    float inv_sqrt = 1.0f / sqrtf((float)R);
    // allocate intermediate grads
    float *dQ = (float*)calloc((size_t)N*R, sizeof(float));
    float *dK = (float*)calloc((size_t)Lc*R, sizeof(float));
    float *dV = (float*)calloc((size_t)Lc*R, sizeof(float));
    float *d_attended = (float*)calloc((size_t)N*R, sizeof(float));
    float *d_attn = (float*)calloc((size_t)N*Lc, sizeof(float));
    float *d_scores = (float*)calloc((size_t)N*Lc, sizeof(float));
    float *d_opt_norm = (float*)calloc((size_t)N*W, sizeof(float));
    float *d_ctx_norm = (float*)calloc((size_t)Lc*W, sizeof(float));
    if(!dQ||!dK||!dV||!d_attended||!d_attn||!d_scores||!d_opt_norm||!d_ctx_norm){
        free(dlogits); free(dQ); free(dK); free(dV); free(d_attended); free(d_attn); free(d_scores); free(d_opt_norm); free(d_ctx_norm);
        if(err) { snprintf(err,err_len,"oom backward"); }
        return -1;
    }
    // 8) logits -> Q and attended
    for(int n=0;n<N;n++){
        float dl = dlogits[n];
        float *dq = dQ + (size_t)n*R;
        float *da = d_attended + (size_t)n*R;
        const float *q = c->Q + (size_t)n*R;
        const float *av = c->attended + (size_t)n*R;
        for(uint32_t r=0;r<R;r++){
            dq[r] += dl * av[r] * inv_sqrt;
            da[r] = dl * q[r] * inv_sqrt;
        }
    }
    // 7) attended = attn * V -> d_attn and dV
    for(int n=0;n<N;n++){
        const float *da = d_attended + (size_t)n*R;
        float *d_a = d_attn + (size_t)n*Lc;
        for(int l=0;l<Lc;l++){
            float *vv = dV + (size_t)l*R;
            const float *v = c->V + (size_t)l*R;
            float attn = c->attn[(size_t)n*Lc + l];
            // d_attn contribution
            double dot=0;
            for(uint32_t r=0;r<R;r++) dot += (double)da[r] * v[r];
            d_a[l] += (float)dot;
            // dV
            for(uint32_t r=0;r<R;r++) vv[r] += attn * da[r];
        }
    }
    // 6) softmax scores -> attn: d_scores = attn * (d_attn - sum(d_attn*attn))
    for(int n=0;n<N;n++){
        const float *attn = c->attn + (size_t)n*Lc;
        const float *da = d_attn + (size_t)n*Lc;
        float *ds = d_scores + (size_t)n*Lc;
        double sum =0;
        for(int l=0;l<Lc;l++) sum += (double)da[l]*attn[l];
        for(int l=0;l<Lc;l++) ds[l] = attn[l] * (da[l] - (float)sum);
    }
    // 5) scores = Q*K^T / sqrt -> dQ via scores and dK
    for(int n=0;n<N;n++){
        float *dq = dQ + (size_t)n*R;
        const float *ds_row = d_scores + (size_t)n*Lc;
        for(int l=0;l<Lc;l++){
            float ds = ds_row[l] * inv_sqrt;
            const float *k = c->K + (size_t)l*R;
            float *dk = dK + (size_t)l*R;
            for(uint32_t r=0;r<R;r++){
                dq[r] += ds * k[r];
                dk[r] += ds * c->Q[(size_t)n*R + r];
            }
        }
    }
    // 4) QKV linears
    // d_opt_norm from dQ
    for(int n=0;n<N;n++){
        const float *dq = dQ + (size_t)n*R;
        float *d_on = d_opt_norm + (size_t)n*W;
        for(uint32_t r=0;r<R;r++){
            float dqr = dq[r];
            const float *wq = m->Wq + (size_t)r*W;
            for(uint32_t w=0;w<W;w++){
                d_on[w] += dqr * wq[w];
                grad->Wq[(size_t)r*W + w] += dqr * c->opt_norm[(size_t)n*W + w];
            }
        }
    }
    // d_ctx_norm from dK and dV
    for(int l=0;l<Lc;l++){
        float *d_cn = d_ctx_norm + (size_t)l*W;
        const float *dk = dK + (size_t)l*R;
        const float *dv = dV + (size_t)l*R;
        for(uint32_t r=0;r<R;r++){
            float dkr = dk[r];
            float dvr = dv[r];
            const float *wk = m->Wk + (size_t)r*W;
            const float *wv = m->Wv + (size_t)r*W;
            for(uint32_t w=0;w<W;w++){
                if(dkr) d_cn[w] += dkr * wk[w];
                if(dvr) d_cn[w] += dvr * wv[w];
            }
            // grad Wk/Wv accumulation
            const float *cn = c->ctx_norm + (size_t)l*W;
            for(uint32_t w=0;w<W;w++){
                if(dkr) grad->Wk[(size_t)r*W + w] += dkr * cn[w];
                if(dvr) grad->Wv[(size_t)r*W + w] += dvr * cn[w];
            }
        }
    }
    // layernorm backward for ctx and opt
    // Need temporary arrays for d_x to compute dx
    float *d_ctx_emb = (float*)malloc((size_t)Lc*W*sizeof(float));
    float *d_opt_pool = (float*)malloc((size_t)N*W*sizeof(float));
    if(!d_ctx_emb||!d_opt_pool){ free(dlogits);free(dQ);free(dK);free(dV);free(d_attended);free(d_attn);free(d_scores);free(d_opt_norm);free(d_ctx_norm);free(d_ctx_emb);free(d_opt_pool); if(err) snprintf(err,err_len,"oom"); return -1;}
    // zero init gamma/beta grads accumulation will be inside helper, but we need to zero those slices? The grad arrays already zero at batch start, helper adds.
    // For ctx: x = ctx_emb, dout = d_ctx_norm, gamma = ln_c_w
    // implement layernorm backward directly here to add to grad
    {
        const double eps=1e-5;
        for(int r=0;r<Lc;r++){
            const float *xr = c->ctx_emb + (size_t)r*W;
            const float *dr = d_ctx_norm + (size_t)r*W;
            float *dxr = d_ctx_emb + (size_t)r*W;
            double sum=0;
            for(uint32_t w=0;w<W;w++) sum += (double)xr[w];
            double mean = sum / W;
            double var_sum=0;
            for(uint32_t w=0;w<W;w++){ double d=(double)xr[w]-mean; var_sum+=d*d; }
            double var=var_sum/W;
            double inv = 1.0 / sqrt(var + eps);
            float inv_f=(float)inv; float mean_f=(float)mean;
            double sum_dx_hat=0, sum_dx_hat_xhat=0;
            // first pass compute sums and gamma grad
            for(uint32_t w=0;w<W;w++){
                float x_hat=(xr[w]-mean_f)*inv_f;
                float dx_hat=dr[w]*m->ln_c_w[w];
                sum_dx_hat += dx_hat;
                sum_dx_hat_xhat += (double)dx_hat * x_hat;
                grad->ln_c_w[w] += dr[w]*x_hat;
                grad->ln_c_b[w] += dr[w];
            }
            double mean_dx_hat=sum_dx_hat/W;
            double mean_dx_hat_xhat=sum_dx_hat_xhat/W;
            for(uint32_t w=0;w<W;w++){
                float x_hat=(xr[w]-mean_f)*inv_f;
                float dx_hat=dr[w]*m->ln_c_w[w];
                dxr[w]=(float)(inv*( (double)dx_hat - mean_dx_hat - (double)x_hat*mean_dx_hat_xhat));
            }
        }
    }
    {
        const double eps=1e-5;
        for(int r=0;r<N;r++){
            const float *xr = c->opt_pool + (size_t)r*W;
            const float *dr = d_opt_norm + (size_t)r*W;
            float *dxr = d_opt_pool + (size_t)r*W;
            double sum=0;
            for(uint32_t w=0;w<W;w++) sum+=(double)xr[w];
            double mean=sum/W;
            double var_sum=0;
            for(uint32_t w=0;w<W;w++){double d=(double)xr[w]-mean; var_sum+=d*d;}
            double var=var_sum/W;
            double inv=1.0/sqrt(var+eps);
            float inv_f=(float)inv; float mean_f=(float)mean;
            double sum_dx_hat=0,sum_dx_hat_xhat=0;
            for(uint32_t w=0;w<W;w++){
                float x_hat=(xr[w]-mean_f)*inv_f;
                float dx_hat=dr[w]*m->ln_o_w[w];
                sum_dx_hat+=dx_hat;
                sum_dx_hat_xhat+=(double)dx_hat*x_hat;
                grad->ln_o_w[w]+=dr[w]*x_hat;
                grad->ln_o_b[w]+=dr[w];
            }
            double mean_dx_hat=sum_dx_hat/W;
            double mean_dx_hat_xhat=sum_dx_hat_xhat/W;
            for(uint32_t w=0;w<W;w++){
                float x_hat=(xr[w]-mean_f)*inv_f;
                float dx_hat=dr[w]*m->ln_o_w[w];
                dxr[w]=(float)(inv*((double)dx_hat - mean_dx_hat - (double)x_hat*mean_dx_hat_xhat));
            }
        }
    }
    // now d_ctx_emb and d_opt_pool ready
    // accumulate to embedding and position
    for(int l=0;l<Lc;l++){
        float *g_pos = grad->position + (size_t)l*W;
        uint32_t tid = c->ctx_ids[l];
        float *g_emb = grad->embedding + (size_t)tid*W;
        float *d = d_ctx_emb + (size_t)l*W;
        for(uint32_t w=0;w<W;w++){
            g_pos[w] += d[w];
            if(tid!=0) g_emb[w] += d[w];
            // if tid==0, skip (padding)
        }
    }
    for(int n=0;n<N;n++){
        size_t ln = c->opt_lens[n];
        if(ln==0) continue;
        float *d = d_opt_pool + (size_t)n*W;
        float inv = 1.0f/(float)ln;
        for(size_t k=0;k<ln;k++){
            uint32_t tid=c->opt_ids[n][k];
            if(tid==0) continue;
            float *g_emb = grad->embedding + (size_t)tid*W;
            for(uint32_t w=0;w<W;w++) g_emb[w] += d[w]*inv;
        }
    }
    free(dlogits); free(dQ); free(dK); free(dV); free(d_attended); free(d_attn); free(d_scores); free(d_opt_norm); free(d_ctx_norm); free(d_ctx_emb); free(d_opt_pool);
    return 0;
}

float cache_loss(const struct forward_cache *c){ return c->loss; }

int eval_dataset(const struct sjev_model *m, const struct dataset *ds, float *out_nll, float *out_top1, float *out_top3, char *err, size_t err_len) {
    double total_loss=0;
    int correct1=0, correct3=0;
    size_t n=ds->n;
    for(size_t i=0;i<n;i++){
        struct forward_cache c;
        if(forward_one(m, &ds->examples[i], &c, err, err_len)!=0) return -1;
        total_loss += c.loss;
        // top1/top3 from logits
        int best=-1; float bestv=-1e30;
        for(int k=0;k<c.N;k++) if(c.logits[k]>bestv){bestv=c.logits[k]; best=k;}
        if(best==c.label) correct1++;
        // top3
        // simple selection: find top 3
        int top[3]={-1,-1,-1};
        float topv[3]={-1e30,-1e30,-1e30};
        for(int k=0;k<c.N;k++){
            float v=c.logits[k];
            for(int t=0;t<3;t++){
                if(v>topv[t]){
                    for(int s=2;s>t;s--){ topv[s]=topv[s-1]; top[s]=top[s-1]; }
                    topv[t]=v; top[t]=k; break;
                }
            }
        }
        for(int t=0;t<3 && t<c.N;t++) if(top[t]==c.label) {correct3++; break;}
        cache_free(&c);
    }
    if(out_nll) *out_nll = (float)(total_loss / n);
    if(out_top1) *out_top1 = (float)correct1 / (float)n;
    if(out_top3) *out_top3 = (float)correct3 / (float)n;
    return 0;
}

int train_loop(const struct sjev_model *model_init, const struct dataset *train, const struct dataset *valid,
               int epochs, int batch_size, struct adam_state *adam, const char *output_path,
               char *err, size_t err_len) {
    // model_init is template for dims; we will train in-place on a copy
    // Caller owns model_init; we train on mutable model (passed as non-const via cast)
    // For this function, model is mutable
    struct sjev_model *m = (struct sjev_model*)model_init;
    struct sjev_grad grad;
    if(grad_alloc(m,&grad,err,err_len)!=0) return -1;
    // copy best model params
    struct sjev_model *best=NULL;
    // allocate best
    size_t emb_n=(size_t)257*m->width;
    size_t pos_n=(size_t)m->context_tokens*m->width;
    size_t norm_n=m->width;
    size_t qkv_n=(size_t)m->rank*m->width;
    best=(struct sjev_model*)calloc(1,sizeof(*best));
    if(!best){ grad_free(m,&grad); if(err) snprintf(err,err_len,"oom best"); return -1; }
    best->width=m->width; best->rank=m->rank; best->context_tokens=m->context_tokens; best->option_tokens=m->option_tokens;
    best->embedding=(float*)malloc(emb_n*sizeof(float)); best->position=(float*)malloc(pos_n*sizeof(float));
    best->ln_c_w=(float*)malloc(norm_n*sizeof(float)); best->ln_c_b=(float*)malloc(norm_n*sizeof(float));
    best->ln_o_w=(float*)malloc(norm_n*sizeof(float)); best->ln_o_b=(float*)malloc(norm_n*sizeof(float));
    best->Wq=(float*)malloc(qkv_n*sizeof(float)); best->Wk=(float*)malloc(qkv_n*sizeof(float)); best->Wv=(float*)malloc(qkv_n*sizeof(float));
    if(!best->embedding||!best->position||!best->ln_c_w||!best->ln_c_b||!best->ln_o_w||!best->ln_o_b||!best->Wq||!best->Wk||!best->Wv){
        grad_free(m,&grad); sjev_model_free(best); if(err) snprintf(err,err_len,"oom best alloc"); return -1;
    }
    // init best with current
    memcpy(best->embedding,m->embedding,emb_n*sizeof(float));
    memcpy(best->position,m->position,pos_n*sizeof(float));
    memcpy(best->ln_c_w,m->ln_c_w,norm_n*sizeof(float)); memcpy(best->ln_c_b,m->ln_c_b,norm_n*sizeof(float));
    memcpy(best->ln_o_w,m->ln_o_w,norm_n*sizeof(float)); memcpy(best->ln_o_b,m->ln_o_b,norm_n*sizeof(float));
    memcpy(best->Wq,m->Wq,qkv_n*sizeof(float)); memcpy(best->Wk,m->Wk,qkv_n*sizeof(float)); memcpy(best->Wv,m->Wv,qkv_n*sizeof(float));

    float best_val = 1e30f;
    // indices for shuffling
    size_t train_n=train->n;
    size_t *indices=(size_t*)malloc(train_n*sizeof(size_t));
    if(!indices){ grad_free(m,&grad); sjev_model_free(best); if(err) snprintf(err,err_len,"oom indices"); return -1; }
    for(size_t i=0;i<train_n;i++) indices[i]=i;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for(int epoch=0; epoch<epochs; epoch++){
        // shuffle
        // use xorshift with seed = epoch + 0x9e3779b9
        uint64_t rs = 0x9e3779b97f4a7c15ULL + (uint64_t)epoch*0xbf58476d1ce4e5b9ULL;
        for(size_t i=train_n-1;i>0;i--){
            // xorshift
            rs ^= rs>>12; rs ^= rs<<25; rs ^= rs>>27; rs *= 0x2545F4914F6CDD1DULL;
            size_t j = rs % (i+1);
            size_t tmp=indices[i]; indices[i]=indices[j]; indices[j]=tmp;
        }
        double train_loss_sum=0;
        size_t train_cnt=0;
        // iterate batches
        for(size_t start=0; start<train_n; start+=batch_size){
            size_t end = start+batch_size; if(end>train_n) end=train_n;
            size_t blen = end-start;
            grad_zero(m,&grad);
            double batch_loss=0;
            for(size_t b=start;b<end;b++){
                size_t idx=indices[b];
                struct forward_cache c;
                if(forward_one(m, &train->examples[idx], &c, err, err_len)!=0){ free(indices); grad_free(m,&grad); sjev_model_free(best); return -1; }
                batch_loss += c.loss;
                if(backward_one(m, &c, &grad, err, err_len)!=0){ cache_free(&c); free(indices); grad_free(m,&grad); sjev_model_free(best); return -1; }
                cache_free(&c);
            }
            train_loss_sum += batch_loss;
            train_cnt += blen;
            grad_scale(m,&grad, 1.0f/(float)blen);
            grad_clip(m,&grad, 1.0f);
            adam_step(m, &grad, adam);
        }
        float train_nll = (float)(train_loss_sum / train_cnt);
        float val_nll, val_top1, val_top3;
        if(eval_dataset(m, valid, &val_nll, &val_top1, &val_top3, err, err_len)!=0){ free(indices); grad_free(m,&grad); sjev_model_free(best); return -1; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
        printf("{\"epoch\": %d, \"train_nll\": %.6f, \"validation_nll\": %.6f, \"val_top1\": %.4f, \"val_top3\": %.4f, \"elapsed\": %.2f}\n", epoch+1, train_nll, val_nll, val_top1, val_top3, elapsed);
        fflush(stdout);
        if(val_nll < best_val){
            best_val=val_nll;
            memcpy(best->embedding,m->embedding,emb_n*sizeof(float));
            memcpy(best->position,m->position,pos_n*sizeof(float));
            memcpy(best->ln_c_w,m->ln_c_w,norm_n*sizeof(float)); memcpy(best->ln_c_b,m->ln_c_b,norm_n*sizeof(float));
            memcpy(best->ln_o_w,m->ln_o_w,norm_n*sizeof(float)); memcpy(best->ln_o_b,m->ln_o_b,norm_n*sizeof(float));
            memcpy(best->Wq,m->Wq,qkv_n*sizeof(float)); memcpy(best->Wk,m->Wk,qkv_n*sizeof(float)); memcpy(best->Wv,m->Wv,qkv_n*sizeof(float));
        }
    }
    // copy best back to m and save
    memcpy(m->embedding,best->embedding,emb_n*sizeof(float));
    memcpy(m->position,best->position,pos_n*sizeof(float));
    memcpy(m->ln_c_w,best->ln_c_w,norm_n*sizeof(float)); memcpy(m->ln_c_b,best->ln_c_b,norm_n*sizeof(float));
    memcpy(m->ln_o_w,best->ln_o_w,norm_n*sizeof(float)); memcpy(m->ln_o_b,best->ln_o_b,norm_n*sizeof(float));
    memcpy(m->Wq,best->Wq,qkv_n*sizeof(float)); memcpy(m->Wk,best->Wk,qkv_n*sizeof(float)); memcpy(m->Wv,best->Wv,qkv_n*sizeof(float));
    // ensure padding row zero
    for(uint32_t w=0;w<m->width;w++) m->embedding[w]=0;
    int rc = sjev_model_save(m, output_path, err, err_len);
    if(rc==0){
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double total = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
        printf("{\"checkpoint\": \"%s\", \"best_validation_nll\": %.6f, \"total_time\": %.2f}\n", output_path, best_val, total);
    }
    free(indices);
    grad_free(m,&grad);
    sjev_model_free(best);
    return rc;
}
