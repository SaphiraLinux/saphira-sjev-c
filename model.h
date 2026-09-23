#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct sjev_model {
    uint32_t width;
    uint32_t rank;
    uint32_t context_tokens;
    uint32_t option_tokens;
    float *embedding;   // [257 * W]
    float *position;    // [C * W]
    float *ln_c_w;      // [W]
    float *ln_c_b;      // [W]
    float *ln_o_w;      // [W]
    float *ln_o_b;      // [W]
    float *Wq;          // [R * W]
    float *Wk;          // [R * W]
    float *Wv;          // [R * W]
};

/* Load model from file. Returns 0 on success, -1 on failure with err filled. */
int sjev_model_load(const char *path, struct sjev_model **out, char *err, size_t err_len);
void sjev_model_free(struct sjev_model *m);
int sjev_model_save(const struct sjev_model *m, const char *path, char *err, size_t err_len);
int sjev_model_init(struct sjev_model **out, uint32_t width, uint32_t rank, uint32_t context_tokens, uint32_t option_tokens, uint64_t seed, char *err, size_t err_len);

/* Inference: logits and probs are caller-allocated [n_options]. */
int sjev_predict(const struct sjev_model *m,
                 const char *context,
                 const char * const *options,
                 int n_options,
                 float *out_logits,
                 float *out_probs,
                 char *err, size_t err_len);

#ifdef __cplusplus
}
#endif
