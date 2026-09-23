#include "token.h"
#include <string.h>

size_t sjev_truncated_len(const char *text, uint32_t max_len) {
    if (!text) return 0;
    size_t n = strlen(text);
    if (n > max_len) n = max_len;
    return n;
}

size_t sjev_tokenize(const char *text, uint32_t max_len, uint32_t *out_ids) {
    if (!text || !out_ids) return 0;
    size_t n = strlen(text);
    if (n > max_len) n = max_len;
    for (size_t i = 0; i < n; i++) {
        unsigned char b = (unsigned char)text[i];
        out_ids[i] = (uint32_t)b + 1u;
    }
    return n;
}
