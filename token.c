/*
 * token.c — SJEV byte tokenisation.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the Business Source License 1.1 — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 * Developed for SHAMPOO, the Shared Human-Agent-Model Platform for
 * Orchestration and Operations (https://shampoo.op2.uk/).
 */
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
