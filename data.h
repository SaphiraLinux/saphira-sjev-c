/*
 * data.h — SJEV dataset structure and loading interface.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the Business Source License 1.1 — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 * Developed for SHAMPOO, the Shared Human-Agent-Model Platform for
 * Orchestration and Operations (https://shampoo.op2.uk/).
 */
#pragma once
#include <stddef.h>

struct example {
    char *context;      // decoded UTF-8, malloced
    char **options;     // array of decoded strings
    int n_options;
    int label;
};

struct dataset {
    struct example *examples;
    size_t n;
};

int dataset_load(const char *path, struct dataset *out, char *err, size_t err_len);
void dataset_free(struct dataset *ds);
