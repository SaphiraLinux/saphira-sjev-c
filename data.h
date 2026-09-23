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
