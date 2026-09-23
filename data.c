/*
 * data.c — SJEV JSONL dataset loading.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the Business Source License 1.1 — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 * Developed for SHAMPOO, the Shared Human-Agent-Model Platform for
 * Orchestration and Operations (https://shampoo.op2.uk/).
 */
#define _POSIX_C_SOURCE 200809L
#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>

/* JSON string decoding utilities */

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int encode_utf8(uint32_t cp, char *out) {
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        return 1;
    } else if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp <= 0xFFFF) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else if (cp <= 0x10FFFF) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    } else {
        // replacement char U+FFFD
        out[0] = (char)0xEF; out[1]=(char)0xBF; out[2]=(char)0xBD;
        return 3;
    }
}

// Decode JSON string starting at *p which points at opening '"'
// Returns allocated NUL-terminated UTF-8 string, updates *p to after closing '"'
// On error returns NULL
static char *decode_json_string(const char **pp, char *err, size_t err_len) {
    const char *p = *pp;
    if (*p != '"') {
        snprintf(err, err_len, "expected opening quote");
        return NULL;
    }
    p++;
    // allocate buffer growing
    size_t cap = 256, len = 0;
    char *buf = (char*)malloc(cap);
    if (!buf) {
        snprintf(err, err_len, "oom");
        return NULL;
    }
    while (1) {
        char c = *p;
        if (c == '\0') {
            snprintf(err, err_len, "unterminated string");
            free(buf);
            return NULL;
        }
        if (c == '"') {
            p++;
            break;
        }
        if (c == '\\') {
            p++;
            char esc = *p;
            if (esc == '\0') {
                snprintf(err, err_len, "unterminated escape");
                free(buf);
                return NULL;
            }
            if (esc == '"' || esc == '\\' || esc == '/') {
                if (len + 1 >= cap) { cap*=2; char *nb=(char*)realloc(buf, cap); if(!nb){free(buf); snprintf(err,err_len,"oom"); return NULL;} buf=nb; }
                buf[len++] = esc;
                p++;
            } else if (esc == 'b') { if (len+1>=cap){cap*=2; char*nb=(char*)realloc(buf,cap); if(!nb){free(buf);snprintf(err,err_len,"oom");return NULL;}buf=nb;} buf[len++]='\b'; p++; }
            else if (esc == 'f') { if (len+1>=cap){cap*=2; char*nb=(char*)realloc(buf,cap); if(!nb){free(buf);snprintf(err,err_len,"oom");return NULL;}buf=nb;} buf[len++]='\f'; p++; }
            else if (esc == 'n') { if (len+1>=cap){cap*=2; char*nb=(char*)realloc(buf,cap); if(!nb){free(buf);snprintf(err,err_len,"oom");return NULL;}buf=nb;} buf[len++]='\n'; p++; }
            else if (esc == 'r') { if (len+1>=cap){cap*=2; char*nb=(char*)realloc(buf,cap); if(!nb){free(buf);snprintf(err,err_len,"oom");return NULL;}buf=nb;} buf[len++]='\r'; p++; }
            else if (esc == 't') { if (len+1>=cap){cap*=2; char*nb=(char*)realloc(buf,cap); if(!nb){free(buf);snprintf(err,err_len,"oom");return NULL;}buf=nb;} buf[len++]='\t'; p++; }
            else if (esc == 'u') {
                // parse 4 hex
                if (!isxdigit((unsigned char)p[1]) || !isxdigit((unsigned char)p[2]) || !isxdigit((unsigned char)p[3]) || !isxdigit((unsigned char)p[4])) {
                    snprintf(err, err_len, "invalid \\u escape");
                    free(buf);
                    return NULL;
                }
                int h0=hex_val(p[1]), h1=hex_val(p[2]), h2=hex_val(p[3]), h3=hex_val(p[4]);
                uint32_t cp = (h0<<12)|(h1<<8)|(h2<<4)|h3;
                p+=5; // consumed u + 4 hex
                // check surrogate pair
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    // expect next \uXXXX for low surrogate
                    if (p[0]=='\\' && p[1]=='u' && isxdigit((unsigned char)p[2]) && isxdigit((unsigned char)p[3]) && isxdigit((unsigned char)p[4]) && isxdigit((unsigned char)p[5])) {
                        int l0=hex_val(p[2]), l1=hex_val(p[3]), l2=hex_val(p[4]), l3=hex_val(p[5]);
                        uint32_t lo = (l0<<12)|(l1<<8)|(l2<<4)|l3;
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800)<<10) + (lo - 0xDC00);
                            p+=6;
                        } else {
                            cp = 0xFFFD;
                        }
                    } else {
                        cp = 0xFFFD;
                    }
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    cp = 0xFFFD;
                }
                char utf8[4];
                int n = encode_utf8(cp, utf8);
                if (len + n >= cap) { while (len+n >= cap) cap*=2; char *nb=(char*)realloc(buf, cap); if(!nb){free(buf);snprintf(err,err_len,"oom");return NULL;} buf=nb; }
                memcpy(buf+len, utf8, n);
                len+=n;
            } else {
                snprintf(err, err_len, "invalid escape \\%c", esc);
                free(buf);
                return NULL;
            }
        } else {
            // raw byte, must be valid UTF-8 pass-through (we just copy bytes)
            // Ensure we don't allow unescaped control chars (<0x20) per JSON spec, but Python may emit raw UTF-8
            if ((unsigned char)c < 0x20) {
                snprintf(err, err_len, "unescaped control char");
                free(buf);
                return NULL;
            }
            if (len+1 >= cap) { cap*=2; char*nb=(char*)realloc(buf, cap); if(!nb){free(buf);snprintf(err,err_len,"oom");return NULL;} buf=nb; }
            buf[len++]=c;
            p++;
        }
    }
    if (len+1 >= cap) { cap+=1; char*nb=(char*)realloc(buf, cap); if(!nb){free(buf);snprintf(err,err_len,"oom");return NULL;} buf=nb; }
    buf[len]='\0';
    *pp = p;
    return buf;
}

static void skip_ws(const char **pp) {
    const char *p=*pp;
    while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') p++;
    *pp=p;
}

static int parse_int(const char **pp, int *out, char *err, size_t err_len) {
    const char *p=*pp;
    skip_ws(&p);
    char *end;
    long v = strtol(p, &end, 10);
    if (end==p) { snprintf(err, err_len, "expected integer"); return -1; }
    *out = (int)v;
    *pp = end;
    return 0;
}

// Parse one JSON line into example. Returns 0 on success.
static int parse_line(const char *line, struct example *ex, char *err, size_t err_len) {
    const char *p = line;
    skip_ws(&p);
    if (*p!='{') { snprintf(err, err_len, "expected '{'"); return -1; }
    p++;
    char *context = NULL;
    char **options = NULL;
    int n_opts = 0, cap_opts = 0;
    int label = -1;
    int have_context=0, have_options=0, have_label=0;
    while (1) {
        skip_ws(&p);
        if (*p=='}') { p++; break; }
        if (*p!='"') { snprintf(err, err_len, "expected key string"); goto fail; }
        char *key = decode_json_string(&p, err, err_len);
        if (!key) goto fail;
        skip_ws(&p);
        if (*p!=':') { snprintf(err, err_len, "expected ':'"); free(key); goto fail; }
        p++;
        skip_ws(&p);
        if (strcmp(key,"context")==0) {
            if (*p!='"') { snprintf(err, err_len, "context must be string"); free(key); goto fail; }
            char *val = decode_json_string(&p, err, err_len);
            if (!val) { free(key); goto fail; }
            if (context) free(context);
            context = val;
            have_context=1;
        } else if (strcmp(key,"options")==0) {
            if (*p!='[') { snprintf(err, err_len, "options must be array"); free(key); goto fail; }
            p++;
            // free previous if any
            if (options) { for(int i=0;i<n_opts;i++) free(options[i]); free(options); options=NULL; n_opts=0; cap_opts=0; }
            while (1) {
                skip_ws(&p);
                if (*p==']') { p++; break; }
                if (*p!='"') { snprintf(err, err_len, "option must be string"); free(key); goto fail; }
                char *opt = decode_json_string(&p, err, err_len);
                if (!opt) { free(key); goto fail; }
                if (opt[0]=='\0') { snprintf(err, err_len, "option must be non-empty"); free(opt); free(key); goto fail; }
                if (n_opts >= cap_opts) { int ncap = cap_opts?cap_opts*2:4; char **nb=(char**)realloc(options, ncap*sizeof(char*)); if(!nb){free(opt); free(key); snprintf(err,err_len,"oom"); goto fail;} options=nb; cap_opts=ncap; }
                options[n_opts++]=opt;
                skip_ws(&p);
                if (*p==',') { p++; continue; }
                else if (*p==']') { p++; break; }
                else { snprintf(err, err_len, "expected ',' or ']' in options"); free(key); goto fail; }
            }
            have_options=1;
        } else if (strcmp(key,"label")==0) {
            int v;
            if (parse_int(&p, &v, err, err_len)!=0) { free(key); goto fail; }
            label = v;
            have_label=1;
        } else {
            // unknown key, skip value (string/object/array/int)
            // For our files unknown keys not expected, but skip generically
            if (*p=='"') {
                char *tmp=decode_json_string(&p, err, err_len);
                if(!tmp){free(key); goto fail;}
                free(tmp);
            } else if (*p=='[') {
                int depth=0;
                while (*p) {
                    if (*p=='"') { char *tmp=decode_json_string(&p, err, err_len); if(!tmp){free(key); goto fail;} free(tmp); continue; }
                    if (*p=='[') depth++;
                    if (*p==']') { depth--; p++; if(depth==0) break; else continue; }
                    p++;
                }
            } else if (*p=='{') {
                int depth=0;
                while (*p) {
                    if (*p=='"') { char *tmp=decode_json_string(&p, err, err_len); if(!tmp){free(key); goto fail;} free(tmp); continue; }
                    if (*p=='{') depth++;
                    if (*p=='}') { depth--; p++; if(depth==0) break; else continue; }
                    p++;
                }
            } else {
                // number, skip
                while (*p && *p!=',' && *p!='}' && !isspace((unsigned char)*p)) p++;
            }
        }
        free(key);
        skip_ws(&p);
        if (*p==',') { p++; continue; }
        else if (*p=='}') { p++; break; }
        else if (*p=='\0') { snprintf(err, err_len, "unexpected end"); goto fail; }
        else { snprintf(err, err_len, "expected ',' or '}'"); goto fail; }
    }
    if (!have_context) { snprintf(err, err_len, "missing context"); goto fail; }
    if (!have_options) { snprintf(err, err_len, "missing options"); goto fail; }
    if (!have_label) { snprintf(err, err_len, "missing label"); goto fail; }
    if (n_opts < 2) { snprintf(err, err_len, "options must have at least 2"); goto fail; }
    if (label <0 || label >= n_opts) { snprintf(err, err_len, "label out of range"); goto fail; }
    ex->context = context;
    ex->options = options;
    ex->n_options = n_opts;
    ex->label = label;
    return 0;
fail:
    if (context) free(context);
    if (options) { for(int i=0;i<n_opts;i++) free(options[i]); free(options); }
    return -1;
}

int dataset_load(const char *path, struct dataset *out, char *err, size_t err_len) {
    if (!path || !out) { snprintf(err, err_len, "invalid args"); return -1; }
    memset(out,0,sizeof(*out));
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, err_len, "open %s: %s", path, strerror(errno)); return -1; }
    char *line=NULL;
    size_t linecap=0;
    ssize_t linelen;
    size_t cap=0, n=0;
    struct example *arr=NULL;
    size_t lineno=0;
    // Use getline if available, else fallback
    while ((linelen = getline(&line, &linecap, f)) != -1) {
        lineno++;
        // trim whitespace, skip empty
        char *p=line;
        while (*p && isspace((unsigned char)*p)) p++;
        char *end=line+linelen;
        while (end>p && isspace((unsigned char)end[-1])) end--;
        if (p==end) continue;
        // ensure NUL termination at end
        *end='\0';
        struct example ex;
        memset(&ex,0,sizeof(ex));
        char perr[512]={0};
        if (parse_line(p, &ex, perr, sizeof(perr))!=0) {
            snprintf(err, err_len, "%s:%zu: %s", path, lineno, perr[0]?perr:"parse error");
            free(line);
            // free already parsed
            for(size_t i=0;i<n;i++) { free(arr[i].context); for(int k=0;k<arr[i].n_options;k++) free(arr[i].options[k]); free(arr[i].options); }
            free(arr);
            fclose(f);
            return -1;
        }
        if (n >= cap) {
            size_t ncap = cap? cap*2: 64;
            struct example *nb = (struct example*)realloc(arr, ncap*sizeof(struct example));
            if (!nb) { snprintf(err, err_len, "oom"); free(ex.context); for(int k=0;k<ex.n_options;k++) free(ex.options[k]); free(ex.options); free(line); for(size_t i=0;i<n;i++){free(arr[i].context); for(int k=0;k<arr[i].n_options;k++) free(arr[i].options[k]); free(arr[i].options);} free(arr); fclose(f); return -1; }
            arr=nb; cap=ncap;
        }
        arr[n++]=ex;
    }
    free(line);
    fclose(f);
    if (n==0) { snprintf(err, err_len, "no examples in %s", path); free(arr); return -1; }
    out->examples = arr;
    out->n = n;
    return 0;
}

void dataset_free(struct dataset *ds) {
    if (!ds) return;
    for(size_t i=0;i<ds->n;i++) {
        free(ds->examples[i].context);
        for(int k=0;k<ds->examples[i].n_options;k++) free(ds->examples[i].options[k]);
        free(ds->examples[i].options);
    }
    free(ds->examples);
    ds->examples=NULL;
    ds->n=0;
}
