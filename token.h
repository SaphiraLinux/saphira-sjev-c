#pragma once
#include <stddef.h>
#include <stdint.h>

/* Byte tokenisation compatible with jevlike/data.py:_bytes
 * (independent implementation; see ARCHITECTURE.md and NOTICE).
 * - UTF-8 encode is implicit: the C string bytes are the UTF-8 bytes.
 * - truncate to max_len bytes
 * - map byte b -> b+1, 0 reserved for padding
 *
 * out_ids must have capacity >= max_len. Returns actual length.
 */
size_t sjev_tokenize(const char *text, uint32_t max_len, uint32_t *out_ids);

/* Helper: byte length truncated */
size_t sjev_truncated_len(const char *text, uint32_t max_len);
