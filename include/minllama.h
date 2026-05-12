#ifndef MINLLAMA_H
#define MINLLAMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t ml_token;

typedef struct ml_model ml_model;
typedef struct ml_context ml_context;

typedef enum ml_status {
    ML_OK = 0,
    ML_ERROR_INVALID_ARGUMENT = -1,
    ML_ERROR_BUFFER_TOO_SMALL = -2,
    ML_ERROR_NOT_IMPLEMENTED = -3
} ml_status;

ml_model *ml_model_load(const char *path);
void ml_model_free(ml_model *model);

ml_context *ml_context_create(ml_model *model);
void ml_context_free(ml_context *ctx);
void ml_context_reset(ml_context *ctx);

int ml_tokenize(ml_model *model, const char *text, ml_token *out_tokens, size_t capacity);
int ml_prefill(ml_context *ctx, const ml_token *tokens, size_t count);
int ml_next_token(ml_context *ctx, ml_token *out_token);
const float *ml_get_logits(ml_context *ctx, size_t *out_count);
int ml_detokenize(ml_model *model, ml_token token, char *out_text, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
