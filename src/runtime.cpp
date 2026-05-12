#include "minllama_internal.h"

#include <new>

ml_context *ml_context_create(ml_model *model) {
    auto *ctx = new (std::nothrow) ml_context();
    if (!ctx) {
        return nullptr;
    }
    ctx->model = model;
    return ctx;
}

void ml_context_free(ml_context *ctx) {
    delete ctx;
}

void ml_context_reset(ml_context *ctx) {
    if (!ctx) {
        return;
    }
    ++ctx->reset_count;
}

int ml_prefill(ml_context *ctx, const ml_token *tokens, size_t count) {
    if (!ctx || (!tokens && count > 0)) {
        return ML_ERROR_INVALID_ARGUMENT;
    }

    // Skeleton boundary: prompt evaluation will be implemented with model tensors.
    return ML_ERROR_NOT_IMPLEMENTED;
}

int ml_next_token(ml_context *ctx, ml_token *out_token) {
    if (!ctx || !out_token) {
        return ML_ERROR_INVALID_ARGUMENT;
    }

    // Skeleton boundary: decode cannot produce logits until inference exists.
    *out_token = 0;
    return ML_ERROR_NOT_IMPLEMENTED;
}

const float *ml_get_logits(ml_context *ctx, size_t *out_count) {
    if (out_count) {
        *out_count = 0;
    }
    if (!ctx) {
        return nullptr;
    }

    // No logits are available in the skeleton build.
    return nullptr;
}
