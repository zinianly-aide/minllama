#include "minllama.h"
#include "test_gguf_util.h"

#include <cassert>
#include <cstring>
#include <cstdio>

int main() {
    const char *model_path = "test_smoke_minimal.gguf";
    {
        bool ok = minllama_test::write_fake_llama_gguf(model_path);
        assert(ok);
    }

    ml_model *model = ml_model_load(model_path);
    assert(model != nullptr);

    ml_context *ctx = ml_context_create(model);
    assert(ctx != nullptr);
    ml_context_reset(ctx);

    ml_token tokens[1] = {};
    const int token_count = ml_tokenize(model, "hello", tokens, 1);
    assert(token_count == 1);
    assert(tokens[0] == 1);

    char text[32] = {};
    const int text_len = ml_detokenize(model, tokens[0], text, sizeof(text));
    assert(text_len == static_cast<int>(std::strlen("<skeleton>")));
    assert(std::strcmp(text, "<skeleton>") == 0);

    assert(ml_prefill(ctx, tokens, 1) == ML_ERROR_NOT_IMPLEMENTED);

    size_t logits_count = 123;
    assert(ml_get_logits(ctx, &logits_count) == nullptr);
    assert(logits_count == 0);

    ml_context_free(ctx);
    ml_model_free(model);
    std::remove(model_path);
    return 0;
}
