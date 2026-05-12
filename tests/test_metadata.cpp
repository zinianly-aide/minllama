#include "minllama_internal.h"
#include "test_gguf_util.h"

#include <cassert>
#include <cmath>
#include <cstdio>

namespace {

void expect_fails(const char *path) {
    ml_model *model = ml_model_load(path);
    assert(model == nullptr);
}

} // namespace

int main() {
    const char *valid_path = "test_metadata_valid.gguf";
    const char *bad_arch_path = "test_metadata_bad_arch.gguf";

    std::remove(valid_path);
    std::remove(bad_arch_path);

    assert(minllama_test::write_fake_llama_gguf(valid_path));
    ml_model *model = ml_model_load(valid_path);
    assert(model != nullptr);
    assert(model->config.n_vocab == 3);
    assert(model->config.n_layer == 32);
    assert(model->config.n_embd == 4096);
    assert(model->config.n_head == 32);
    assert(model->config.n_head_kv == 32);
    assert(model->config.n_ctx_train == 4096);
    assert(std::fabs(model->config.rope_theta - 10000.0f) < 0.001f);
    assert(std::fabs(model->config.rms_norm_eps - 0.000001f) < 0.0000001f);
    ml_model_free(model);

    assert(minllama_test::write_fake_llama_gguf(bad_arch_path, "gptneox"));
    expect_fails(bad_arch_path);

    std::remove(valid_path);
    std::remove(bad_arch_path);
    return 0;
}
