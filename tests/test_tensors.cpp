#include "minllama_internal.h"
#include "test_gguf_util.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

void expect_fails(const char *path) {
    ml_model *model = ml_model_load(path);
    assert(model == nullptr);
}

std::vector<minllama_test::TensorInfoSpec> sample_tensors() {
    return {
        {"token_embd.weight", 2, {4, 3}, 0, 0},
        {"blk.0.attn_q.weight", 2, {4, 4}, 1, 64},
        {"output_norm.weight", 1, {4}, 0, 128},
    };
}

bool write_truncated_tensor_info_gguf(const char *path) {
    constexpr std::uint64_t n_kv = 12;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }

    out.write("GGUF", 4);
    minllama_test::write_u32_le(out, 3);
    minllama_test::write_u64_le(out, 1);
    minllama_test::write_u64_le(out, n_kv);
    minllama_test::write_llama_metadata(out, "llama");

    minllama_test::write_gguf_string(out, "truncated.weight");
    minllama_test::write_u32_le(out, 2);
    minllama_test::write_u64_le(out, 4096);
    // Missing second dim, tensor type, and offset.
    return static_cast<bool>(out);
}

} // namespace

int main() {
    const char *valid_path = "test_tensors_valid.gguf";
    const char *duplicate_path = "test_tensors_duplicate.gguf";
    const char *bad_dims_path = "test_tensors_bad_dims.gguf";
    const char *truncated_path = "test_tensors_truncated.gguf";

    std::remove(valid_path);
    std::remove(duplicate_path);
    std::remove(bad_dims_path);
    std::remove(truncated_path);

    assert(minllama_test::write_fake_llama_gguf_with_tensors(valid_path, sample_tensors(), "llama", 144));
    ml_model *model = ml_model_load(valid_path);
    assert(model != nullptr);
    assert(model->tensor_index.size() == 3);

    const TensorInfo *attn_q = model->tensor_index.find("blk.0.attn_q.weight");
    assert(attn_q != nullptr);
    assert(attn_q->name == "blk.0.attn_q.weight");
    assert(attn_q->gguf_type == 1);
    assert(attn_q->n_dims == 2);
    assert(attn_q->dims[0] == 4);
    assert(attn_q->dims[1] == 4);
    assert(attn_q->offset == 64);
    assert(model->tensor_index.find("missing.weight") == nullptr);
    ml_model_free(model);

    std::vector<minllama_test::TensorInfoSpec> duplicate = sample_tensors();
    duplicate[2].name = duplicate[0].name;
    assert(minllama_test::write_fake_llama_gguf_with_tensors(duplicate_path, duplicate));
    expect_fails(duplicate_path);

    std::vector<minllama_test::TensorInfoSpec> bad_dims = {
        {"too_many_dims.weight", 5, {1, 2, 3, 4, 5}, 0, 0},
    };
    assert(minllama_test::write_fake_llama_gguf_with_tensors(bad_dims_path, bad_dims));
    expect_fails(bad_dims_path);

    assert(write_truncated_tensor_info_gguf(truncated_path));
    expect_fails(truncated_path);

    std::remove(valid_path);
    std::remove(duplicate_path);
    std::remove(bad_dims_path);
    std::remove(truncated_path);
    return 0;
}
