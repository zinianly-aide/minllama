#include "minllama_internal.h"
#include "test_gguf_util.h"

#include <cassert>
#include <cstdio>
#include <vector>

namespace {

void expect_fails(const char *path) {
    ml_model *model = ml_model_load(path);
    assert(model == nullptr);
}

std::vector<minllama_test::TensorInfoSpec> data_view_tensors() {
    return {
        {"f32.weight", 1, {4}, 0, 0},
        {"f16.weight", 2, {2, 2}, 1, 16},
        {"q4_0.weight", 1, {32}, 2, 24},
    };
}

} // namespace

int main() {
    const char *valid_path = "test_tensor_data_valid.gguf";
    const char *truncated_path = "test_tensor_data_truncated.gguf";
    const char *unsupported_path = "test_tensor_data_unsupported.gguf";

    std::remove(valid_path);
    std::remove(truncated_path);
    std::remove(unsupported_path);

    assert(minllama_test::write_fake_llama_gguf_with_tensors(valid_path, data_view_tensors(), "llama", 42));
    ml_model *model = ml_model_load(valid_path);
    assert(model != nullptr);
    assert(model->tensor_index.size() == 3);

    const TensorView *f32 = model->tensor_index.find_view("f32.weight");
    const TensorView *f16 = model->tensor_index.find_view("f16.weight");
    const TensorView *q4_0 = model->tensor_index.find_view("q4_0.weight");
    assert(f32 != nullptr);
    assert(f16 != nullptr);
    assert(q4_0 != nullptr);

    assert(f32->data_begin == model->gguf.data_offset);
    assert(f32->byte_size == 16);
    assert(f32->data_end == f32->data_begin + f32->byte_size);

    assert(f16->data_begin == model->gguf.data_offset + 16);
    assert(f16->byte_size == 8);
    assert(f16->data_end == f16->data_begin + f16->byte_size);

    assert(q4_0->data_begin == model->gguf.data_offset + 24);
    assert(q4_0->byte_size == 18);
    assert(q4_0->data_end == q4_0->data_begin + q4_0->byte_size);
    ml_model_free(model);

    assert(minllama_test::write_fake_llama_gguf_with_tensors(
        truncated_path, {{"short.weight", 1, {4}, 0, 0}}, "llama", 15));
    expect_fails(truncated_path);

    assert(minllama_test::write_fake_llama_gguf_with_tensors(
        unsupported_path, {{"unsupported.weight", 1, {4}, 99, 0}}, "llama", 64));
    expect_fails(unsupported_path);

    std::remove(valid_path);
    std::remove(truncated_path);
    std::remove(unsupported_path);
    return 0;
}
