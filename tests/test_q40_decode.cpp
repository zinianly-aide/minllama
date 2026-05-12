#include "minllama_internal.h"
#include "test_gguf_util.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

constexpr std::uint64_t kMetadataCount = 12;

bool write_q40_gguf(const char *path) {
    const std::vector<minllama_test::TensorInfoSpec> tensors = {
        {"q4_0.weight", 1, {32}, 2, 0},
    };

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }

    out.write("GGUF", 4);
    minllama_test::write_u32_le(out, 3);
    minllama_test::write_u64_le(out, static_cast<std::uint64_t>(tensors.size()));
    minllama_test::write_u64_le(out, kMetadataCount);
    minllama_test::write_llama_metadata(out, "llama");

    for (const minllama_test::TensorInfoSpec &tensor : tensors) {
        minllama_test::write_tensor_info(out, tensor);
    }

    return minllama_test::write_tensor_payload_q4_0_block(
        out,
        0x3c00u,
        {-8, -7, -6, -5, -4, -3, -2, -1,
          0,  1,  2,  3,  4,  5,  6,  7,
          7,  6,  5,  4,  3,  2,  1,  0,
         -1, -2, -3, -4, -5, -6, -7, -8});
}

bool write_incomplete_q40_gguf(const char *path) {
    const std::vector<minllama_test::TensorInfoSpec> tensors = {
        {"q4_0.weight", 1, {32}, 2, 0},
    };

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }

    out.write("GGUF", 4);
    minllama_test::write_u32_le(out, 3);
    minllama_test::write_u64_le(out, static_cast<std::uint64_t>(tensors.size()));
    minllama_test::write_u64_le(out, kMetadataCount);
    minllama_test::write_llama_metadata(out, "llama");

    for (const minllama_test::TensorInfoSpec &tensor : tensors) {
        minllama_test::write_tensor_info(out, tensor);
    }

    minllama_test::write_f16_le(out, 0x3c00u);
    minllama_test::write_tensor_data_padding(out, 15);
    return static_cast<bool>(out);
}

void assert_close(float actual, float expected) {
    assert(std::fabs(actual - expected) < 0.00001f);
}

} // namespace

int main() {
    const char *valid_path = "test_q40_decode_valid.gguf";
    const char *incomplete_path = "test_q40_decode_incomplete.gguf";
    std::remove(valid_path);
    std::remove(incomplete_path);

    assert(write_q40_gguf(valid_path));
    ml_model *model = ml_model_load(valid_path);
    assert(model != nullptr);

    std::vector<float> values;
    assert(minllama::load_tensor_as_f32(valid_path, model->tensor_index, "q4_0.weight", &values));
    assert(values.size() == 32);

    const std::vector<float> expected = {
        -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
         0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
         7.0f,  6.0f,  5.0f,  4.0f,  3.0f,  2.0f,  1.0f,  0.0f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f,
    };
    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert_close(values[i], expected[i]);
    }

    assert(write_incomplete_q40_gguf(incomplete_path));
    assert(ml_model_load(incomplete_path) == nullptr);

    {
        std::ofstream truncated(valid_path, std::ios::binary | std::ios::trunc);
        truncated.write("GGUF", 4);
    }
    assert(!minllama::load_tensor_as_f32(valid_path, model->tensor_index, "q4_0.weight", &values));

    ml_model_free(model);
    std::remove(valid_path);
    std::remove(incomplete_path);
    return 0;
}
