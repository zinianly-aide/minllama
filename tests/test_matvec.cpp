#include "minllama_internal.h"
#include "test_gguf_util.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

constexpr std::uint64_t kMetadataCount = 12;

bool write_matvec_gguf(const char *path) {
    const std::vector<minllama_test::TensorInfoSpec> tensors = {
        {"f32.weight", 2, {3, 2}, 0, 0},
        {"f16.weight", 2, {3, 2}, 1, 24},
        {"q4_0.weight", 2, {16, 2}, 2, 36},
        {"f32.vector", 1, {3}, 0, 54},
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

    minllama_test::write_tensor_payload_f32(out, {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 4.0f,
    });
    minllama_test::write_tensor_payload_f16(out, {
        0x3c00u, 0x4000u, 0x4200u,
        0xbc00u, 0x3800u, 0x4400u,
    });
    if (!minllama_test::write_tensor_payload_q4_0_block(
        out,
        0x3c00u,
        {1, 0, -1, 2, 3, -2, 1, 0, -3, 4, 2, -1, 0, 1, -2, 3,
         0, 1,  2, 3, 4,  5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8})) {
        return false;
    }
    minllama_test::write_tensor_payload_f32(out, {0.0f, 0.0f, 0.0f});
    return static_cast<bool>(out);
}

void assert_close(float actual, float expected) {
    assert(std::fabs(actual - expected) < 0.00001f);
}

void assert_vector_close(const std::vector<float> &actual, const std::vector<float> &expected) {
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert_close(actual[i], expected[i]);
    }
}

} // namespace

int main() {
    const char *path = "test_matvec_valid.gguf";
    std::remove(path);

    assert(write_matvec_gguf(path));
    ml_model *model = ml_model_load(path);
    assert(model != nullptr);

    const std::vector<float> input = {2.0f, -1.0f, 0.5f};
    std::vector<float> out;

    assert(minllama::matvec_tensor_as_f32(path, model->tensor_index, "f32.weight", input, &out));
    assert_vector_close(out, {1.5f, -0.5f});

    out.clear();
    assert(minllama::matvec_tensor_as_f32(path, model->tensor_index, "f16.weight", input, &out));
    assert_vector_close(out, {1.5f, -0.5f});

    const std::vector<float> q40_input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
    };
    out.clear();
    assert(minllama::matvec_q40_f32(path, model->tensor_index, "q4_0.weight", q40_input, &out));
    assert_vector_close(out, {71.0f, -324.0f});

    out.clear();
    assert(!minllama::matvec_tensor_as_f32(path, model->tensor_index, "f32.weight", {1.0f, 2.0f}, &out));
    assert(!minllama::matvec_tensor_as_f32(path, model->tensor_index, "f32.vector", input, &out));
    assert(!minllama::matvec_q40_f32(path, model->tensor_index, "f32.weight", input, &out));

    const float f32_matrix[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float direct_out[] = {0.0f, 0.0f};
    assert(minllama::matvec_f32_f32(f32_matrix, 2, 2, input.data(), 2, direct_out, 2));
    assert_close(direct_out[0], 0.0f);
    assert_close(direct_out[1], 2.0f);
    assert(!minllama::matvec_f32_f32(f32_matrix, 2, 2, input.data(), 3, direct_out, 2));

    const std::uint16_t f16_matrix[] = {0x3c00u, 0x4000u, 0x4200u, 0x4400u};
    assert(minllama::matvec_f16_f32(f16_matrix, 2, 2, input.data(), 2, direct_out, 2));
    assert_close(direct_out[0], 0.0f);
    assert_close(direct_out[1], 2.0f);

    ml_model_free(model);
    std::remove(path);
    return 0;
}
