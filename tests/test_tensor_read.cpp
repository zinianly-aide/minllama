#include "minllama_internal.h"
#include "test_gguf_util.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

constexpr std::uint64_t kMetadataCount = 12;

bool write_tensor_read_gguf(const char *path) {
    std::vector<minllama_test::TensorInfoSpec> tensors = {
        {"f32.weight", 1, {3}, 0, 0},
        {"f16.weight", 2, {2, 2}, 1, 12},
        {"q4_0.weight", 1, {32}, 2, 20},
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

    minllama_test::write_alignment_padding(out, 32);

    minllama_test::write_tensor_payload_f32(out, {1.25f, -2.5f, 3.75f});
    minllama_test::write_tensor_payload_f16(out, {0x3c00u, 0xc000u, 0x4200u, 0x3800u});
    minllama_test::write_tensor_data_padding(out, 18);
    return static_cast<bool>(out);
}

void assert_close(float actual, float expected) {
    assert(std::fabs(actual - expected) < 0.00001f);
}

} // namespace

int main() {
    const char *path = "test_tensor_read_valid.gguf";
    std::remove(path);

    assert(write_tensor_read_gguf(path));
    ml_model *model = ml_model_load(path);
    assert(model != nullptr);

    std::vector<float> values;
    assert(minllama::load_tensor_f32(path, model->tensor_index, "f32.weight", &values));
    assert(values.size() == 3);
    assert_close(values[0], 1.25f);
    assert_close(values[1], -2.5f);
    assert_close(values[2], 3.75f);

    values.clear();
    assert(minllama::load_tensor_as_f32(path, model->tensor_index, "f16.weight", &values));
    assert(values.size() == 4);
    assert_close(values[0], 1.0f);
    assert_close(values[1], -2.0f);
    assert_close(values[2], 3.0f);
    assert_close(values[3], 0.5f);

    values.clear();
    assert(!minllama::load_tensor_f32(path, model->tensor_index, "f16.weight", &values));
    assert(minllama::load_tensor_as_f32(path, model->tensor_index, "q4_0.weight", &values));
    assert(values.size() == 32);
    for (float value : values) {
        assert_close(value, 0.0f);
    }
    assert(!minllama::load_tensor_as_f32(path, model->tensor_index, "missing.weight", &values));

    {
        std::ofstream truncated(path, std::ios::binary | std::ios::trunc);
        truncated.write("GGUF", 4);
    }
    assert(!minllama::load_tensor_f32(path, model->tensor_index, "f32.weight", &values));

    ml_model_free(model);
    std::remove(path);
    return 0;
}
