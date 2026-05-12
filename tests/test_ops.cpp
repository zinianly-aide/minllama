#include "minllama_internal.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

void assert_close(float actual, float expected, float tolerance = 0.00001f) {
    assert(std::fabs(actual - expected) < tolerance);
}

void assert_vector_close(const std::vector<float> &actual,
                         const std::vector<float> &expected,
                         float tolerance = 0.00001f) {
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert_close(actual[i], expected[i], tolerance);
    }
}

} // namespace

int main() {
    const std::vector<float> x = {1.0f, 2.0f, 3.0f};
    const std::vector<float> weight = {1.0f, 1.5f, -2.0f};
    std::vector<float> out(3, 0.0f);

    assert(minllama::rmsnorm_f32(x.data(), weight.data(), x.size(), 0.0f, out.data(), out.size()));
    const float scale = 1.0f / std::sqrt((1.0f + 4.0f + 9.0f) / 3.0f);
    assert_vector_close(out, {
        1.0f * 1.0f * scale,
        1.5f * 2.0f * scale,
        -2.0f * 3.0f * scale,
    });
    assert(!minllama::rmsnorm_f32(x.data(), weight.data(), x.size(), 0.0f, out.data(), 2));

    const std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    assert(minllama::softmax_f32(logits.data(), logits.size(), out.data(), out.size()));
    const float e0 = std::exp(-2.0f);
    const float e1 = std::exp(-1.0f);
    const float e2 = 1.0f;
    const float sum = e0 + e1 + e2;
    assert_vector_close(out, {e0 / sum, e1 / sum, e2 / sum});
    assert_close(out[0] + out[1] + out[2], 1.0f);

    const std::vector<float> large_logits = {1000.0f, 1000.0f};
    std::vector<float> large_out(2, 0.0f);
    assert(minllama::softmax_f32(large_logits.data(), large_logits.size(), large_out.data(), large_out.size()));
    assert_vector_close(large_out, {0.5f, 0.5f});
    assert(!minllama::softmax_f32(logits.data(), logits.size(), out.data(), 2));

    assert_close(minllama::silu_f32(0.0f), 0.0f);
    assert_close(minllama::silu_f32(2.0f), 2.0f / (1.0f + std::exp(-2.0f)));
    assert_close(minllama::silu_f32(-1.0f), -1.0f / (1.0f + std::exp(1.0f)));

    const std::vector<float> gate = {0.0f, 2.0f, -1.0f};
    const std::vector<float> up = {3.0f, -4.0f, 5.0f};
    assert(minllama::swiglu_f32(gate.data(), up.data(), gate.size(), out.data(), out.size()));
    assert_vector_close(out, {
        minllama::silu_f32(0.0f) * 3.0f,
        minllama::silu_f32(2.0f) * -4.0f,
        minllama::silu_f32(-1.0f) * 5.0f,
    });
    assert(!minllama::swiglu_f32(gate.data(), up.data(), gate.size(), out.data(), 2));

    return 0;
}
