#include "minllama_internal.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

void assert_close(float actual, float expected, float tolerance = 0.00001f) {
    assert(std::fabs(actual - expected) < tolerance);
}

} // namespace

int main() {
    {
        std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> original = v;
        assert(minllama::rope_apply_f32(v.data(), v.size(), 0, 10000.0f));
        for (std::size_t i = 0; i < v.size(); ++i) {
            assert_close(v[i], original[i]);
        }
    }

    {
        std::vector<float> v = {1.0f, 0.0f, 0.0f, 1.0f};
        assert(minllama::rope_apply_f32(v.data(), v.size(), 1, 10000.0f));
        assert_close(v[0], std::cos(1.0f));
        assert_close(v[1], std::sin(1.0f));
        const float freq = std::pow(10000.0f, -0.5f);
        assert_close(v[2], -std::sin(freq));
        assert_close(v[3], std::cos(freq));
    }

    {
        std::vector<float> odd = {1.0f, 2.0f, 3.0f};
        assert(!minllama::rope_apply_f32(odd.data(), odd.size(), 1, 10000.0f));
        assert(!minllama::rope_apply_f32(nullptr, 4, 1, 10000.0f));
        assert(!minllama::rope_apply_f32(odd.data(), 0, 1, 10000.0f));
    }

    return 0;
}
