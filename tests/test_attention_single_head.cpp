#include "minllama_internal.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

bool close(float a, float b, float tolerance = 0.001f) {
    return std::fabs(a - b) < tolerance;
}

} // namespace

int main() {
    // -----------------------------------------------------------------
    // Test 1: Single token attention.
    // With one key/value pair, softmax over a single score is always 1,
    // so output must equal the single value.
    // -----------------------------------------------------------------
    {
        constexpr int n_tokens = 1;
        constexpr int dim = 2;
        const float query[] = {0.5f, -3.0f};
        const float keys[] = {1.0f, 2.0f};
        const float values[] = {7.0f, 8.0f};
        float output[2] = {};

        assert(minllama::attention_single_head_f32(
            query, keys, values, n_tokens, dim, output));

        assert(close(output[0], values[0]));
        assert(close(output[1], values[1]));
    }

    // -----------------------------------------------------------------
    // Test 2: Two token attention with known analytical result.
    // query = [1, 0]
    // key0  = [1, 0]  value0 = [10, 0]
    // key1  = [0, 1]  value1 = [0, 20]
    // -----------------------------------------------------------------
    {
        constexpr int n_tokens = 2;
        constexpr int dim = 2;
        const float query[] = {1.0f, 0.0f};
        const float keys[] = {
            1.0f, 0.0f,
            0.0f, 1.0f,
        };
        const float values[] = {
            10.0f, 0.0f,
            0.0f, 20.0f,
        };
        float output[2] = {};

        assert(minllama::attention_single_head_f32(
            query, keys, values, n_tokens, dim, output));

        // Manual computation:
        // score0 = dot([1,0],[1,0]) / sqrt(2) = 1 / sqrt(2)
        // score1 = dot([1,0],[0,1]) / sqrt(2) = 0
        // exp(1/sqrt(2)) ≈ 2.028115, exp(0) = 1
        // sum ≈ 3.028115
        // w0 ≈ 0.66976, w1 ≈ 0.33024
        // out[0] = w0*10 ≈ 6.6976
        // out[1] = w1*20 ≈ 6.6048
        const float expected_0 = 6.6976f;
        const float expected_1 = 6.6048f;
        assert(close(output[0], expected_0));
        assert(close(output[1], expected_1));
    }

    // -----------------------------------------------------------------
    // Test 3: Multi-token, dim=4.  Sanity-check: no NaN and equals
    // softmax-weighted sum computed by hand.
    // -----------------------------------------------------------------
    {
        constexpr int n_tokens = 3;
        constexpr int dim = 4;
        const float query[] = {1.0f, 1.0f, 1.0f, 1.0f};
        const float keys[] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
        };
        const float values[] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 3.0f, 0.0f,
        };
        float output[4] = {};

        assert(minllama::attention_single_head_f32(
            query, keys, values, n_tokens, dim, output));

        // All three scores are identical (1/sqrt(4) = 0.5), so softmax
        // gives 1/3 for each.
        // output = (1/3)*v0 + (1/3)*v1 + (1/3)*v2
        //        = [1/3, 2/3, 1, 0]
        const float expected[] = {1.0f / 3.0f, 2.0f / 3.0f, 1.0f, 0.0f};

        for (int j = 0; j < dim; ++j) {
            // Not NaN
            assert(!std::isnan(output[j]));
            assert(close(output[j], expected[j]));
        }
    }

    // -----------------------------------------------------------------
    // Test 4: Invalid inputs.
    // -----------------------------------------------------------------
    {
        const float q[] = {1.0f};
        const float k[] = {1.0f};
        const float v[] = {1.0f};
        float out[1] = {};

        // nullptr query
        assert(!minllama::attention_single_head_f32(
            nullptr, k, v, 1, 1, out));
        // nullptr keys
        assert(!minllama::attention_single_head_f32(
            q, nullptr, v, 1, 1, out));
        // nullptr values
        assert(!minllama::attention_single_head_f32(
            q, k, nullptr, 1, 1, out));
        // nullptr output
        assert(!minllama::attention_single_head_f32(
            q, k, v, 1, 1, nullptr));
        // n_tokens = 0
        assert(!minllama::attention_single_head_f32(
            q, k, v, 0, 1, out));
        // dim = 0
        assert(!minllama::attention_single_head_f32(
            q, k, v, 1, 0, out));
        // n_tokens < 0
        assert(!minllama::attention_single_head_f32(
            q, k, v, -1, 1, out));
        // dim < 0
        assert(!minllama::attention_single_head_f32(
            q, k, v, 1, -1, out));
    }

    return 0;
}
