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
    float dot = 0.0f;
    const float a[] = {1.0f, 2.0f, 3.0f};
    const float b[] = {4.0f, -1.0f, 0.5f};
    assert(minllama::dot_f32(a, b, 3, &dot));
    assert_close(dot, 3.5f);
    assert(!minllama::dot_f32(nullptr, b, 3, &dot));

    float scaled[] = {1.0f, -2.0f, 3.0f};
    assert(minllama::scale_f32(scaled, 3, 0.5f));
    assert_close(scaled[0], 0.5f);
    assert_close(scaled[1], -1.0f);
    assert_close(scaled[2], 1.5f);
    assert(!minllama::scale_f32(nullptr, 3, 1.0f));

    const float query[] = {1.0f, 2.0f};
    const float keys[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };
    float scores[3] = {};
    assert(minllama::attention_scores_f32(query, 2, keys, 3, scores, 3));
    const float inv_sqrt_2 = 1.0f / std::sqrt(2.0f);
    assert_close(scores[0], 1.0f * inv_sqrt_2);
    assert_close(scores[1], 2.0f * inv_sqrt_2);
    assert_close(scores[2], 3.0f * inv_sqrt_2);
    assert(!minllama::attention_scores_f32(query, 2, keys, 3, scores, 2));

    return 0;
}
