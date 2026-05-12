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
    // ----------------------------------------------------------------
    // Test 1: kv_cache_init_f32 successful init.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, 2));
        assert(cache.max_tokens == 4);
        assert(cache.dim == 2);
        assert(cache.keys.size() == 8u);
        assert(cache.values.size() == 8u);
    }

    // ----------------------------------------------------------------
    // Test 2: kv_cache_write_f32 writes to correct position.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, 2));

        const float key[] = {3.0f, 4.0f};
        const float value[] = {5.0f, 6.0f};
        assert(minllama::kv_cache_write_f32(cache, 1, key, value));

        assert(close(cache.keys[1 * 2 + 0], 3.0f));
        assert(close(cache.keys[1 * 2 + 1], 4.0f));
        assert(close(cache.values[1 * 2 + 0], 5.0f));
        assert(close(cache.values[1 * 2 + 1], 6.0f));

        // Position 0 should still be zeroed.
        assert(close(cache.keys[0 * 2 + 0], 0.0f));
        assert(close(cache.keys[0 * 2 + 1], 0.0f));
    }

    // ----------------------------------------------------------------
    // Test 3: Decode single token (position=0). Output == value0.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, 2));

        const float key0[] = {1.0f, 2.0f};
        const float value0[] = {7.0f, 8.0f};
        assert(minllama::kv_cache_write_f32(cache, 0, key0, value0));

        const float query[] = {0.0f, 1.0f}; // arbitrary
        float output[2] = {};
        assert(minllama::attention_decode_single_head_f32(
            query, cache, 0, output));

        // Softmax over single score always 1 → output == value0.
        assert(close(output[0], 7.0f));
        assert(close(output[1], 8.0f));
    }

    // ----------------------------------------------------------------
    // Test 4: Decode multi-token (2 tokens). Matches attention_single_head_f32.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, 2));

        // Write two tokens.
        const float key0[] = {1.0f, 0.0f};
        const float val0[] = {10.0f, 0.0f};
        const float key1[] = {0.0f, 1.0f};
        const float val1[] = {0.0f, 20.0f};
        assert(minllama::kv_cache_write_f32(cache, 0, key0, val0));
        assert(minllama::kv_cache_write_f32(cache, 1, key1, val1));

        // Decode at position=1 with query=[1,0].
        const float query[] = {1.0f, 0.0f};
        float output[2] = {};
        assert(minllama::attention_decode_single_head_f32(
            query, cache, 1, output));

        // Same analytical result as step 11 test 2:
        // score0 = 1/sqrt(2), score1 = 0
        // w0 ≈ 0.66976, w1 ≈ 0.33024
        // out[0] ≈ 6.6976, out[1] ≈ 6.6048
        assert(close(output[0], 6.6976f));
        assert(close(output[1], 6.6048f));
    }

    // ----------------------------------------------------------------
    // Test 5: Invalid inputs.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;

        // init: max_tokens <= 0
        assert(!minllama::kv_cache_init_f32(cache, 0, 1));
        assert(!minllama::kv_cache_init_f32(cache, -1, 1));
        // init: dim <= 0
        assert(!minllama::kv_cache_init_f32(cache, 1, 0));
        assert(!minllama::kv_cache_init_f32(cache, 1, -1));

        // Valid cache for subsequent tests.
        assert(minllama::kv_cache_init_f32(cache, 4, 2));
        const float k[] = {1.0f, 2.0f};
        const float v[] = {3.0f, 4.0f};
        const float q[] = {0.0f, 0.0f};
        float out[2] = {};

        // write: nullptr key/value
        assert(!minllama::kv_cache_write_f32(cache, 0, nullptr, v));
        assert(!minllama::kv_cache_write_f32(cache, 0, k, nullptr));
        // write: position < 0
        assert(!minllama::kv_cache_write_f32(cache, -1, k, v));
        // write: position >= max_tokens
        assert(!minllama::kv_cache_write_f32(cache, 4, k, v));

        // decode: nullptr query
        assert(!minllama::attention_decode_single_head_f32(
            nullptr, cache, 0, out));
        // decode: nullptr output
        assert(!minllama::attention_decode_single_head_f32(
            q, cache, 0, nullptr));
        // decode: position < 0
        assert(!minllama::attention_decode_single_head_f32(
            q, cache, -1, out));
        // decode: position >= max_tokens
        assert(!minllama::attention_decode_single_head_f32(
            q, cache, 4, out));
    }

    return 0;
}
