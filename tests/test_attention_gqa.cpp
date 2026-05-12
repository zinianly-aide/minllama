#include "minllama_internal.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

float rel_error(float a, float b) {
    float denom = std::fabs(a) + std::fabs(b);
    if (denom < 1e-9f) return 0.0f;
    return std::fabs(a - b) / denom;
}

} // namespace

int main() {
    // ----------------------------------------------------------------
    // Test 1: MHA — dim=4, n_heads=2, n_kv_heads=2, head_dim=2.
    // Write 1 token. Each head should output its own value.
    // ----------------------------------------------------------------
    {
        constexpr int max_tokens = 4;
        constexpr int n_heads = 2, n_kv_heads = 2, head_dim = 2;

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, max_tokens, n_kv_heads, head_dim));

        // Write one token: k=[1,0, 0,1], v=[2,3, 4,5]
        float k[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float v[4] = {2.0f, 3.0f, 4.0f, 5.0f};
        assert(minllama::kv_cache_write_f32(cache, 0, k, v));

        // Query: q=[1,0, 0,1] (head0=[1,0], head1=[0,1])
        float q[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float out[4] = {};
        assert(minllama::attention_decode_gqa_f32(q, cache, 0, n_heads, n_kv_heads, head_dim, out));

        // head0: q=[1,0], k_h0=[1,0] → dot=1, 1/sqrt(2) → softmax=1 → v=[2,3]
        // head1: q=[0,1], k_h1=[0,1] → dot=1, 1/sqrt(2) → softmax=1 → v=[4,5]
        // Hmm, that's not right. Let me reconsider.
        // k = [1,0, 0,1] = head0 key [1,0], head1 key [0,1]
        // q = [1,0, 0,1] = head0 query [1,0], head1 query [0,1]
        // 
        // For head0: q0=[1,0], k0=[1,0] → score = dot/√2 = 1/√2, exp → softmax=1 → v0=[2,3]
        // For head1: q1=[0,1], k1=[0,1] → score = dot/√2 = 1/√2, exp → softmax=1 → v1=[4,5]
        // Output: [2,3, 4,5]
        assert(rel_error(out[0], 2.0f) < 0.01f);
        assert(rel_error(out[1], 3.0f) < 0.01f);
        assert(rel_error(out[2], 4.0f) < 0.01f);
        assert(rel_error(out[3], 5.0f) < 0.01f);
    }

    // ----------------------------------------------------------------
    // Test 2: GQA — dim=4, n_heads=2, n_kv_heads=1, head_dim=2.
    // Both query heads share the same KV head.
    // ----------------------------------------------------------------
    {
        constexpr int max_tokens = 4;
        constexpr int n_heads = 2, n_kv_heads = 1, head_dim = 2;

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, max_tokens, n_kv_heads, head_dim));

        // Write one token: k=[1,0], v=[2,3]
        float k[2] = {1.0f, 0.0f};
        float v[2] = {2.0f, 3.0f};
        assert(minllama::kv_cache_write_f32(cache, 0, k, v));

        // Query: q=[1,0, 0,1] (head0=[1,0], head1=[0,1])
        float q[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float out[4] = {};
        assert(minllama::attention_decode_gqa_f32(q, cache, 0, n_heads, n_kv_heads, head_dim, out));

        // Both heads share kv_h=0: k=[1,0], v=[2,3]
        // head0: q0=[1,0], k=[1,0] → dot=1/√2, softmax=1 → v=[2,3]
        // head1: q1=[0,1], k=[1,0] → dot=0, softmax=? → v=[2,3]*0.5 + [2,3]*0.5... wait
        // 
        // Actually since there's only 1 token, softmax always gives 1 regardless of score.
        // head1: q1=[0,1], k=[1,0] → dot=0, 0/√2=0, only 1 token → softmax=1 → v=[2,3]
        // Both heads output same: [2,3, 2,3]
        assert(rel_error(out[0], 2.0f) < 0.02f);
        assert(rel_error(out[1], 3.0f) < 0.02f);
        assert(rel_error(out[2], 2.0f) < 0.02f);
        assert(rel_error(out[3], 3.0f) < 0.02f);
    }

    // ----------------------------------------------------------------
    // Test 3: GQA multi-token — position=1.
    // Verify attention distributes across two tokens.
    // ----------------------------------------------------------------
    {
        constexpr int max_tokens = 4;
        constexpr int n_heads = 1, n_kv_heads = 1, head_dim = 2;

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, max_tokens, n_kv_heads, head_dim));

        // Write two tokens.
        float k0[2] = {1.0f, 0.0f};
        float v0[2] = {0.0f, 0.0f};
        assert(minllama::kv_cache_write_f32(cache, 0, k0, v0));

        float k1[2] = {1.0f, 0.0f};
        float v1[2] = {5.0f, 0.0f};
        assert(minllama::kv_cache_write_f32(cache, 1, k1, v1));

        // Query same as keys.
        float q[2] = {1.0f, 0.0f};
        float out[2] = {};
        assert(minllama::attention_decode_gqa_f32(q, cache, 1, n_heads, n_kv_heads, head_dim, out));

        // Both tokens have same key → equal scores → 0.5 weight each
        // out = 0.5*[0,0] + 0.5*[5,0] = [2.5, 0]
        assert(rel_error(out[0], 2.5f) < 0.01f);
        assert(rel_error(out[1], 0.0f) < 0.01f);
    }

    // ----------------------------------------------------------------
    // Test 4: invalid — n_heads <= 0.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, 4, 1, 2));
        float q[2] = {}, out[2] = {};
        assert(!minllama::attention_decode_gqa_f32(q, cache, 0, 0, 1, 2, out));
    }

    // ----------------------------------------------------------------
    // Test 5: invalid — n_kv_heads <= 0.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, 4, 1, 2));
        float q[2] = {}, out[2] = {};
        assert(!minllama::attention_decode_gqa_f32(q, cache, 0, 1, 0, 2, out));
    }

    // ----------------------------------------------------------------
    // Test 6: invalid — n_heads % n_kv_heads != 0.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, 4, 1, 2));
        float q[6] = {}, out[6] = {};
        assert(!minllama::attention_decode_gqa_f32(q, cache, 0, 3, 2, 2, out));
    }

    // ----------------------------------------------------------------
    // Test 7: invalid — position out of range.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, 4, 1, 2));
        float q[2] = {}, out[2] = {};
        assert(!minllama::attention_decode_gqa_f32(q, cache, 4, 1, 1, 2, out));
        assert(!minllama::attention_decode_gqa_f32(q, cache, -1, 1, 1, 2, out));
    }

    // ----------------------------------------------------------------
    // Test 8: invalid — nullptr.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, 4, 1, 2));
        float q[2] = {}, out[2] = {};
        assert(!minllama::attention_decode_gqa_f32(nullptr, cache, 0, 1, 1, 2, out));
        assert(!minllama::attention_decode_gqa_f32(q, cache, 0, 1, 1, 2, nullptr));
    }

    // ----------------------------------------------------------------
    // Test 9: kv_cache_init_gqa_f32 — invalid params.
    // ----------------------------------------------------------------
    {
        minllama::KvCacheF32 cache;
        assert(!minllama::kv_cache_init_gqa_f32(cache, -1, 1, 2));
        assert(!minllama::kv_cache_init_gqa_f32(cache, 4, 0, 2));
        assert(!minllama::kv_cache_init_gqa_f32(cache, 4, 1, 0));
    }

    return 0;
}
