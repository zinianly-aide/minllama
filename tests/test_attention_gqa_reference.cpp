#include "minllama_internal.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

float rel_error(float a, float b) {
    float denom = std::fabs(a) + std::fabs(b);
    if (denom < 1e-9f) return 0.0f;
    return std::fabs(a - b) / denom;
}

// Manual softmax over an in-place array.
void softmax_inplace(float *x, int len) {
    float mx = x[0];
    for (int i = 1; i < len; ++i) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    if (sum > 0.0f) for (int i = 0; i < len; ++i) x[i] /= sum;
}

// Dot product.
float dot(const float *a, const float *b, int len) {
    float s = 0.0f;
    for (int i = 0; i < len; ++i) s += a[i] * b[i];
    return s;
}

} // namespace

int main() {
    // ================================================================
    // Test 1: GQA with dim=6, n_heads=3, n_kv_heads=1, head_dim=2.
    // position=1. q heads have different values. k/v has 2 tokens.
    // Each step is hand-computed and element-wise asserted.
    // ================================================================
    {
        constexpr int max_tokens = 8;
        constexpr int n_heads = 3, n_kv_heads = 1, head_dim = 2;
        constexpr int kv_dim = n_kv_heads * head_dim; // 2
        // Query dimension = n_heads * head_dim = 6

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, max_tokens, n_kv_heads, head_dim));

        // Verify cache layout.
        assert(cache.keys.size() == static_cast<std::size_t>(max_tokens * kv_dim));
        assert(cache.values.size() == static_cast<std::size_t>(max_tokens * kv_dim));
        assert(cache.n_kv_heads == 1);
        assert(cache.head_dim == 2);
        assert(cache.dim == kv_dim);

        // ---- Write token 0 ----
        // k0 = [3.0, 5.0]   (one kv head)
        // v0 = [7.0, 11.0]
        {
            float k0[2] = {3.0f, 5.0f};
            float v0[2] = {7.0f, 11.0f};
            assert(minllama::kv_cache_write_f32(cache, 0, k0, v0));
        }

        // ---- Write token 1 ----
        // k1 = [13.0, 17.0]
        // v1 = [19.0, 23.0]
        {
            float k1[2] = {13.0f, 17.0f};
            float v1[2] = {19.0f, 23.0f};
            assert(minllama::kv_cache_write_f32(cache, 1, k1, v1));
        }

        // ---- Query ----
        // q = [head0, head1, head2] = [1,2, 3,4, 5,6]
        float q[6] = {1.0f, 2.0f,   // head 0
                       3.0f, 4.0f,   // head 1
                       5.0f, 6.0f};  // head 2
        float out[6] = {};
        assert(minllama::attention_decode_gqa_f32(q, cache, 1, n_heads, n_kv_heads, head_dim, out));

        // ---- Hand calculation ----
        // groups_per_head = 3 / 1 = 3
        // All heads share kv_h = 0 (since h / 3 = 0 for h=0,1,2)
        //
        // KV for kv_h=0:
        //   Token 0: k=[3,5], v=[7,11]
        //   Token 1: k=[13,17], v=[19,23]
        //
        // inv_sqrt_head_dim = 1/sqrt(2) ≈ 0.70710678
        const float inv_sqrt_hd = 1.0f / std::sqrt(2.0f);  // ≈0.70710678

        for (int h = 0; h < n_heads; ++h) {
            const float *q_head = q + h * head_dim;

            // ---- Hand-compute scores ----
            // Token 0: score0 = dot(q_head, [3,5]) / sqrt(2)
            // Token 1: score1 = dot(q_head, [13,17]) / sqrt(2)
            float s0 = (q_head[0] * 3.0f + q_head[1] * 5.0f) * inv_sqrt_hd;
            float s1 = (q_head[0] * 13.0f + q_head[1] * 17.0f) * inv_sqrt_hd;

            // ---- Hand-compute softmax ----
            float scores[2] = {s0, s1};
            softmax_inplace(scores, 2);
            const float w0 = scores[0];
            const float w1 = scores[1];

            // ---- Hand-compute output ----
            const float expected0 = w0 * 7.0f + w1 * 19.0f;
            const float expected1 = w0 * 11.0f + w1 * 23.0f;

            printf("Test1 head %d: w0=%.6f w1=%.6f -> expected [%.6f, %.6f] got [%.6f, %.6f]\n",
                   h, (double)w0, (double)w1,
                   (double)expected0, (double)expected1,
                   (double)out[h * head_dim], (double)out[h * head_dim + 1]);

            assert(rel_error(out[h * head_dim], expected0) < 1e-5f);
            assert(rel_error(out[h * head_dim + 1], expected1) < 1e-5f);
        }
    }

    // ================================================================
    // Test 2: GQA with dim=8, n_heads=4, n_kv_heads=2, head_dim=2.
    // head0,head1 -> kv_h=0; head2,head3 -> kv_h=1.
    // Verify kv head offset calculation.
    // ================================================================
    {
        constexpr int max_tokens = 8;
        constexpr int n_heads = 4, n_kv_heads = 2, head_dim = 2;
        constexpr int kv_dim = n_kv_heads * head_dim; // 4

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, max_tokens, n_kv_heads, head_dim));
        assert(cache.dim == kv_dim);
        assert(cache.n_kv_heads == 2);

        // ---- Write token 0 ----
        // kv_h=0: k=[1,0], v=[100,0]
        // kv_h=1: k=[0,1], v=[0,200]
        float k0[4] = {1.0f, 0.0f,  // kv head 0
                        0.0f, 1.0f}; // kv head 1
        float v0[4] = {100.0f,   0.0f,  // kv head 0
                         0.0f, 200.0f}; // kv head 1
        assert(minllama::kv_cache_write_f32(cache, 0, k0, v0));

        // Query: head0=[1,0], head1=[1,0] -> kv_h=0; head2=[0,1], head3=[0,1] -> kv_h=1
        float q[8] = {1.0f, 0.0f,   // head 0 -> kv_h=0
                       1.0f, 0.0f,   // head 1 -> kv_h=0
                       0.0f, 1.0f,   // head 2 -> kv_h=1
                       0.0f, 1.0f};  // head 3 -> kv_h=1
        float out[8] = {};
        assert(minllama::attention_decode_gqa_f32(q, cache, 0, n_heads, n_kv_heads, head_dim, out));

        // With only 1 token, softmax always = 1 regardless of score.
        // head0: q=[1,0], k_kv0=[1,0] -> v=[100,0] -> out=[100,0]
        // head1: q=[1,0], k_kv0=[1,0] -> v=[100,0] -> out=[100,0]
        // head2: q=[0,1], k_kv1=[0,1] -> v=[0,200] -> out=[0,200]
        // head3: q=[0,1], k_kv1=[0,1] -> v=[0,200] -> out=[0,200]
        printf("Test2: head0=[%.1f,%.1f] head1=[%.1f,%.1f] head2=[%.1f,%.1f] head3=[%.1f,%.1f]\n",
               (double)out[0], (double)out[1],
               (double)out[2], (double)out[3],
               (double)out[4], (double)out[5],
               (double)out[6], (double)out[7]);

        assert(rel_error(out[0], 100.0f) < 0.02f);
        assert(rel_error(out[1],   0.0f) < 0.02f);
        assert(rel_error(out[2], 100.0f) < 0.02f);
        assert(rel_error(out[3],   0.0f) < 0.02f);
        assert(rel_error(out[4],   0.0f) < 0.02f);
        assert(rel_error(out[5], 200.0f) < 0.02f);
        assert(rel_error(out[6],   0.0f) < 0.02f);
        assert(rel_error(out[7], 200.0f) < 0.02f);
    }

    // ================================================================
    // Test 3: GQA multi-token with distinct values.
    // dim=4, n_heads=2, n_kv_heads=1, head_dim=2.
    // Two tokens, different keys → attention weights split.
    // ================================================================
    {
        constexpr int max_tokens = 8;
        constexpr int n_heads = 2, n_kv_heads = 1, head_dim = 2;

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, max_tokens, n_kv_heads, head_dim));

        // Token 0: k=[1,0], v=[10, 0]
        // Token 1: k=[0,1], v=[ 0,20]
        {
            float k0[2] = {1.0f, 0.0f};
            float v0[2] = {10.0f, 0.0f};
            assert(minllama::kv_cache_write_f32(cache, 0, k0, v0));
        }
        {
            float k1[2] = {0.0f, 1.0f};
            float v1[2] = {0.0f, 20.0f};
            assert(minllama::kv_cache_write_f32(cache, 1, k1, v1));
        }

        // Query: head0=[1,0] (strong match with token0), head1=[0,1] (strong match with token1)
        float q[4] = {1.0f, 0.0f,   // head 0
                       0.0f, 1.0f};  // head 1
        float out[4] = {};
        assert(minllama::attention_decode_gqa_f32(q, cache, 1, n_heads, n_kv_heads, head_dim, out));

        const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);

        // head0: q=[1,0]
        //   s0 = dot([1,0], [1,0])/√2 = 1/√2
        //   s1 = dot([1,0], [0,1])/√2 = 0
        float s0_h0 = 1.0f * inv_sqrt2;
        float s1_h0 = 0.0f;
        float scores_h0[2] = {s0_h0, s1_h0};
        softmax_inplace(scores_h0, 2);
        // w0 ≈ e^0.707 / (e^0.707 + 1), w1 ≈ 1 / (e^0.707 + 1)
        printf("Test3 head0: scores=[%.6f, %.6f] softmax=[%.6f, %.6f]\n",
               (double)s0_h0, (double)s1_h0,
               (double)scores_h0[0], (double)scores_h0[1]);

        // head0 out = w0*[10,0] + w1*[0,20]
        float expected_h0_0 = scores_h0[0] * 10.0f;
        float expected_h0_1 = scores_h0[1] * 20.0f;

        assert(rel_error(out[0], expected_h0_0) < 1e-5f);
        assert(rel_error(out[1], expected_h0_1) < 1e-5f);

        // head1: q=[0,1]
        //   s0 = dot([0,1], [1,0])/√2 = 0
        //   s1 = dot([0,1], [0,1])/√2 = 1/√2
        float s0_h1 = 0.0f;
        float s1_h1 = 1.0f * inv_sqrt2;
        float scores_h1[2] = {s0_h1, s1_h1};
        softmax_inplace(scores_h1, 2);
        // w0 ≈ 1/(1+e^0.707), w1 ≈ e^0.707/(1+e^0.707) — same weights swapped
        float expected_h1_0 = scores_h1[0] * 10.0f;
        float expected_h1_1 = scores_h1[1] * 20.0f;

        printf("Test3 head1: out=[%.6f, %.6f]\n",
               (double)out[2], (double)out[3]);

        assert(rel_error(out[2], expected_h1_0) < 1e-5f);
        assert(rel_error(out[3], expected_h1_1) < 1e-5f);

        // Output should be ordered as [head0, head1], not interleaved.
        // head0 outputs go to out[0:2], head1 outputs go to out[2:4].
    }

    // ================================================================
    // Test 4: RoPE per-head — confirm each head gets independent RoPE.
    // Simulate what transformer_layer_decode_f32 does for GQA.
    // ================================================================
    {
        constexpr int n_heads = 2, head_dim = 4;
        float q[8] = {1.0f, 2.0f, 3.0f, 4.0f,    // head 0
                       5.0f, 6.0f, 7.0f, 8.0f};   // head 1

        // Apply RoPE position=0 (should be identity).
        for (int h = 0; h < n_heads; ++h) {
            assert(minllama::rope_apply_f32(q + h * head_dim,
                                             static_cast<std::size_t>(head_dim),
                                             static_cast<std::size_t>(0),
                                             10000.0f));
        }

        // Position 0 → cos(0)=1, sin(0)=0 → values unchanged.
        printf("Test4 pos=0: head0=[%.1f,%.1f,%.1f,%.1f] head1=[%.1f,%.1f,%.1f,%.1f]\n",
               (double)q[0], (double)q[1], (double)q[2], (double)q[3],
               (double)q[4], (double)q[5], (double)q[6], (double)q[7]);
        assert(rel_error(q[0], 1.0f) < 0.01f);
        assert(rel_error(q[4], 5.0f) < 0.01f);

        // Apply RoPE position=1.
        // freq_i = theta^(-2*i/len) where len = head_dim (4)
        // half_len = 2
        // For i=0: freq = 10000^(-0/2) = 1.0    → angle = 1*1 = 1
        // For i=1: freq = 10000^(-1/2) = 0.01   → angle = 1*0.01 = 0.01
        // Row 0: [1,2] rotated by angle=1:
        //   cos(1)=0.5403, sin(1)=0.8415
        //   new0 = 1*0.5403 - 2*0.8415 = -1.1427
        //   new1 = 1*0.8415 + 2*0.5403 = 1.9221
        // Row 2: [3,4] rotated by angle=0.01:
        //   cos(0.01)=0.99995, sin(0.01)=0.0099998
        //   new2 = 3*0.99995 - 4*0.0099998 = 2.95986
        //   new3 = 3*0.0099998 + 4*0.99995 = 4.0298
        for (int h = 0; h < n_heads; ++h) {
            assert(minllama::rope_apply_f32(q + h * head_dim,
                                             static_cast<std::size_t>(head_dim),
                                             static_cast<std::size_t>(1),
                                             10000.0f));
        }

        float c1 = std::cos(1.0f), s1 = std::sin(1.0f);
        float c01 = std::cos(0.01f), s01 = std::sin(0.01f);

        // head0 first pair: [1,2] rotated by angle 1
        float e_h0_0 = 1.0f * c1 - 2.0f * s1;
        float e_h0_1 = 1.0f * s1 + 2.0f * c1;
        // head0 second pair: [3,4] rotated by angle 0.01
        float e_h0_2 = 3.0f * c01 - 4.0f * s01;
        float e_h0_3 = 3.0f * s01 + 4.0f * c01;

        printf("Test4 pos=1 head0: got=[%.4f,%.4f,%.4f,%.4f] expected=[%.4f,%.4f,%.4f,%.4f]\n",
               (double)q[0], (double)q[1], (double)q[2], (double)q[3],
               (double)e_h0_0, (double)e_h0_1, (double)e_h0_2, (double)e_h0_3);

        assert(rel_error(q[0], e_h0_0) < 1e-4f);
        assert(rel_error(q[1], e_h0_1) < 1e-4f);
        assert(rel_error(q[2], e_h0_2) < 1e-4f);
        assert(rel_error(q[3], e_h0_3) < 1e-4f);

        // head1: [5,6,7,8] also rotated by same angles
        float e_h1_0 = 5.0f * c1 - 6.0f * s1;
        float e_h1_1 = 5.0f * s1 + 6.0f * c1;
        float e_h1_2 = 7.0f * c01 - 8.0f * s01;
        float e_h1_3 = 7.0f * s01 + 8.0f * c01;

        printf("Test4 pos=1 head1: got=[%.4f,%.4f,%.4f,%.4f] expected=[%.4f,%.4f,%.4f,%.4f]\n",
               (double)q[4], (double)q[5], (double)q[6], (double)q[7],
               (double)e_h1_0, (double)e_h1_1, (double)e_h1_2, (double)e_h1_3);

        assert(rel_error(q[4], e_h1_0) < 1e-4f);
        assert(rel_error(q[5], e_h1_1) < 1e-4f);
        assert(rel_error(q[6], e_h1_2) < 1e-4f);
        assert(rel_error(q[7], e_h1_3) < 1e-4f);
    }

    // ================================================================
    // Test 5: RMSNorm eps — verify the function passes eps through.
    // Use a synthetic input and test with GGUF model's eps=1e-5.
    // ================================================================
    {
        float x[4] = {2.0f, 2.0f, 2.0f, 2.0f};
        float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float out[4] = {};

        // With eps=1e-5:
        // mean_sq = (4+4+4+4)/4 = 4
        // scale = 1/sqrt(4+1e-5) ≈ 1/2.0000025 ≈ 0.4999994
        // out[i] = 1.0 * 2.0 * 0.4999994 ≈ 0.9999988
        const float eps = 1e-5f;
        assert(minllama::rmsnorm_f32(x, w, 4, eps, out, 4));

        const float expected_scale = 1.0f / std::sqrt(4.0f + eps);
        const float expected = 2.0f * expected_scale;
        printf("Test5 RMSNorm eps=1e-5: out=[%.10f, ...] expected=%.10f scale=%.10f\n",
               (double)out[0], (double)expected, (double)expected_scale);

        assert(rel_error(out[0], expected) < 1e-7f);
        assert(rel_error(out[1], expected) < 1e-7f);
        assert(rel_error(out[2], expected) < 1e-7f);
        assert(rel_error(out[3], expected) < 1e-7f);
    }

    printf("All GQA reference tests passed.\n");
    return 0;
}
