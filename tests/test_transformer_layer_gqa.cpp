#include "minllama_internal.h"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    // ----------------------------------------------------------------
    // Test 1: dim=4, n_heads=2, n_kv_heads=1, head_dim=2.
    // Full layer decode with GQA.
    // ----------------------------------------------------------------
    {
        constexpr int dim = 4, n_heads = 2, n_kv_heads = 1, head_dim = 2;
        constexpr int kv_dim = n_kv_heads * head_dim; // 2

        minllama::TransformerLayerF32 layer;
        layer.dim = dim;
        layer.n_heads = n_heads;
        layer.n_kv_heads = n_kv_heads;
        layer.head_dim = head_dim;
        layer.rms_att_weight.assign(dim, 1.0f);
        layer.rms_norm_eps = 1e-6f;
        layer.rope_theta = 10000.0f;
        layer.hidden_dim = 0; // attention-only

        // wq: [dim, dim] = [4, 4] — identity
        layer.wq.assign(dim * dim, 0.0f);
        for (int i = 0; i < dim; ++i) layer.wq[i * dim + i] = 1.0f;

        // wk: [kv_dim, dim] = [2, 4]
        layer.wk.assign(kv_dim * dim, 0.0f);
        layer.wk[0 * dim + 0] = 1.0f; // head0 row0
        layer.wk[0 * dim + 1] = 0.0f;
        layer.wk[1 * dim + 0] = 0.0f; // head0 row1
        layer.wk[1 * dim + 1] = 1.0f;

        // wv: [kv_dim, dim] = [2, 4]
        layer.wv.assign(kv_dim * dim, 0.0f);
        layer.wv[0 * dim + 0] = 1.0f;
        layer.wv[0 * dim + 1] = 0.0f;
        layer.wv[1 * dim + 0] = 0.0f;
        layer.wv[1 * dim + 1] = 1.0f;

        // wo: [dim, dim] = [4, 4] — identity
        layer.wo.assign(dim * dim, 0.0f);
        for (int i = 0; i < dim; ++i) layer.wo[i * dim + i] = 1.0f;

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, 8, n_kv_heads, head_dim));

        // Cache size = max_tokens * kv_dim = 8 * 2 = 16
        assert(cache.keys.size() == 16);
        assert(cache.values.size() == 16);
        assert(cache.n_kv_heads == 1);
        assert(cache.head_dim == 2);
        assert(cache.dim == 2);

        // Input x = [1, 0, 0, 0]
        float x[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float output[4] = {};

        assert(minllama::transformer_layer_decode_f32(layer, x, cache, 0, output));

        // Verify output is finite (not NaN).
        for (int i = 0; i < dim; ++i) {
            assert(!std::isnan(output[i]));
            assert(std::isfinite(output[i]));
        }
    }

    // ----------------------------------------------------------------
    // Test 2: GQA with FFN.
    // dim=4, n_heads=2, n_kv_heads=1, hidden_dim=4.
    // ----------------------------------------------------------------
    {
        constexpr int dim = 4, n_heads = 2, n_kv_heads = 1, head_dim = 2;
        constexpr int kv_dim = n_kv_heads * head_dim;

        minllama::TransformerLayerF32 layer;
        layer.dim = dim;
        layer.n_heads = n_heads;
        layer.n_kv_heads = n_kv_heads;
        layer.head_dim = head_dim;
        layer.rms_att_weight.assign(dim, 1.0f);
        layer.rms_norm_eps = 1e-6f;
        layer.rope_theta = 10000.0f;
        layer.hidden_dim = 4;

        // Identity attention weights.
        layer.wq.assign(dim * dim, 0.0f);
        for (int i = 0; i < dim; ++i) layer.wq[i * dim + i] = 1.0f;
        layer.wk.assign(kv_dim * dim, 0.0f);
        for (int i = 0; i < kv_dim; ++i) layer.wk[i * dim + i] = 0.1f; // weak K
        layer.wv.assign(kv_dim * dim, 0.0f);
        for (int i = 0; i < kv_dim; ++i) layer.wv[i * dim + i] = 1.0f;
        layer.wo.assign(dim * dim, 0.0f);
        for (int i = 0; i < dim; ++i) layer.wo[i * dim + i] = 1.0f;

        // FFN: gate+up zero, down identity (output ~= residual)
        layer.rms_ffn_weight.assign(dim, 1.0f);
        layer.w1.assign(4 * dim, 0.0f);  // gate all zero → SiLU output small
        layer.w3.assign(4 * dim, 0.0f);  // up all zero
        layer.w2.assign(dim * 4, 0.0f);  // down all zero
        // SiLU(0)*0 = 0 → ffn_out = 0 → output = h + 0 = h

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, 8, n_kv_heads, head_dim));

        float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float output[4] = {};

        assert(minllama::transformer_layer_decode_f32(layer, x, cache, 0, output));

        // All finite.
        for (int i = 0; i < dim; ++i) {
            assert(std::isfinite(output[i]));
        }
    }

    // ----------------------------------------------------------------
    // Test 3: GQA invalid — n_heads % n_kv_heads != 0.
    // ----------------------------------------------------------------
    {
        minllama::TransformerLayerF32 layer;
        layer.dim = 6;
        layer.n_heads = 3;
        layer.n_kv_heads = 2; // 3 % 2 != 0
        layer.head_dim = 2;
        layer.rms_att_weight.assign(6, 1.0f);
        layer.rms_norm_eps = 1e-6f;

        // Weights: identity to pass validation.
        layer.wq.assign(36, 0.0f);
        for (int i = 0; i < 6; ++i) layer.wq[i * 6 + i] = 1.0f;
        layer.wk.assign(4 * 6, 0.0f); // kv_dim=4
        layer.wv.assign(4 * 6, 0.0f);
        layer.wo.assign(36, 0.0f);
        for (int i = 0; i < 6; ++i) layer.wo[i * 6 + i] = 1.0f;

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_gqa_f32(cache, 8, 2, 2));

        float x[6] = {};
        float out[6] = {};
        assert(!minllama::transformer_layer_decode_f32(layer, x, cache, 0, out));
    }

    return 0;
}
