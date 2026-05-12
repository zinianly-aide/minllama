#include "minllama_internal.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

bool close(float a, float b, float tolerance = 0.001f) {
    return std::fabs(a - b) < tolerance;
}

minllama::TransformerLayerF32 make_att_identity_layer(int dim) {
    minllama::TransformerLayerF32 layer;
    layer.dim = dim;
    layer.rms_att_weight.assign(dim, 1.0f);
    layer.rms_norm_eps = 1e-6f;
    layer.rope_theta = 10000.0f;

    layer.wq.assign(dim * dim, 0.0f);
    layer.wk.assign(dim * dim, 0.0f);
    layer.wv.assign(dim * dim, 0.0f);
    layer.wo.assign(dim * dim, 0.0f);
    for (int i = 0; i < dim; ++i) {
        layer.wq[i * dim + i] = 1.0f;
        layer.wk[i * dim + i] = 1.0f;
        layer.wv[i * dim + i] = 1.0f;
        layer.wo[i * dim + i] = 1.0f;
    }
    return layer;
}

minllama::TransformerLayerF32 make_ffn_identity_layer(int dim, int hidden_dim) {
    auto layer = make_att_identity_layer(dim);
    layer.hidden_dim = hidden_dim;
    layer.rms_ffn_weight.assign(dim, 1.0f);

    // w1, w3: [hidden_dim * dim] row-major identity-like
    layer.w1.assign(hidden_dim * dim, 0.0f);
    layer.w3.assign(hidden_dim * dim, 0.0f);
    for (int i = 0; i < hidden_dim && i < dim; ++i) {
        layer.w1[i * dim + i] = 1.0f;
        layer.w3[i * dim + i] = 1.0f;
    }

    // w2: [dim * hidden_dim] row-major identity-like
    layer.w2.assign(dim * hidden_dim, 0.0f);
    for (int i = 0; i < dim && i < hidden_dim; ++i) {
        layer.w2[i * hidden_dim + i] = 1.0f;
    }
    return layer;
}

} // namespace

int main() {
    constexpr int dim = 2;

    // ----------------------------------------------------------------
    // Test 1: FFN disabled — degrades to attention-only (step 13 compat).
    // ----------------------------------------------------------------
    {
        auto layer = make_att_identity_layer(dim);
        // hidden_dim remains 0 → FFN skipped.

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, dim));

        const float x[] = {3.0f, 4.0f};
        float output[dim] = {};
        assert(minllama::transformer_layer_decode_f32(
            layer, x, cache, 0, output));

        // Same result as step 13 attention-only identity test.
        assert(close(output[0], 3.848528f));
        assert(close(output[1], 5.13137f));
    }

    // ----------------------------------------------------------------
    // Test 2: FFN enabled — full attention + FFN, hand-computed result.
    // dim=2, hidden_dim=2, all identity weights, x=[1,1].
    // ----------------------------------------------------------------
    {
        auto layer = make_ffn_identity_layer(dim, dim);

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, dim));

        const float x[] = {1.0f, 1.0f};
        float output[dim] = {};
        assert(minllama::transformer_layer_decode_f32(
            layer, x, cache, 0, output));

        // Manual computation:
        // attention residual h = [2, 2]  (see step 13 calculation)
        // rmsnorm(h=[2,2]) → norm_h = [1, 1]
        // gate = up = [1, 1]
        // silu(1) = 1/(1+e^{-1}) ≈ 0.7310586
        // hidden = [0.7310586, 0.7310586]
        // ffn_out = [0.7310586, 0.7310586]
        // output = h + ffn_out = [2.7310586, 2.7310586]
        const float expected = 2.0f + 0.7310586f;
        assert(close(output[0], expected));
        assert(close(output[1], expected));
    }

    // ----------------------------------------------------------------
    // Test 3: FFN partial weights — hidden_dim > 0 but some weights empty.
    // ----------------------------------------------------------------
    {
        auto layer = make_att_identity_layer(dim);
        layer.hidden_dim = 2;
        layer.rms_ffn_weight.assign(dim, 1.0f);
        layer.w1.assign(2 * dim, 1.0f);  // ok
        // w2 and w3 are empty → should fail

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, dim));

        const float x[] = {1.0f, 1.0f};
        float output[dim] = {};
        assert(!minllama::transformer_layer_decode_f32(
            layer, x, cache, 0, output));
    }

    // ----------------------------------------------------------------
    // Test 4: FFN bad size — hidden_dim > 0, weights present but wrong sizes.
    // ----------------------------------------------------------------
    {
        auto layer = make_att_identity_layer(dim);
        layer.hidden_dim = 2;
        layer.rms_ffn_weight.assign(dim, 1.0f);
        // w1 should be 2*2=4 but we give 3
        layer.w1.assign(3, 1.0f);
        layer.w2.assign(4, 1.0f);
        layer.w3.assign(4, 1.0f);

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, dim));

        const float x[] = {1.0f, 1.0f};
        float output[dim] = {};
        assert(!minllama::transformer_layer_decode_f32(
            layer, x, cache, 0, output));
    }

    return 0;
}
