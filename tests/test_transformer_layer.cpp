#include "minllama_internal.h"

#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

bool close(float a, float b, float tolerance = 0.001f) {
    return std::fabs(a - b) < tolerance;
}

minllama::TransformerLayerF32 make_identity_layer(int dim) {
    minllama::TransformerLayerF32 layer;
    layer.dim = dim;
    layer.rms_att_weight.assign(dim, 1.0f);
    layer.rms_norm_eps = 1e-6f;
    layer.rope_theta = 10000.0f;

    // Identity matrices: row-major, 1 on diagonal, 0 elsewhere.
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

} // namespace

int main() {
    // ----------------------------------------------------------------
    // Test 1: Identity-like single token decode.
    // dim=2, all weights = I, rms_att_weight = [1,1], eps=1e-6.
    // x = [3, 4], position = 0.
    //
    // Since only one token, attention collapses to v = normalized(x).
    // output = x + projected = x + normalized(x).
    // ----------------------------------------------------------------
    {
        constexpr int dim = 2;
        auto layer = make_identity_layer(dim);

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, dim));

        const float x[] = {3.0f, 4.0f};
        float output[dim] = {};

        assert(minllama::transformer_layer_decode_f32(
            layer, x, cache, 0, output));

        // Manual computation:
        // sum_sq = 9+16 = 25, mean_sq = 12.5
        // scale = 1/sqrt(12.5 + 1e-6) ≈ 0.2828427
        // xn = [3*0.2828427, 4*0.2828427] = [0.848528, 1.13137]
        // att = v = xn (single token + identity weights)
        // projected = att (Wo = I)
        // output = x + projected = [3.848528, 5.13137]
        assert(close(output[0], 3.848528f));
        assert(close(output[1], 5.13137f));
    }

    // ----------------------------------------------------------------
    // Test 2: Two-token decode.
    // Feed two different tokens at position 0 and 1, then verify:
    // - both outputs are finite and not NaN
    // - cache has two tokens worth of data
    // ----------------------------------------------------------------
    {
        constexpr int dim = 2;
        auto layer = make_identity_layer(dim);

        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, dim));

        // Position 0: x0 = [1, 1]
        {
            const float x0[] = {1.0f, 1.0f};
            float out0[dim] = {};
            assert(minllama::transformer_layer_decode_f32(
                layer, x0, cache, 0, out0));
            assert(!std::isnan(out0[0]) && !std::isnan(out0[1]));
            assert(std::isfinite(out0[0]) && std::isfinite(out0[1]));
        }

        // Position 1: x1 = [2, 0]
        {
            const float x1[] = {2.0f, 0.0f};
            float out1[dim] = {};
            assert(minllama::transformer_layer_decode_f32(
                layer, x1, cache, 1, out1));
            assert(!std::isnan(out1[0]) && !std::isnan(out1[1]));
            assert(std::isfinite(out1[0]) && std::isfinite(out1[1]));
        }

        // Cache internal state: keys[0], values[0] should both be non-zero
        // after RoPE (pos 0 RoPE is identity, so equal to normalized x0).
        assert(!close(cache.keys[0], 0.0f));   // k0 written
        assert(!close(cache.keys[2], 0.0f));   // k1 written
        assert(!close(cache.values[0], 0.0f)); // v0 written
        assert(!close(cache.values[2], 0.0f)); // v1 written
    }

    // ----------------------------------------------------------------
    // Test 3: Invalid inputs.
    // ----------------------------------------------------------------
    {
        constexpr int dim = 2;
        auto layer = make_identity_layer(dim);
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, dim));

        const float x[] = {1.0f, 1.0f};
        float out[dim] = {};

        // nullptr x
        {
            auto bad = layer;
            assert(!minllama::transformer_layer_decode_f32(
                bad, nullptr, cache, 0, out));
        }
        // nullptr output
        {
            auto bad = layer;
            assert(!minllama::transformer_layer_decode_f32(
                bad, x, cache, 0, nullptr));
        }
        // dim <= 0
        {
            auto bad = layer;
            bad.dim = 0;
            assert(!minllama::transformer_layer_decode_f32(
                bad, x, cache, 0, out));
        }
        // weight size wrong: wq too small
        {
            auto bad = layer;
            bad.wq.pop_back();
            assert(!minllama::transformer_layer_decode_f32(
                bad, x, cache, 0, out));
        }
        // weight size wrong: rms_att_weight too small
        {
            auto bad = layer;
            bad.rms_att_weight.pop_back();
            assert(!minllama::transformer_layer_decode_f32(
                bad, x, cache, 0, out));
        }
        // cache dim mismatch
        {
            minllama::KvCacheF32 bad_cache;
            assert(minllama::kv_cache_init_f32(bad_cache, 4, 3)); // dim=3 != 2
            assert(!minllama::transformer_layer_decode_f32(
                layer, x, bad_cache, 0, out));
        }
        // position < 0
        assert(!minllama::transformer_layer_decode_f32(
            layer, x, cache, -1, out));
        // position >= max_tokens
        assert(!minllama::transformer_layer_decode_f32(
            layer, x, cache, 4, out));
    }

    return 0;
}
