#include "minllama_internal.h"

#include <cassert>
#include <cmath>
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
    constexpr int dim = 2;

    // ----------------------------------------------------------------
    // Test 1: Single layer model — result matches direct layer call.
    // ----------------------------------------------------------------
    {
        auto layer = make_identity_layer(dim);
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, dim));

        minllama::TransformerModelF32 model;
        model.dim = dim;
        model.n_layers = 1;
        model.layers.push_back(layer);
        // copy the cache so model has its own
        model.kv_caches.push_back({});
        assert(minllama::kv_cache_init_f32(model.kv_caches[0], 4, dim));

        const float x[] = {3.0f, 4.0f};
        float model_out[dim] = {};
        assert(minllama::transformer_model_decode_f32(
            model, x, 0, model_out));

        // Direct layer call for comparison.
        float direct_out[dim] = {};
        assert(minllama::transformer_layer_decode_f32(
            layer, x, cache, 0, direct_out));

        assert(close(model_out[0], direct_out[0]));
        assert(close(model_out[1], direct_out[1]));
    }

    // ----------------------------------------------------------------
    // Test 2: Two-layer model — output finite, non-NaN, both caches written.
    // ----------------------------------------------------------------
    {
        auto layer0 = make_identity_layer(dim);
        auto layer1 = make_identity_layer(dim);

        minllama::TransformerModelF32 model;
        model.dim = dim;
        model.n_layers = 2;
        model.layers.push_back(layer0);
        model.layers.push_back(layer1);
        for (int i = 0; i < 2; ++i) {
            model.kv_caches.push_back({});
            assert(minllama::kv_cache_init_f32(model.kv_caches[i], 4, dim));
        }

        // Position 0
        {
            const float x[] = {1.0f, 1.0f};
            float output[dim] = {};
            assert(minllama::transformer_model_decode_f32(
                model, x, 0, output));
            assert(!std::isnan(output[0]) && !std::isnan(output[1]));
            assert(std::isfinite(output[0]) && std::isfinite(output[1]));
        }

        // Position 1
        {
            const float x[] = {2.0f, 0.0f};
            float output[dim] = {};
            assert(minllama::transformer_model_decode_f32(
                model, x, 1, output));
            assert(!std::isnan(output[0]) && !std::isnan(output[1]));
            assert(std::isfinite(output[0]) && std::isfinite(output[1]));
        }

        // Both layers' caches have non-zero entries.
        for (int layer_idx = 0; layer_idx < 2; ++layer_idx) {
            auto &cache = model.kv_caches[layer_idx];
            assert(!close(cache.keys[0 * dim + 0], 0.0f)); // k at pos 0
            assert(!close(cache.keys[1 * dim + 0], 0.0f)); // k at pos 1
            assert(!close(cache.values[0 * dim + 0], 0.0f)); // v at pos 0
            assert(!close(cache.values[1 * dim + 0], 0.0f)); // v at pos 1
        }
    }

    // ----------------------------------------------------------------
    // Test 3: Invalid model configurations.
    // ----------------------------------------------------------------
    {
        minllama::TransformerModelF32 model;
        auto layer = make_identity_layer(dim);
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, dim));

        const float x[] = {1.0f, 1.0f};
        float out[dim] = {};

        // dim = 0
        {
            auto bad = model;
            bad.dim = 0;
            bad.n_layers = 1;
            bad.layers.push_back(layer);
            bad.kv_caches.push_back(cache);
            assert(!minllama::transformer_model_decode_f32(
                bad, x, 0, out));
        }

        // n_layers = 0
        {
            auto bad = model;
            bad.dim = dim;
            bad.n_layers = 0;
            assert(!minllama::transformer_model_decode_f32(
                bad, x, 0, out));
        }

        // layers size != n_layers
        {
            auto bad = model;
            bad.dim = dim;
            bad.n_layers = 2;
            bad.layers.push_back(layer);
            bad.kv_caches.push_back(cache);
            bad.kv_caches.push_back(cache);
            assert(!minllama::transformer_model_decode_f32(
                bad, x, 0, out));
        }

        // kv_caches size != n_layers
        {
            auto bad = model;
            bad.dim = dim;
            bad.n_layers = 1;
            bad.layers.push_back(layer);
            assert(!minllama::transformer_model_decode_f32(
                bad, x, 0, out));
        }

        // cache dim mismatch
        {
            minllama::KvCacheF32 bad_cache;
            assert(minllama::kv_cache_init_f32(bad_cache, 4, 3)); // dim=3 != 2

            auto bad = model;
            bad.dim = dim;
            bad.n_layers = 1;
            bad.layers.push_back(layer);
            bad.kv_caches.push_back(bad_cache);
            assert(!minllama::transformer_model_decode_f32(
                bad, x, 0, out));
        }

        // nullptr
        {
            auto good = model;
            good.dim = dim;
            good.n_layers = 1;
            good.layers.push_back(layer);
            good.kv_caches.push_back(cache);
            assert(!minllama::transformer_model_decode_f32(
                good, nullptr, 0, out));
            assert(!minllama::transformer_model_decode_f32(
                good, x, 0, nullptr));
        }
    }

    return 0;
}
