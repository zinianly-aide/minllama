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
    constexpr int vocab_size = 3;

    // ----------------------------------------------------------------
    // Test 1: Logits simple — hand-computed result.
    // dim=2, vocab_size=3, 1 identity layer, x=[1,1], position=0.
    // ----------------------------------------------------------------
    {
        auto layer = make_identity_layer(dim);

        minllama::TransformerModelF32 model;
        model.dim = dim;
        model.n_layers = 1;
        model.layers.push_back(layer);
        model.kv_caches.push_back({});
        assert(minllama::kv_cache_init_f32(model.kv_caches[0], 4, dim));

        model.rms_norm_eps = 1e-6f;
        model.final_norm_weight.assign(dim, 1.0f);
        model.vocab_size = vocab_size;

        // lm_head: 3×2 row-major
        // token 0: [1, 0]
        // token 1: [0, 1]
        // token 2: [1, 1]
        model.lm_head = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f,
        };

        const float x[] = {1.0f, 1.0f};
        float logits[vocab_size] = {};
        assert(minllama::transformer_model_logits_f32(
            model, x, 0, logits));

        // Manual:
        // h (attention residual) = [2, 2]
        // rmsnorm([2,2]) with [1,1]: sum_sq=8, mean_sq=4, scale=0.5 → [1, 1]
        // logit[0] = dot([1,0],[1,1]) = 1
        // logit[1] = dot([0,1],[1,1]) = 1
        // logit[2] = dot([1,1],[1,1]) = 2
        assert(close(logits[0], 1.0f));
        assert(close(logits[1], 1.0f));
        assert(close(logits[2], 2.0f));
    }

    // ----------------------------------------------------------------
    // Test 2: Invalid model configurations.
    // ----------------------------------------------------------------
    {
        auto layer = make_identity_layer(dim);
        minllama::KvCacheF32 cache;
        assert(minllama::kv_cache_init_f32(cache, 4, dim));

        auto make_good = [&]() {
            minllama::TransformerModelF32 m;
            m.dim = dim;
            m.n_layers = 1;
            m.layers.push_back(layer);
            m.kv_caches.push_back(cache);
            m.rms_norm_eps = 1e-6f;
            m.final_norm_weight.assign(dim, 1.0f);
            m.vocab_size = vocab_size;
            m.lm_head.assign(vocab_size * dim, 1.0f);
            return m;
        };

        const float x[] = {1.0f, 1.0f};
        float out[vocab_size] = {};

        // vocab_size = 0
        {
            auto bad = make_good();
            bad.vocab_size = 0;
            assert(!minllama::transformer_model_logits_f32(bad, x, 0, out));
        }

        // final_norm_weight size wrong
        {
            auto bad = make_good();
            bad.final_norm_weight.pop_back();
            assert(!minllama::transformer_model_logits_f32(bad, x, 0, out));
        }

        // lm_head size wrong
        {
            auto bad = make_good();
            bad.lm_head.pop_back();
            assert(!minllama::transformer_model_logits_f32(bad, x, 0, out));
        }

        // nullptr x
        {
            auto good = make_good();
            assert(!minllama::transformer_model_logits_f32(
                good, nullptr, 0, out));
        }

        // nullptr logits
        {
            auto good = make_good();
            assert(!minllama::transformer_model_logits_f32(
                good, x, 0, nullptr));
        }
    }

    // ----------------------------------------------------------------
    // Test 3: Logits call updates KV cache.
    // ----------------------------------------------------------------
    {
        auto layer = make_identity_layer(dim);

        minllama::TransformerModelF32 model;
        model.dim = dim;
        model.n_layers = 1;
        model.layers.push_back(layer);
        model.kv_caches.push_back({});
        assert(minllama::kv_cache_init_f32(model.kv_caches[0], 4, dim));

        model.rms_norm_eps = 1e-6f;
        model.final_norm_weight.assign(dim, 1.0f);
        model.vocab_size = vocab_size;
        model.lm_head.assign(vocab_size * dim, 1.0f);

        const float x[] = {1.0f, 1.0f};
        float logits[vocab_size] = {};
        assert(minllama::transformer_model_logits_f32(
            model, x, 0, logits));

        // KV cache at position 0 should be non-zero.
        auto &cache = model.kv_caches[0];
        assert(!close(cache.keys[0 * dim + 0], 0.0f));
        assert(!close(cache.values[0 * dim + 0], 0.0f));
    }

    return 0;
}
