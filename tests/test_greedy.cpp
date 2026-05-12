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
    // ----------------------------------------------------------------
    // Test 1: argmax basic.
    // ----------------------------------------------------------------
    {
        const float vals[] = {0.1f, 2.0f, 1.0f};
        assert(minllama::argmax_f32(vals, 3) == 1);
    }

    // ----------------------------------------------------------------
    // Test 2: argmax tie — first index wins.
    // ----------------------------------------------------------------
    {
        const float vals[] = {1.0f, 3.0f, 3.0f};
        assert(minllama::argmax_f32(vals, 3) == 1);
    }

    // ----------------------------------------------------------------
    // Test 3: argmax invalid inputs.
    // ----------------------------------------------------------------
    {
        assert(minllama::argmax_f32(nullptr, 3) == -1);
        assert(minllama::argmax_f32(nullptr, -1) == -1);
        const float v[] = {1.0f};
        assert(minllama::argmax_f32(v, 0) == -1);
        assert(minllama::argmax_f32(v, -1) == -1);
    }

    // ----------------------------------------------------------------
    // Test 4: Greedy step simple.
    // Use a dim=2, vocab=3 model where token 2 gets highest logit.
    // ----------------------------------------------------------------
    {
        constexpr int dim = 2;
        constexpr int vocab_size = 3;

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

        // lm_head: make token 2 produce the largest dot-product.
        // token 0: [1, 0]
        // token 1: [0, 1]
        // token 2: [2, 2]
        model.lm_head = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            2.0f, 2.0f,
        };

        const float x[] = {1.0f, 1.0f};
        int token_id = -1;
        assert(minllama::transformer_model_greedy_step_f32(
            model, x, 0, &token_id));

        // norm_hidden = [1, 1] (identity model with x=[1,1])
        // logit[0] = dot([1,0],[1,1]) = 1
        // logit[1] = dot([0,1],[1,1]) = 1
        // logit[2] = dot([2,2],[1,1]) = 4  ← argmax
        assert(token_id == 2);
    }

    // ----------------------------------------------------------------
    // Test 5: Greedy step invalid inputs.
    // ----------------------------------------------------------------
    {
        constexpr int dim = 2;
        constexpr int vocab_size = 3;

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

        // nullptr token_id
        assert(!minllama::transformer_model_greedy_step_f32(
            model, x, 0, nullptr));

        // bad model: vocab_size = 0
        {
            auto bad = model;
            bad.vocab_size = 0;
            int tid = -1;
            assert(!minllama::transformer_model_greedy_step_f32(
                bad, x, 0, &tid));
        }
    }

    return 0;
}
