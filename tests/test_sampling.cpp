#include "minllama_internal.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

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
    // Test 1: RNG determinism — same seed, same output.
    // ----------------------------------------------------------------
    {
        uint32_t s1 = 42;
        uint32_t s2 = 42;
        for (int i = 0; i < 10; ++i) {
            assert(minllama::rng_next_u32(&s1) == minllama::rng_next_u32(&s2));
        }
    }

    // ----------------------------------------------------------------
    // Test 2: RNG produces different values.
    // ----------------------------------------------------------------
    {
        uint32_t s = 1;
        uint32_t a = minllama::rng_next_u32(&s);
        uint32_t b = minllama::rng_next_u32(&s);
        assert(a != b);
    }

    // ----------------------------------------------------------------
    // Test 3: rng_uniform01 in [0, 1).
    // ----------------------------------------------------------------
    {
        uint32_t s = 123;
        for (int i = 0; i < 100; ++i) {
            float v = minllama::rng_uniform01(&s);
            assert(v >= 0.0f);
            assert(v < 1.0f);
        }
    }

    // ----------------------------------------------------------------
    // Test 4: sample_temperature_f32 — temperature <= 0 → greedy.
    // ----------------------------------------------------------------
    {
        const float logits[] = {1.0f, 3.0f, 2.0f};
        uint32_t rng = 42;
        int result = minllama::sample_temperature_f32(logits, 3, 0.0f, &rng);
        assert(result == 1); // argmax
    }

    // ----------------------------------------------------------------
    // Test 5: sample_temperature_f32 — deterministic same seed.
    // ----------------------------------------------------------------
    {
        const float logits[] = {0.1f, 0.2f, 0.7f, 0.0f};
        uint32_t rng1 = 42;
        uint32_t rng2 = 42;
        int r1 = minllama::sample_temperature_f32(logits, 4, 1.0f, &rng1);
        int r2 = minllama::sample_temperature_f32(logits, 4, 1.0f, &rng2);
        assert(r1 == r2);
        assert(r1 >= 0 && r1 < 4);
    }

    // ----------------------------------------------------------------
    // Test 6: sample_temperature_f32 — heavily skewed logits.
    // logits=[999, 0, 0], T=1 → nearly always token 0.
    // ----------------------------------------------------------------
    {
        const float logits[] = {999.0f, 0.0f, 0.0f};
        uint32_t rng = 1;
        // Sample many times; should all be token 0.
        bool all_zero = true;
        for (int i = 0; i < 20; ++i) {
            int t = minllama::sample_temperature_f32(logits, 3, 1.0f, &rng);
            assert(t >= 0 && t < 3);
            if (t != 0) all_zero = false;
        }
        assert(all_zero);
    }

    // ----------------------------------------------------------------
    // Test 7: sample_temperature_f32 — high temperature flattens.
    // logits=[100, 1, 1], T=100 → almost uniform.
    // Just verify tokens are valid.
    // ----------------------------------------------------------------
    {
        const float logits[] = {100.0f, 1.0f, 1.0f};
        uint32_t rng = 7;
        for (int i = 0; i < 10; ++i) {
            int t = minllama::sample_temperature_f32(logits, 3, 100.0f, &rng);
            assert(t >= 0 && t < 3);
        }
    }

    // ----------------------------------------------------------------
    // Test 8: invalid — nullptr logits.
    // ----------------------------------------------------------------
    {
        uint32_t rng = 1;
        assert(minllama::sample_temperature_f32(nullptr, 3, 1.0f, &rng) == -1);
    }

    // ----------------------------------------------------------------
    // Test 9: invalid — vocab_size <= 0.
    // ----------------------------------------------------------------
    {
        const float logits[] = {1.0f};
        uint32_t rng = 1;
        assert(minllama::sample_temperature_f32(logits, 0, 1.0f, &rng) == -1);
        assert(minllama::sample_temperature_f32(logits, -1, 1.0f, &rng) == -1);
    }

    // ----------------------------------------------------------------
    // Test 10: invalid — nullptr rng_state.
    // ----------------------------------------------------------------
    {
        const float logits[] = {1.0f, 2.0f};
        assert(minllama::sample_temperature_f32(logits, 2, 1.0f, nullptr) == -1);
    }

    // ----------------------------------------------------------------
    // Test 11: transformer_model_sample_step_f32 — basic.
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

        model.lm_head = {
            1.0f, 0.0f,  // token 0
            0.0f, 1.0f,  // token 1
            2.0f, 2.0f,  // token 2 ← argmax
        };

        const float x[] = {1.0f, 1.0f};
        uint32_t rng = 123;
        int tid = -1;

        // temperature=0 → greedy.
        assert(minllama::transformer_model_sample_step_f32(
            model, x, 0, 0.0f, &rng, &tid));
        assert(tid == 2); // argmax

        // temperature=1 → sampling, still valid.
        rng = 456;
        assert(minllama::transformer_model_sample_step_f32(
            model, x, 0, 1.0f, &rng, &tid));
        assert(tid >= 0 && tid < vocab_size);
    }

    // ----------------------------------------------------------------
    // Test 12: transformer_model_sample_step_f32 — invalid args.
    // ----------------------------------------------------------------
    {
        minllama::TransformerModelF32 model;
        model.dim = 2;
        model.vocab_size = 3;
        uint32_t rng = 1;
        int tid = -1;
        const float x[] = {1.0f, 1.0f};

        // nullptr token_id.
        assert(!minllama::transformer_model_sample_step_f32(
            model, x, 0, 1.0f, &rng, nullptr));
        // nullptr rng_state.
        assert(!minllama::transformer_model_sample_step_f32(
            model, x, 0, 1.0f, nullptr, &tid));
        // vocab_size=0.
        model.vocab_size = 0;
        assert(!minllama::transformer_model_sample_step_f32(
            model, x, 0, 1.0f, &rng, &tid));
    }

    return 0;
}
