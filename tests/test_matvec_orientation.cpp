// test_matvec_orientation: verify matvec row-major layout for all weight matrices.
// Tests that minllama's matvec convention is self-consistent and matches manual computation.
#include "minllama_internal.h"
#include <cassert>
#include <cmath>
#include <cstdio>

static float rel_err(float a, float b) {
    float d = std::fabs(a) + std::fabs(b);
    return d < 1e-9f ? 0 : std::fabs(a - b) / d;
}

// Manual matvec for comparison: output[r] = sum_c weight[r*cols + c] * input[c]
static void manual_matvec(const float *weight, int rows, int cols,
                          const float *input, float *output) {
    for (int r = 0; r < rows; ++r) {
        float s = 0;
        for (int c = 0; c < cols; ++c)
            s += weight[r * cols + c] * input[c];
        output[r] = s;
    }
}

int main() {
    // ----------------------------------------------------------------
    // Test 1: matvec_f32_f32 orientation — verify output = manual matvec
    // ----------------------------------------------------------------
    {
        const int R = 3, C = 4;
        // Weight stored row-major: [r0c0, r0c1, r0c2, r0c3, r1c0, ...]
        float w[R*C] = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10,11,12
        };
        float x[C] = {0.5f, -1.0f, 2.0f, 0.0f};

        float out[R] = {};
        assert(minllama::matvec_f32_f32(w, R, C, x, C, out, R));

        // Manual:
        // r0 = 1*0.5 + 2*(-1) + 3*2 + 4*0 = 0.5 - 2 + 6 = 4.5
        // r1 = 5*0.5 + 6*(-1) + 7*2 + 8*0 = 2.5 - 6 + 14 = 10.5
        // r2 = 9*0.5 + 10*(-1) + 11*2 + 12*0 = 4.5 - 10 + 22 = 16.5
        float expected[R] = {4.5f, 10.5f, 16.5f};
        for (int i = 0; i < R; ++i)
            assert(rel_err(out[i], expected[i]) < 1e-5f);
        std::printf("Test1 passed: matvec_f32_f32 orientation correct\n");
    }

    // ----------------------------------------------------------------
    // Test 2: token_embedding_lookup_f32 — verify row extraction
    // ----------------------------------------------------------------
    {
        // Simulate a small embedding table: [vocab=3, dim=4]
        float emb_table[3*4] = {
            0.1f, 0.2f, 0.3f, 0.4f,  // token 0
            1.1f, 1.2f, 1.3f, 1.4f,  // token 1
            2.1f, 2.2f, 2.3f, 2.4f,  // token 2
        };
        minllama::TransformerModelF32 model;
        model.dim = 4;
        model.vocab_size = 3;
        model.token_embedding.assign(emb_table, emb_table + 12);

        float out[4];
        assert(minllama::token_embedding_lookup_f32(model, 0, out));
        assert(rel_err(out[0], 0.1f) < 1e-5f);
        assert(rel_err(out[2], 0.3f) < 1e-5f);

        assert(minllama::token_embedding_lookup_f32(model, 2, out));
        assert(rel_err(out[0], 2.1f) < 1e-5f);
        assert(rel_err(out[3], 2.4f) < 1e-5f);
        std::printf("Test2 passed: token embedding lookup correct\n");
    }

    // ----------------------------------------------------------------
    // Test 3: SmolLM real weights — verify matvec = manual matvec
    // ----------------------------------------------------------------
    {
        ml_model *ml = ml_model_load("../models/SmolLM-135M.Q4_0.gguf");
        if (!ml) { std::fprintf(stderr, "skip model load\n"); return 0; }

        minllama::TransformerModelF32 model;
        std::string err;
        {
            bool ok = minllama::load_transformer_model_f32_from_tensors(*ml, model, &err);
            assert(ok);
        }

        auto &l = model.layers[0];
        int dim = l.dim;

        // Create a known input vector.
        std::vector<float> input(dim, 0.0f);
        for (int i = 0; i < dim; ++i) input[i] = 0.01f * (i + 1);  // increasing ramp

        // Test Wq
        {
            std::vector<float> out1(dim), out2(dim);
            assert(minllama::matvec_f32_f32(l.wq.data(), dim, dim, input.data(), dim, out1.data(), dim));
            manual_matvec(l.wq.data(), dim, dim, input.data(), out2.data());
            for (int i = 0; i < dim; ++i)
                assert(rel_err(out1[i], out2[i]) < 1e-5f);
        }
        // Test Wk
        {
            int kv_dim = l.n_kv_heads * l.head_dim;
            std::vector<float> out1(kv_dim), out2(kv_dim);
            assert(minllama::matvec_f32_f32(l.wk.data(), kv_dim, dim, input.data(), dim, out1.data(), kv_dim));
            manual_matvec(l.wk.data(), kv_dim, dim, input.data(), out2.data());
            for (int i = 0; i < kv_dim; ++i)
                assert(rel_err(out1[i], out2[i]) < 1e-5f);
        }
        // Test Wv
        {
            int kv_dim = l.n_kv_heads * l.head_dim;
            std::vector<float> out1(kv_dim), out2(kv_dim);
            assert(minllama::matvec_f32_f32(l.wv.data(), kv_dim, dim, input.data(), dim, out1.data(), kv_dim));
            manual_matvec(l.wv.data(), kv_dim, dim, input.data(), out2.data());
            for (int i = 0; i < kv_dim; ++i)
                assert(rel_err(out1[i], out2[i]) < 1e-5f);
        }
        // Test Wo
        {
            std::vector<float> out1(dim), out2(dim);
            assert(minllama::matvec_f32_f32(l.wo.data(), dim, dim, input.data(), dim, out1.data(), dim));
            manual_matvec(l.wo.data(), dim, dim, input.data(), out2.data());
            for (int i = 0; i < dim; ++i)
                assert(rel_err(out1[i], out2[i]) < 1e-5f);
        }
        // Test W1 (gate)
        {
            int hd = l.hidden_dim;
            std::vector<float> out1(hd), out2(hd);
            assert(minllama::matvec_f32_f32(l.w1.data(), hd, dim, input.data(), dim, out1.data(), hd));
            manual_matvec(l.w1.data(), hd, dim, input.data(), out2.data());
            for (int i = 0; i < hd; ++i)
                assert(rel_err(out1[i], out2[i]) < 1e-5f);
        }
        // Test W2 (down) — input is hidden_dim, output is dim
        {
            int hd = l.hidden_dim;
            std::vector<float> in_hd(hd, 0.0f);
            for (int i = 0; i < hd; ++i) in_hd[i] = 0.01f * (i + 1);
            std::vector<float> out1(dim), out2(dim);
            assert(minllama::matvec_f32_f32(l.w2.data(), dim, hd, in_hd.data(), hd, out1.data(), dim));
            manual_matvec(l.w2.data(), dim, hd, in_hd.data(), out2.data());
            for (int i = 0; i < dim; ++i)
                assert(rel_err(out1[i], out2[i]) < 1e-5f);
        }
        // Test W3 (up)
        {
            int hd = l.hidden_dim;
            std::vector<float> out1(hd), out2(hd);
            assert(minllama::matvec_f32_f32(l.w3.data(), hd, dim, input.data(), dim, out1.data(), hd));
            manual_matvec(l.w3.data(), hd, dim, input.data(), out2.data());
            for (int i = 0; i < hd; ++i)
                assert(rel_err(out1[i], out2[i]) < 1e-5f);
        }
        // Test lm_head
        {
            std::vector<float> out1(model.vocab_size), out2(model.vocab_size);
            assert(minllama::matvec_f32_f32(model.lm_head.data(), model.vocab_size, dim, input.data(), dim, out1.data(), model.vocab_size));
            manual_matvec(model.lm_head.data(), model.vocab_size, dim, input.data(), out2.data());
            for (int i = 0; i < 100; ++i)  // check first 100
                assert(rel_err(out1[i], out2[i]) < 1e-5f);
        }
        std::printf("Test3 passed: all SmolLM weight matvecs == manual matvecs\n");

        ml_model_free(ml);
    }

    // ----------------------------------------------------------------
    // Test 4: verify RMSNorm orientation
    // ----------------------------------------------------------------
    {
        float x[4] = {1, 2, 3, 4};
        float w[4] = {0.5f, 1.0f, 1.5f, 2.0f};
        float out[4];
        assert(minllama::rmsnorm_f32(x, w, 4, 1e-5f, out, 4));

        // mean_sq = (1+4+9+16)/4 = 7.5
        // scale = 1/sqrt(7.5 + 1e-5) ≈ 0.365148
        // out[0] = 0.5 * 1 * 0.365148 = 0.182574
        float mean_sq = 7.5f;
        float scale = 1.0f / std::sqrt(mean_sq + 1e-5f);
        float e0 = w[0] * x[0] * scale;
        assert(rel_err(out[0], e0) < 1e-5f);
        std::printf("Test4 passed: rmsnorm orientation correct\n");
    }

    std::printf("\nAll matvec orientation tests passed.\n");
    return 0;
}
