#include "minllama_internal.h"

#include <cassert>
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

// Build a model where token 0 produces token 1.
// vocab=3, dim=2.
// token_embedding: [0]=[1,0], [1]=[0,1], [2]=[0,0]
// lm_head: [0]=[0,0], [1]=[1,0], [2]=[0,0]
// → input 0 produces 1 (argmax=1)
minllama::TransformerModelF32 make_seq_model() {
    constexpr int dim = 2;
    constexpr int vocab_size = 3;

    auto layer = make_identity_layer(dim);

    minllama::TransformerModelF32 model;
    model.dim = dim;
    model.n_layers = 1;
    model.layers.push_back(layer);
    model.kv_caches.push_back({});
    minllama::kv_cache_init_f32(model.kv_caches[0], 16, dim);
    model.rms_norm_eps = 1e-6f;
    model.final_norm_weight.assign(dim, 1.0f);
    model.vocab_size = vocab_size;

    model.token_embedding = {
        1.0f, 0.0f,  // token 0
        0.0f, 1.0f,  // token 1
        0.0f, 0.0f,  // token 2
    };

    model.lm_head = {
        0.0f, 0.0f,  // token 0
        1.0f, 0.0f,  // token 1
        0.0f, 0.0f,  // token 2
    };

    return model;
}

} // namespace

int main() {
    // ----------------------------------------------------------------
    // Test 1: no eos configured (eos_token_id=-1).
    // Generates all max_new_tokens.
    // ----------------------------------------------------------------
    {
        auto model = make_seq_model();

        const int prompt[] = {0};
        constexpr int max_new = 4;
        int output[max_new] = {};
        int output_len = -1;

        assert(minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, max_new, output, max_new, &output_len, -1));
        assert(output_len == 4);
    }

    // ----------------------------------------------------------------
    // Test 2: eos first token.
    // First generated token is eos → output_len==1, stops.
    // Build model where token 0 produces token 2, and eos=2.
    // ----------------------------------------------------------------
    {
        // Same model but with eos=2. The model makes token 0 produce 1.
        // Actually, this model produces token 1 from token 0, not 2.
        // Let me build a model where token 0 → 2 (eos).
        constexpr int dim = 2;
        constexpr int vocab_size = 3;

        auto layer = make_identity_layer(dim);

        minllama::TransformerModelF32 model;
        model.dim = dim;
        model.n_layers = 1;
        model.layers.push_back(layer);
        model.kv_caches.push_back({});
        minllama::kv_cache_init_f32(model.kv_caches[0], 16, dim);
        model.rms_norm_eps = 1e-6f;
        model.final_norm_weight.assign(dim, 1.0f);
        model.vocab_size = vocab_size;

        // token_embedding: token 0 = [1, 0]
        model.token_embedding = {
            1.0f, 0.0f,  // token 0
            0.0f, 1.0f,  // token 1
            0.0f, 0.0f,  // token 2
        };

        // lm_head: make token 2 (eos) the argmax
        model.lm_head = {
            0.0f, 0.0f,  // token 0
            0.0f, 1.0f,  // token 1
            1.0f, 0.0f,  // token 2 ← argmax
        };

        const int prompt[] = {0};
        constexpr int max_new = 4;
        int output[max_new] = {};
        int output_len = -1;

        assert(minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, max_new, output, max_new, &output_len, 2));
        // First generated token should be 2 (eos), stops immediately.
        assert(output_len == 1);
        assert(output[0] == 2);
    }

    // ----------------------------------------------------------------
    // Test 3: eos after second token.
    // Build model: 0→1→2, eos=2. Should get output_len==2.
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
        minllama::kv_cache_init_f32(model.kv_caches[0], 16, dim);
        model.rms_norm_eps = 1e-6f;
        model.final_norm_weight.assign(dim, 1.0f);
        model.vocab_size = vocab_size;

        // Embeddings: 0=[1,0], 1=[0,1], 2=[0,0]
        model.token_embedding = {
            1.0f, 0.0f,  // token 0
            0.0f, 1.0f,  // token 1
            0.0f, 0.0f,  // token 2
        };

        // lm_head:
        //   token 0→? (for token 0 input)
        //   token 1→? (for token 1 input)
        //   token 2→? (for token 2 input)
        // We want: input 0 → output 1, input 1 → output 2
        // With identity model + RMSNorm:
        //   input [1,0] → norm [√2,0]
        //   input [0,1] → norm [0,√2]
        // Need: dot(lm_head[1], [√2,0]) = max → lm_head[1]=[1,0]
        //       dot(lm_head[2], [0,√2]) = max → lm_head[2]=[0,1]
        model.lm_head = {
            0.0f, 0.0f,  // token 0
            1.0f, 0.0f,  // token 1 ← max for input 0
            0.0f, 1.0f,  // token 2 ← max for input 1
        };

        const int prompt[] = {0};
        constexpr int max_new = 4;
        int output[max_new] = {};
        int output_len = -1;

        assert(minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, max_new, output, max_new, &output_len, 2));
        // 0 → 1 → 2(eos), stops at output_len=2.
        assert(output_len == 2);
        assert(output[0] == 1);
        assert(output[1] == 2);
    }

    // ----------------------------------------------------------------
    // Test 4: max_new_tokens == 0 with eos.
    // ----------------------------------------------------------------
    {
        auto model = make_seq_model();

        const int prompt[] = {0};
        int output[1] = {};
        int output_len = -1;

        assert(minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, 0, output, 1, &output_len, 2));
        assert(output_len == 0);
    }

    // ----------------------------------------------------------------
    // Test 5: eos_id out of vocab → returns false.
    // ----------------------------------------------------------------
    {
        auto model = make_seq_model();

        const int prompt[] = {0};
        int output[4] = {};
        int output_len = -1;

        assert(!minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, 4, output, 4, &output_len, 3));
        // vocab_size=3, eos=3 is out of range
    }

    // ----------------------------------------------------------------
    // Test 6: eos_id out of vocab with eos=-1 (no eos) — still passes.
    // ----------------------------------------------------------------
    {
        auto model = make_seq_model();

        const int prompt[] = {0};
        int output[4] = {};
        int output_len = -1;

        // eos=-1 means no eos, should succeed.
        assert(minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, 4, output, 4, &output_len, -1));
        assert(output_len == 4);
    }

    // ----------------------------------------------------------------
    // Test 7: eos that is NOT matched — no early stop.
    // Model: 0→1→1→1... (never hits eos=2).
    // With max_new=5, should produce 5 tokens.
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
        minllama::kv_cache_init_f32(model.kv_caches[0], 16, dim);
        model.rms_norm_eps = 1e-6f;
        model.final_norm_weight.assign(dim, 1.0f);
        model.vocab_size = vocab_size;

        // token_embedding: token 0 = [1, 0]
        model.token_embedding = {
            1.0f, 0.0f,  // 0
            0.0f, 1.0f,  // 1
            0.0f, 0.0f,  // 2
        };
        // lm_head: token 1 is always argmax
        // For input [1,0] norm [√2,0]: logit[1]=√2, others 0
        // For input [0,1] norm [0,√2]: logit[1]=0, but token 0 has no advantage
        // Actually: for [0,√2]: dot with [0,1]=√2, dot with [1,0]=0
        // Wait, we want token 1 to always win regardless of input.
        // For input [0,√2]: need lm_head[1] = [0,1] → dot = √2
        // For input [√2,0]: need lm_head[1] = [1,0] → dot = √2
        // These conflict for a single lm_head. Let me just use token 0 as the
        // non-eos recurring token.
        // Model: 0→1→0→1→0... never hits eos=2.
        // Input 0 embedding [1,0] norm [√2,0]: lm_head[1]=[1,0] → √2
        // Input 1 embedding [0,1] norm [0,√2]: lm_head[0]=[0,1] → √2
        model.lm_head = {
            0.0f, 1.0f,  // 0 ← max for input 1
            1.0f, 0.0f,  // 1 ← max for input 0
            0.0f, 0.0f,  // 2
        };

        const int prompt[] = {0};
        constexpr int max_new = 5;
        int output[max_new] = {};
        int output_len = -1;

        // eos=2 but model never produces it. Should generate all 5.
        assert(minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, max_new, output, max_new, &output_len, 2));
        assert(output_len == 5);
        // All tokens should be 0 or 1, never 2.
        for (int i = 0; i < output_len; ++i) {
            assert(output[i] != 2);
        }
    }

    return 0;
}
