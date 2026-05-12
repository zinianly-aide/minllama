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
    // Test 1: embedding lookup - vocab=3, dim=2, token_id=1.
    // ----------------------------------------------------------------
    {
        constexpr int dim = 2;
        constexpr int vocab_size = 3;

        minllama::TransformerModelF32 model;
        model.dim = dim;
        model.vocab_size = vocab_size;

        // token_embedding [vocab*dim] row-major:
        //   token 0: [1.0, 0.0]
        //   token 1: [3.0, 4.0]
        //   token 2: [5.0, 6.0]
        model.token_embedding = {
            1.0f, 0.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
        };

        float out[2] = {};
        assert(minllama::token_embedding_lookup_f32(model, 1, out));
        assert(close(out[0], 3.0f));
        assert(close(out[1], 4.0f));
    }

    // ----------------------------------------------------------------
    // Test 2: token step simple — input token_id, embedding + forward
    // produce correct next_token_id.
    // dim=2, vocab=3.  lm_head designed so token 2 wins.
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

        // token_embedding:
        //   token 0: [1.0, 0.0]
        //   token 1: [0.0, 1.0]
        //   token 2: [2.0, 2.0]
        model.token_embedding = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            2.0f, 2.0f,
        };

        // lm_head makes token 2 the argmax:
        model.lm_head = {
            1.0f, 0.0f,   // token 0
            0.0f, 1.0f,   // token 1
            2.0f, 2.0f,   // token 2  ← dot([2,2], [1,1]) = 4
        };

        int next_id = -1;
        assert(minllama::transformer_model_greedy_token_step_f32(
            model, 0, 0, &next_id));
        // token 0 embedding = [1,0] → identity model → norm([1,0]) = [1,0]
        // logit[0] = dot([1,0],[1,0])=1
        // logit[1] = dot([0,1],[1,0])=0
        // logit[2] = dot([2,2],[1,0])=2  ← argmax
        assert(next_id == 2);
    }

    // ----------------------------------------------------------------
    // Test 3: generate one token.
    // prompt=[0], max_new_tokens=1 → output 1 token.
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

        model.token_embedding = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            2.0f, 2.0f,
        };

        model.lm_head = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            2.0f, 2.0f,
        };

        const int prompt[] = {0};
        constexpr int max_new = 1;
        int output_tokens[max_new] = {};
        int output_len = -1;

        assert(minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, max_new, output_tokens, max_new, &output_len));
        assert(output_len == 1);
        assert(output_tokens[0] == 2);  // token 2 is argmax
    }

    // ----------------------------------------------------------------
    // Test 4: generate multiple tokens.
    // prompt=[0], max_new_tokens=3 → output 3 tokens, all in [0, vocab).
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
        assert(minllama::kv_cache_init_f32(model.kv_caches[0], 8, dim));
        model.rms_norm_eps = 1e-6f;
        model.final_norm_weight.assign(dim, 1.0f);
        model.vocab_size = vocab_size;

        model.token_embedding = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            2.0f, 2.0f,
        };

        model.lm_head = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            2.0f, 2.0f,
        };

        const int prompt[] = {0};
        constexpr int max_new = 3;
        int output_tokens[max_new] = {};
        int output_len = -1;

        assert(minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, max_new, output_tokens, max_new, &output_len));
        assert(output_len == 3);

        for (int i = 0; i < output_len; ++i) {
            assert(output_tokens[i] >= 0);
            assert(output_tokens[i] < vocab_size);
        }
    }

    // ----------------------------------------------------------------
    // Test 5: invalid — bad token_embedding size.
    // ----------------------------------------------------------------
    {
        constexpr int dim = 2;
        constexpr int vocab_size = 3;

        minllama::TransformerModelF32 model;
        model.dim = dim;
        model.vocab_size = vocab_size;
        // token_embedding is empty (size 0 ≠ 6)
        model.token_embedding.clear();

        float out[2];
        assert(!minllama::token_embedding_lookup_f32(model, 1, out));
    }

    // ----------------------------------------------------------------
    // Test 6: invalid — token_id out of bounds.
    // ----------------------------------------------------------------
    {
        constexpr int dim = 2;
        constexpr int vocab_size = 3;

        minllama::TransformerModelF32 model;
        model.dim = dim;
        model.vocab_size = vocab_size;
        model.token_embedding.assign(vocab_size * dim, 0.0f);

        float out[2];
        // token_id = 3 >= vocab_size
        assert(!minllama::token_embedding_lookup_f32(model, 3, out));
        // negative token_id
        assert(!minllama::token_embedding_lookup_f32(model, -1, out));
    }

    // ----------------------------------------------------------------
    // Test 7: invalid — prompt_len <= 0.
    // ----------------------------------------------------------------
    {
        minllama::TransformerModelF32 model;
        model.dim = 2;
        model.vocab_size = 3;
        model.n_layers = 1;
        model.layers.push_back(make_identity_layer(2));
        model.kv_caches.push_back({});
        assert(minllama::kv_cache_init_f32(model.kv_caches[0], 4, 2));
        model.final_norm_weight.assign(2, 1.0f);
        model.token_embedding.assign(6, 0.0f);
        model.lm_head.assign(6, 1.0f);

        const int prompt[] = {0};
        int output[1];
        int output_len = -1;

        assert(!minllama::transformer_model_generate_greedy_f32(
            model, prompt, 0, 1, output, 1, &output_len));

        assert(!minllama::transformer_model_generate_greedy_f32(
            model, prompt, -1, 1, output, 1, &output_len));
    }

    // ----------------------------------------------------------------
    // Test 8: invalid — output_capacity < max_new_tokens.
    // ----------------------------------------------------------------
    {
        minllama::TransformerModelF32 model;
        model.dim = 2;
        model.vocab_size = 3;
        model.n_layers = 1;
        model.layers.push_back(make_identity_layer(2));
        model.kv_caches.push_back({});
        assert(minllama::kv_cache_init_f32(model.kv_caches[0], 4, 2));
        model.final_norm_weight.assign(2, 1.0f);
        model.token_embedding.assign(6, 0.0f);
        model.lm_head.assign(6, 1.0f);

        const int prompt[] = {0};
        int output[1];
        int output_len = -1;

        // capacity=1, max_new=2 → false
        assert(!minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, 2, output, 1, &output_len));
    }

    // ----------------------------------------------------------------
    // Test 9: invalid — nullptr args.
    // ----------------------------------------------------------------
    {
        float out[2];
        // nullptr output
        minllama::TransformerModelF32 m;
        m.dim = 2;
        m.vocab_size = 3;
        m.token_embedding.assign(6, 0.0f);
        assert(!minllama::token_embedding_lookup_f32(m, 0, nullptr));
    }

    {
        minllama::TransformerModelF32 model;
        model.dim = 2;
        model.vocab_size = 3;
        model.n_layers = 1;
        model.layers.push_back(make_identity_layer(2));
        model.kv_caches.push_back({});
        assert(minllama::kv_cache_init_f32(model.kv_caches[0], 4, 2));
        model.final_norm_weight.assign(2, 1.0f);
        model.token_embedding.assign(6, 0.0f);
        model.lm_head.assign(6, 1.0f);

        const int prompt[] = {0};
        int output[1];
        int output_len = -1;

        assert(!minllama::transformer_model_generate_greedy_f32(
            model, nullptr, 1, 1, output, 1, &output_len));
        assert(!minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, 1, nullptr, 1, &output_len));
        assert(!minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, 1, output, 1, nullptr));

        // nullptr next_token_id
        minllama::TransformerModelF32 m2;
        m2.dim = 2;
        m2.vocab_size = 3;
        m2.n_layers = 1;
        m2.layers.push_back(make_identity_layer(2));
        m2.kv_caches.push_back({});
        minllama::kv_cache_init_f32(m2.kv_caches[0], 4, 2);
        m2.rms_norm_eps = 1e-6f;
        m2.final_norm_weight.assign(2, 1.0f);
        m2.token_embedding.assign(6, 0.0f);
        m2.lm_head.assign(6, 1.0f);
        assert(!minllama::transformer_model_greedy_token_step_f32(
            m2, 0, 0, nullptr));
    }

    // ----------------------------------------------------------------
    // Test 10: max_new_tokens=0 — prefill only, output_len=0.
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

        model.token_embedding = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            2.0f, 2.0f,
        };

        model.lm_head = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            2.0f, 2.0f,
        };

        const int prompt[] = {0, 1};
        int output_tokens[1] = {};
        int output_len = -1;

        assert(minllama::transformer_model_generate_greedy_f32(
            model, prompt, 2, 0, output_tokens, 1, &output_len));
        assert(output_len == 0);
    }

    // ----------------------------------------------------------------
    // Test 11: invalid — max_new_tokens < 0.
    // ----------------------------------------------------------------
    {
        minllama::TransformerModelF32 model;
        model.dim = 2;
        model.vocab_size = 3;
        model.n_layers = 1;
        model.layers.push_back(make_identity_layer(2));
        model.kv_caches.push_back({});
        assert(minllama::kv_cache_init_f32(model.kv_caches[0], 4, 2));
        model.final_norm_weight.assign(2, 1.0f);
        model.token_embedding.assign(6, 0.0f);
        model.lm_head.assign(6, 1.0f);

        const int prompt[] = {0};
        int output[1];
        int output_len = -1;

        assert(!minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, -1, output, 1, &output_len));
    }

    return 0;
}
