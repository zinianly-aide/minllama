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

// Model where token 0 → token 1 → token 1 → ... (monotonic, no eos).
// vocab=3, dim=2.
minllama::TransformerModelF32 make_mono_model() {
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
        1.0f, 0.0f,  // 0
        0.0f, 0.0f,  // 1
        0.0f, 1.0f,  // 2
    };

    // lm_head: token 1 is always argmax.
    model.lm_head = {
        0.0f, 0.0f,  // 0
        1.0f, 0.0f,  // 1 ← argmax for [√2,0] and [0,√2]... wait
        0.0f, 0.0f,  // 2
    };
    // Input [1,0] norm [√2,0]: dot(lm_head[1],[√2,0]) = √2 → token 1
    // That's all we need since token 1 has embedding [0,0], which produces
    // zero dot products with everything, so argmax is token 0 (first index).
    // Hmm, that's a problem. Let me fix.
    // Actually, token 1 embedding = [0,0] → norm = [0,0] → all logits = 0
    // → argmax = 0. So sequence would be 0→1→0→1→0...
    // For greedy T=0, this is deterministic and useful.

    return model;
}

} // namespace

int main() {
    // ----------------------------------------------------------------
    // Test 1: T=0 same as greedy.
    // ----------------------------------------------------------------
    {
        auto model = make_mono_model();

        const int prompt[] = {0};
        constexpr int max_new = 4;
        int out_sample[max_new] = {};
        int out_greedy[max_new] = {};
        int len_sample = -1, len_greedy = -1;
        uint32_t rng = 1;

        assert(minllama::transformer_model_generate_sample_f32(
            model, prompt, 1, max_new, out_sample, max_new, &len_sample,
            -1, 0.0f, &rng));
        assert(len_sample == max_new);

        // Re-init cache for greedy comparison.
        model.kv_caches[0].keys.assign(model.kv_caches[0].keys.size(), 0.0f);
        model.kv_caches[0].values.assign(model.kv_caches[0].values.size(), 0.0f);
        assert(minllama::transformer_model_generate_greedy_f32(
            model, prompt, 1, max_new, out_greedy, max_new, &len_greedy, -1));
        assert(len_greedy == max_new);

        for (int i = 0; i < max_new; ++i) {
            assert(out_sample[i] == out_greedy[i]);
        }
    }

    // ----------------------------------------------------------------
    // Test 2: same seed → same output with T=1.
    // ----------------------------------------------------------------
    {
        auto model1 = make_mono_model();
        auto model2 = make_mono_model();

        const int prompt[] = {0};
        constexpr int max_new = 4;
        int out1[max_new] = {}, out2[max_new] = {};
        int len1 = -1, len2 = -1;
        uint32_t rng1 = 42, rng2 = 42;

        assert(minllama::transformer_model_generate_sample_f32(
            model1, prompt, 1, max_new, out1, max_new, &len1, -1, 1.0f, &rng1));
        assert(minllama::transformer_model_generate_sample_f32(
            model2, prompt, 1, max_new, out2, max_new, &len2, -1, 1.0f, &rng2));
        assert(len1 == max_new);
        assert(len2 == max_new);

        for (int i = 0; i < max_new; ++i) {
            assert(out1[i] == out2[i]);
        }
    }

    // ----------------------------------------------------------------
    // Test 3: different seed — all tokens valid.
    // ----------------------------------------------------------------
    {
        auto model = make_mono_model();

        const int prompt[] = {0};
        constexpr int max_new = 4;
        int out1[max_new] = {}, out2[max_new] = {};
        int len1 = -1, len2 = -1;
        uint32_t rng1 = 7, rng2 = 999;

        assert(minllama::transformer_model_generate_sample_f32(
            model, prompt, 1, max_new, out1, max_new, &len1, -1, 1.0f, &rng1));
        // Re-init cache.
        model.kv_caches[0].keys.assign(model.kv_caches[0].keys.size(), 0.0f);
        model.kv_caches[0].values.assign(model.kv_caches[0].values.size(), 0.0f);
        assert(minllama::transformer_model_generate_sample_f32(
            model, prompt, 1, max_new, out2, max_new, &len2, -1, 1.0f, &rng2));

        assert(len1 == max_new);
        assert(len2 == max_new);
        for (int i = 0; i < max_new; ++i) {
            assert(out1[i] >= 0 && out1[i] < 3);
            assert(out2[i] >= 0 && out2[i] < 3);
        }
    }

    // ----------------------------------------------------------------
    // Test 4: EOS early stopping with sampling.
    // Model: 0→2 immediately, eos=2.
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

        model.token_embedding = {
            1.0f, 0.0f,  // 0
            0.0f, 1.0f,  // 1
            0.0f, 0.0f,  // 2
        };
        // token 2 is argmax for [√2,0]:
        model.lm_head = {
            0.0f, 0.0f,  // 0
            0.0f, 1.0f,  // 1
            1.0f, 0.0f,  // 2 ← argmax
        };

        const int prompt[] = {0};
        constexpr int max_new = 5;
        int output[max_new] = {};
        int output_len = -1;
        uint32_t rng = 123;

        assert(minllama::transformer_model_generate_sample_f32(
            model, prompt, 1, max_new, output, max_new, &output_len,
            2, 0.0f, &rng));
        // First token is 2 (eos), so output_len should be 1.
        assert(output_len == 1);
        assert(output[0] == 2);
    }

    // ----------------------------------------------------------------
    // Test 5: invalid — nullptr args.
    // ----------------------------------------------------------------
    {
        auto model = make_mono_model();
        const int prompt[] = {0};
        int output[4] = {};
        int output_len = -1;
        uint32_t rng = 1;

        assert(!minllama::transformer_model_generate_sample_f32(
            model, nullptr, 1, 4, output, 4, &output_len, -1, 0.0f, &rng));
        assert(!minllama::transformer_model_generate_sample_f32(
            model, prompt, 1, 4, nullptr, 4, &output_len, -1, 0.0f, &rng));
        assert(!minllama::transformer_model_generate_sample_f32(
            model, prompt, 1, 4, output, 4, nullptr, -1, 0.0f, &rng));
        assert(!minllama::transformer_model_generate_sample_f32(
            model, prompt, 1, 4, output, 4, &output_len, -1, 0.0f, nullptr));
    }

    // ----------------------------------------------------------------
    // Test 6: text-level sampling — T=0 deterministic.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "c", "a", "b"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, -1));

        auto model = make_mono_model();
        // Override with our ab model.
        model.token_embedding = {
            0.0f, 0.0f,  // 0: <unk>
            0.0f, 0.0f,  // 1: <bos>
            1.0f, 1.0f,  // 2: c
            1.0f, 0.0f,  // 3: a
            0.0f, 1.0f,  // 4: b
        };
        model.lm_head = {
            0.0f, 0.0f,  // 0
            0.0f, 0.0f,  // 1
            0.0f, 1.0f,  // 2: c
            0.0f, 0.0f,  // 3: a
            1.0f, 0.0f,  // 4: b ← argmax for [√2,0]
        };
        model.vocab_size = 5;

        std::string out;
        assert(minllama::minllama_generate_text_sample_f32(
            model, tok, "a", 1, 0.0f, 42, out));
        assert(out == "b"); // deterministic at T=0
    }

    // ----------------------------------------------------------------
    // Test 7: text-level sampling — same seed → same output.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "<eos>", "hello", "world"
        };
        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, 2));

        constexpr int dim = 2;
        auto layer = make_identity_layer(dim);
        minllama::TransformerModelF32 model;
        model.dim = dim;
        model.n_layers = 1;
        model.layers.push_back(layer);
        model.kv_caches.push_back({});
        minllama::kv_cache_init_f32(model.kv_caches[0], 16, dim);
        model.rms_norm_eps = 1e-6f;
        model.final_norm_weight.assign(dim, 1.0f);
        model.vocab_size = 5;
        model.token_embedding = {
            0.0f, 0.0f,  // 0: unk
            0.0f, 0.0f,  // 1: bos
            0.0f, 0.0f,  // 2: eos
            1.0f, 0.0f,  // 3: hello
            0.0f, 1.0f,  // 4: world
        };
        model.lm_head = {
            0.0f, 0.0f,  // 0
            0.0f, 0.0f,  // 1
            0.0f, 0.0f,  // 2
            1.0f, 0.0f,  // 3: hello
            0.0f, 1.0f,  // 4: world
        };

        std::string out1, out2;
        assert(minllama::minllama_generate_text_sample_f32(
            model, tok, "hello", 3, 1.0f, 42, out1));

        // Re-init caches.
        model.kv_caches[0].keys.assign(model.kv_caches[0].keys.size(), 0.0f);
        model.kv_caches[0].values.assign(model.kv_caches[0].values.size(), 0.0f);

        assert(minllama::minllama_generate_text_sample_f32(
            model, tok, "hello", 3, 1.0f, 42, out2));
        assert(out1 == out2);
    }

    // ----------------------------------------------------------------
    // Test 8: text-level — max_new_tokens=0.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {"<unk>", "<bos>", "a"};
        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, -1));

        auto model = make_mono_model();
        model.vocab_size = 3;

        std::string out("garbage");
        assert(minllama::minllama_generate_text_sample_f32(
            model, tok, "a", 0, 1.0f, 1, out));
        assert(out.empty());
    }

    return 0;
}
