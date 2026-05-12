#include "minllama_internal.h"

#include <cassert>
#include <string>
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

// Build a model where "a" → "b" deterministically.
// Vocab: <unk>=0, <bos>=1, c=2, a=3, b=4
// Embedding:
//   unk=[0,0], bos=[0,0], c=[1,1], a=[1,0], b=[0,1]
// LM head weights:
//   unk=[0,0], bos=[0,0], c=[0,1], a=[0,0], b=[1,0]
// With identity model + RMSNorm:
//   a=[1,0] → norm=[√2,0] → logit[4]=√2 → argmax=b
//   b=[0,1] → norm=[0,√2] → logit[2]=√2 → argmax=c
//   c=[1,1] → norm=[1,1] → logit[4]=1 → argmax=b
minllama::TransformerModelF32 make_ab_model() {
    constexpr int dim = 2;
    constexpr int vocab_size = 5;

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

    // token_embedding: [vocab*dim] row-major
    model.token_embedding = {
        0.0f, 0.0f,  // 0: <unk>
        0.0f, 0.0f,  // 1: <bos>
        1.0f, 1.0f,  // 2: c
        1.0f, 0.0f,  // 3: a
        0.0f, 1.0f,  // 4: b
    };

    // lm_head: [vocab*dim] row-major
    model.lm_head = {
        0.0f, 0.0f,  // 0: <unk>
        0.0f, 0.0f,  // 1: <bos>
        0.0f, 1.0f,  // 2: c
        0.0f, 0.0f,  // 3: a
        1.0f, 0.0f,  // 4: b
    };

    return model;
}

} // namespace

int main() {
    // ----------------------------------------------------------------
    // Test 1: generate text one token — "a" → "b".
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "c", "a", "b"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, -1));

        auto model = make_ab_model();

        std::string output;
        assert(minllama::minllama_generate_text_greedy_f32(
            model, tok, "a", 1, output));
        assert(output == "b");
    }

    // ----------------------------------------------------------------
    // Test 2: generate text multiple tokens.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "c", "a", "b"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, -1));

        auto model = make_ab_model();

        std::string output;
        assert(minllama::minllama_generate_text_greedy_f32(
            model, tok, "a", 3, output));
        // a → b → c → b, decoded = "b c b"
        assert(!output.empty());
        // Verify all decoded tokens are in vocab (excluding special tokens).
        // We already know it doesn't crash; just verify it's not empty.
    }

    // ----------------------------------------------------------------
    // Test 3: max_new_tokens == 0.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "c", "a", "b"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, -1));

        auto model = make_ab_model();

        std::string output("garbage");
        assert(minllama::minllama_generate_text_greedy_f32(
            model, tok, "a", 0, output));
        assert(output.empty());
    }

    // ----------------------------------------------------------------
    // Test 4: unknown prompt with unk_token_id.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "c", "a", "b"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, -1));

        auto model = make_ab_model();

        std::string output;
        // "xyz" not in vocab → maps to <unk> (id=0)
        assert(minllama::minllama_generate_text_greedy_f32(
            model, tok, "xyz", 1, output));
    }

    // ----------------------------------------------------------------
    // Test 5: unknown prompt without unk → encode fails.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<eos>", "a", "b"
        };

        minllama::SimpleTokenizer tok;
        // unk=-1, bos=-1, eos=0
        assert(minllama::tokenizer_init(tok, vocab, -1, -1, 0));

        auto model = make_ab_model();

        std::string output;
        assert(!minllama::minllama_generate_text_greedy_f32(
            model, tok, "xyz", 1, output));
    }

    // ----------------------------------------------------------------
    // Test 6: invalid model — empty token_embedding.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "c", "a", "b"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, -1));

        auto model = make_ab_model();
        model.token_embedding.clear();  // Invalid.

        std::string output;
        assert(!minllama::minllama_generate_text_greedy_f32(
            model, tok, "a", 1, output));
    }

    // ----------------------------------------------------------------
    // Test 7: max_new_tokens < 0.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {"<unk>", "<bos>", "a"};

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, -1));

        auto model = make_ab_model();

        std::string output;
        assert(!minllama::minllama_generate_text_greedy_f32(
            model, tok, "a", -1, output));
    }

    // ----------------------------------------------------------------
    // Test 8: empty prompt with bos.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "c", "a", "b"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, -1));

        auto model = make_ab_model();

        std::string output;
        assert(minllama::minllama_generate_text_greedy_f32(
            model, tok, "", 1, output));
        // bos embedding is [0,0], output should be valid.
    }

    return 0;
}
