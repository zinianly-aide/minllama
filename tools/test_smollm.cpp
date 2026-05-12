// Quick test: load SmolLM-135M GGUF and run inference.
// Build: cd build && cmake .. && make test_smollm
// Run: ./test_smollm

#include "minllama.h"
#include "minllama_internal.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    const char *model_path = "../models/SmolLM-135M.Q4_0.gguf";
    if (argc > 1) model_path = argv[1];

    std::printf("=== minllama: SmolLM-135M inference test ===\n\n");

    // 1. Load GGUF
    std::printf("[1/4] Loading GGUF: %s\n", model_path);
    auto t0 = std::chrono::steady_clock::now();

    ml_model *ml = ml_model_load(model_path);
    if (!ml) {
        std::fprintf(stderr, "FAIL: ml_model_load returned null\n");
        return 1;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms_load = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::printf("  GGUF loaded in %lld ms\n", (long long)ms_load);

    const auto &cfg = ml->config;
    std::printf("  Architecture: llama\n");
    std::printf("  dim=%d  n_layers=%d  vocab=%d  ctx=%d\n",
                cfg.n_embd, cfg.n_layer, cfg.n_vocab, cfg.n_ctx_train);
    std::printf("  n_head=%d  n_head_kv=%d  rope_theta=%.1f\n",
                cfg.n_head, cfg.n_head_kv, cfg.rope_theta);
    std::printf("  n_tensors=%zu\n", ml->tensor_index.size());

    // 2. Build TransformerModelF32
    std::printf("\n[2/4] Building TransformerModelF32...\n");
    minllama::TransformerModelF32 model;
    std::string error;
    if (!minllama::load_transformer_model_f32_from_tensors(*ml, model, &error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        ml_model_free(ml);
        return 1;
    }

    auto t2 = std::chrono::steady_clock::now();
    auto ms_build = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::printf("  Model built in %lld ms\n", (long long)ms_build);
    std::printf("  n_layers=%d  dim=%d  vocab=%d  hidden_dim=%d\n",
                model.n_layers, model.dim, model.vocab_size,
                model.layers.empty() ? 0 : model.layers[0].hidden_dim);

    // 3. Load tokenizer
    std::printf("\n[3/4] Loading tokenizer...\n");
    minllama::SimpleTokenizer tokenizer;
    if (!minllama::load_simple_tokenizer_from_gguf(*ml, tokenizer, &error)) {
        std::fprintf(stderr, "FAIL loading tokenizer: %s\n", error.c_str());
        ml_model_free(ml);
        return 1;
    }
    std::printf("  vocab_size=%zu  bos=%d  eos=%d  unk=%d\n",
                tokenizer.id_to_token.size(),
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.unk_token_id);

    // 4. Test inference with raw BOS token
    std::printf("\n[4/4] Running inference (prompt=[BOS])...\n");

    std::vector<int> prompt_ids;
    if (tokenizer.bos_token_id >= 0) {
        prompt_ids.push_back(tokenizer.bos_token_id);
    }

    int max_tokens = 20;
    std::string output;
    int generated = 0;

    // Reset KV caches
    int ctx_len = cfg.n_ctx_train > 0 ? cfg.n_ctx_train : 2048;
    for (auto &kv : model.kv_caches) {
        minllama::kv_cache_init_f32(kv, ctx_len, model.dim);
    }

    // --- Prefill ---
    std::vector<float> x(model.dim, 0.0f);
    for (size_t pi = 0; pi < prompt_ids.size(); ++pi) {
        int tid = prompt_ids[pi];
        std::vector<float> emb(model.dim, 0.0f);
        if (!minllama::token_embedding_lookup_f32(model, tid, emb.data())) {
            std::fprintf(stderr, "FAIL: token_embedding_lookup for id %d\n", tid);
            ml_model_free(ml);
            return 1;
        }
        for (int d = 0; d < model.dim; ++d) x[d] = emb[d];

        std::vector<float> out_x(model.dim, 0.0f);
        if (!minllama::transformer_model_decode_f32(model, x.data(), (int)pi, out_x.data())) {
            std::fprintf(stderr, "FAIL: decode at position %zu\n", pi);
            ml_model_free(ml);
            return 1;
        }
        x = out_x;
    }

    // --- Decode loop ---
    int pos = (int)prompt_ids.size();
    for (int step = 0; step < max_tokens; ++step) {
        int next_token = -1;
        uint32_t rng_state = 42;
        if (!minllama::transformer_model_sample_step_f32(
                model, x.data(), pos - 1, 0.8f, &rng_state, &next_token)) {
            break;
        }
        if (next_token < 0 || next_token >= model.vocab_size) break;
        if (tokenizer.eos_token_id >= 0 && next_token == tokenizer.eos_token_id) break;

        // Detokenize (raw token string)
        if (next_token >= 0 && next_token < (int)tokenizer.id_to_token.size()) {
            const std::string &tok_str = tokenizer.id_to_token[next_token];
            // Replace "Ġ" (U+0120) prefix with space for readability
            if (tok_str.size() >= 2 &&
                (unsigned char)tok_str[0] == 0xC4 &&
                (unsigned char)tok_str[1] == 0xA0) {
                output += " " + tok_str.substr(2);
            } else {
                output += tok_str;
            }
        }
        ++generated;

        // Embed next token and decode
        std::vector<float> next_emb(model.dim, 0.0f);
        if (!minllama::token_embedding_lookup_f32(model, next_token, next_emb.data())) break;

        std::vector<float> out_x(model.dim, 0.0f);
        if (!minllama::transformer_model_decode_f32(model, next_emb.data(), pos, out_x.data())) break;
        x = out_x;
        ++pos;
    }

    auto t3 = std::chrono::steady_clock::now();
    auto ms_infer = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::printf("\n=== Results ===\n");
    std::printf("Generated %d tokens in %lld ms (%.1f tok/s)\n",
                generated, (long long)ms_infer,
                generated > 0 ? (generated * 1000.0 / ms_infer) : 0.0);
    std::printf("Output: [%s]\n", output.c_str());

    std::printf("\n=== Summary ===\n");
    std::printf("  Load:    %lld ms\n", (long long)ms_load);
    std::printf("  Build:   %lld ms\n", (long long)ms_build);
    std::printf("  Infer:   %lld ms\n", (long long)ms_infer);

    ml_model_free(ml);
    return 0;
}
