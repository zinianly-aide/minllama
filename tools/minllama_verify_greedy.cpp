// minllama_verify_greedy.cpp
// Verify that generate_greedy's internal logits match dump logits.
// Usage: minllama_verify_greedy --model <gguf_path> [--prompt-tokens 28120,905]
#include "minllama.h"
#include "minllama_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void clear_kv_caches(minllama::TransformerModelF32 &model) {
    for (auto &c : model.kv_caches) {
        std::fill(c.keys.begin(), c.keys.end(), 0.0f);
        std::fill(c.values.begin(), c.values.end(), 0.0f);
    }
}

static void print_top_k(const float *logits, int vocab_size, int k, const char *label) {
    std::vector<std::pair<float, int>> sorted;
    sorted.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i)
        sorted.emplace_back(logits[i], i);
    int nk = std::min(k, vocab_size);
    std::partial_sort(sorted.begin(), sorted.begin() + nk, sorted.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    std::printf("%s top-%d:\n", label, nk);
    for (int i = 0; i < nk; ++i) {
        std::printf("  #%d: id=%5d  logit=%.4f\n", i+1, sorted[i].second, sorted[i].first);
    }
    std::printf("  argmax: id=%d  logit=%.4f\n", sorted[0].second, sorted[0].first);
}

int main(int argc, char **argv) {
    std::string model_path = "../models/SmolLM-135M.Q4_0.gguf";
    std::string prompt_tokens_str = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--prompt-tokens" && i + 1 < argc) prompt_tokens_str = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: minllama_verify_greedy --model <path> [--prompt-tokens id,id,...]\n");
            return 0;
        }
    }

    // Parse prompt tokens
    std::vector<int> prompt_tokens;
    if (!prompt_tokens_str.empty()) {
        const char *p = prompt_tokens_str.c_str();
        while (*p) {
            prompt_tokens.push_back(std::atoi(p));
            while (*p && *p != ',') ++p;
            if (*p == ',') ++p;
        }
    }
    if (prompt_tokens.empty()) {
        prompt_tokens = {28120, 905};  // "hello world"
    }

    ml_model *ml = ml_model_load(model_path.c_str());
    if (!ml) { std::fprintf(stderr, "FAIL: load model\n"); return 1; }

    minllama::TransformerModelF32 model;
    std::string error;
    if (!minllama::load_transformer_model_f32_from_tensors(*ml, model, &error)) {
        std::fprintf(stderr, "FAIL: load transformer: %s\n", error.c_str());
        ml_model_free(ml); return 1;
    }

    minllama::SimpleTokenizer tokenizer;
    if (!minllama::load_simple_tokenizer_from_gguf(*ml, tokenizer, &error)) {
        std::fprintf(stderr, "FAIL: load tokenizer: %s\n", error.c_str());
        ml_model_free(ml); return 1;
    }
    ml_model_free(ml);

    std::printf("model: dim=%d n_layers=%d vocab=%d\n", model.dim, model.n_layers, model.vocab_size);
    std::printf("prompt tokens [prompt_len=%zu]:", prompt_tokens.size());
    for (int t : prompt_tokens) std::printf(" %d", t);
    std::printf("\n");

    const int prompt_len = (int)prompt_tokens.size();
    const int dim = model.dim;
    const int vocab = model.vocab_size;

    // ======================================================================
    // PATH A: Prefill via generate_greedy (full pipeline).
    // This resets KV cache and runs all prompt tokens.
    // ======================================================================
    std::printf("\n=== PATH A: generate_greedy (reset KV cache) ===\n");
    clear_kv_caches(model);
    int output_tokens[32];
    int output_len = 0;
    minllama::transformer_model_generate_greedy_f32(
        model, prompt_tokens.data(), prompt_len, 2, output_tokens, 32, &output_len, -1);
    std::printf("generated token[0]: id=%d\n", output_tokens[0]);
    std::printf("generated token[1]: id=%d\n", output_tokens[1]);

    // ======================================================================
    // PATH B: Same as generate_greedy's prefill but manual.
    // For prompt_len=2:
    //   i=0: token[0]=28120, position=0 → embed then greedystep (fills KV[0])
    //   i=1: token[1]=905,   position=1 → embed then greedystep (fills KV[1])
    // The last step's logits are what generate_greedy uses.
    // ======================================================================
    std::printf("\n=== PATH B: manual prefill (same as generate_greedy) ===\n");
    clear_kv_caches(model);

    int last_tid = -1;
    for (int i = 0; i < prompt_len; ++i) {
        std::vector<float> emb(dim);
        minllama::token_embedding_lookup_f32(model, prompt_tokens[i], emb.data());
        int tid = -1;
        minllama::transformer_model_greedy_step_f32(model, emb.data(), i, &tid);
        std::printf("  prefill step %d: token=%d position=%d → next_tid=%d\n", i, prompt_tokens[i], i, tid);
        if (i == prompt_len - 1) last_tid = tid;
    }
    std::printf("next_token (from last prefill step): id=%d\n", last_tid);
    std::printf("generate_greedy first gen token: id=%d\n", output_tokens[0]);
    std::printf("%s\n", last_tid == output_tokens[0] ? "MATCH ✓" : "MISMATCH ✗");

    if (last_tid != output_tokens[0]) {
        std::printf("\n=== MISMATCH INVESTIGATION ===\n");
        // The issue: generate_greedy captures tid from the loop, but does it use
        // the correct tid? Let me check what happens on the first generate step.
        //
        // After prefill (prompt_len=2, KV cache has 2 entries):
        // cur_token = next_token (the tid from the last prefill step)
        // generate step i=0: position = prompt_len + i = 2
        // token_embedding_lookup for cur_token, forward at position 2

        // Let's do the first generate step manually:
        std::vector<float> emb_next(dim);
        minllama::token_embedding_lookup_f32(model, last_tid, emb_next.data());
        int first_gen_tid = -1;
        minllama::transformer_model_greedy_step_f32(model, emb_next.data(), prompt_len, &first_gen_tid);
        std::printf("manual first gen step (token=%d, position=%d): next_tid=%d\n",
                    last_tid, prompt_len, first_gen_tid);
        std::printf("Does this match generate_greedy[1]? output_tokens[1]=%d %s\n",
                    output_tokens[1],
                    first_gen_tid == output_tokens[1] ? "MATCH ✓" : "MISMATCH ✗");
    }

    // ======================================================================
    // PATH C: Also check: if we redo the prefill but also capture the *logits*
    // from the last prefill step, does that logits argmax match?
    // ======================================================================
    std::printf("\n=== PATH C: Verify logits during prefill ===\n");
    clear_kv_caches(model);

    std::vector<float> prefill_logits;
    for (int i = 0; i < prompt_len; ++i) {
        std::vector<float> emb(dim);
        minllama::token_embedding_lookup_f32(model, prompt_tokens[i], emb.data());
        if (i == prompt_len - 1) {
            // Capture the logits from the last step
            prefill_logits.resize(vocab);
            minllama::transformer_model_logits_f32(model, emb.data(), i, prefill_logits.data());
        } else {
            int tid = -1;
            minllama::transformer_model_greedy_step_f32(model, emb.data(), i, &tid);
        }
    }
    print_top_k(prefill_logits.data(), vocab, 5, "prefill last step logits (greedy_step)");

    // ======================================================================
    // PATH D: Compare with the logits from a BOS-only forward (old compare_logits way)
    // ======================================================================
    std::printf("\n=== PATH D: BOS-only forward (position=0, like old compare_logits) ===\n");
    clear_kv_caches(model);
    int bos_id = tokenizer.bos_token_id;
    if (bos_id < 0) bos_id = 0;
    std::vector<float> emb_bos(dim);
    minllama::token_embedding_lookup_f32(model, bos_id, emb_bos.data());
    std::vector<float> logits_bos(vocab);
    minllama::transformer_model_logits_f32(model, emb_bos.data(), 0, logits_bos.data());
    print_top_k(logits_bos.data(), vocab, 5, "BOS-only logits (pos=0)");

    std::printf("\n=== FINAL: generate_greedy first token = %d ===\n", output_tokens[0]);

    return 0;
}
