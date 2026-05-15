#include "minllama.h"
#include "minllama_internal.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct BenchOpts {
    std::string model_path;
    std::string prompt;
    int max_new_tokens = 128;
    int n_threads = 1;
    bool help = false;
    bool q8_lm_head = false;
};

bool parse_bench_args(int argc, const char **argv, BenchOpts &opts) {
    if (!argv) return false;
    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) return false;
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") { opts.help = true; }
        else if (arg == "--model") { if (++i >= argc) return false; opts.model_path = argv[i]; }
        else if (arg == "--prompt") { if (++i >= argc) return false; opts.prompt = argv[i]; }
        else if (arg == "--max-new-tokens") {
            if (++i >= argc) return false;
            char *end = nullptr;
            long n = std::strtol(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || n < 0 || n > 2147483647) return false;
            opts.max_new_tokens = static_cast<int>(n);
        } else if (arg == "--threads") {
            if (++i >= argc) return false;
            char *end = nullptr;
            long n = std::strtol(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || (n != 1 && n != 2)) {
                std::fprintf(stderr, "Error: --threads %ld not supported. "
                    "Currently only --threads 1 or 2 supported.\n", n);
                return false;
            }
            opts.n_threads = static_cast<int>(n);
        } else if (arg == "--q8-lm-head") {
            opts.q8_lm_head = true;
        } else { return false; }
    }
    return true;
}

} // namespace

int main(int argc, const char **argv) {
    BenchOpts opts;
    if (!parse_bench_args(argc, argv, opts)) {
        std::fprintf(stderr, "Usage: minllama_bench --model <path> --prompt <text> [--max-new-tokens <n>] [--threads <1|2>] [--q8-lm-head]\n");
        return 1;
    }
    if (opts.help) { return 0; }

    // ---- Load model ----
    ml_model *ml = ml_model_load(opts.model_path.c_str());
    if (!ml) { std::fprintf(stderr, "Failed to load model\n"); return 1; }
    minllama::TransformerModelF32 model;
    std::string error;
    if (!minllama::load_transformer_model_f32_from_tensors(*ml, model, &error)) {
        std::fprintf(stderr, "Load error: %s\n", error.c_str());
        ml_model_free(ml); return 1;
    }
    model.n_threads = opts.n_threads;
    model.q8_lm_head_enabled = opts.q8_lm_head;

    // ---- Tokenize ----
    minllama::BpeTokenizer bpe_tok;
    if (!minllama::bpe_tokenizer_load(opts.model_path, bpe_tok, &error)) {
        std::fprintf(stderr, "Tokenizer error: %s\n", error.c_str());
        ml_model_free(ml); return 1;
    }
    std::vector<int> prompt_ids;
    if (!minllama::bpe_encode(bpe_tok, opts.prompt, prompt_ids) || prompt_ids.empty()) {
        std::fprintf(stderr, "Tokenize failed\n"); ml_model_free(ml); return 1;
    }
    int eos_id = bpe_tok.eos_token_id >= 0 ? bpe_tok.eos_token_id : -1;
    ml_model_free(ml);

    const int prompt_len = static_cast<int>(prompt_ids.size());

    // Log how many parallel_for calls the generate loop makes
    // 1. prompt prefill: prompt_len tokens → each triggers greedy_token_step
    //    which does: embedding_lookup + model_logits:
    //      - model_decode (n_layers=5 layers × ~10 matvecs each = 50 parallel_for)
    //      - rmsnorm (sequential)
    //      - lm_head matvec (parallel_for)
    //    So per token: ~51 parallel_for calls
    // 2. decode: max_new_tokens tokens × ~51 = 51 * 128 = ~6528 parallel_for calls
    const int total_iters = (prompt_len + opts.max_new_tokens) * 5 * 11; // rough upper bound
    std::fprintf(stderr, "[debug] threads=%d prompt_len=%d max_new=%d est_parallel_for_calls=%d\n",
                 opts.n_threads, prompt_len, opts.max_new_tokens, total_iters);
    std::fflush(stderr);

    auto t5 = Clock::now();
    std::vector<int> output_ids(opts.max_new_tokens);
    int output_len = 0;
    bool ok = minllama::transformer_model_generate_greedy_f32(
        model, prompt_ids.data(), prompt_len,
        opts.max_new_tokens, output_ids.data(),
        static_cast<int>(output_ids.size()),
        &output_len, eos_id);

    if (!ok) {
        std::fprintf(stderr, "Generation FAILED\n");
        return 1;
    }
    auto t6 = Clock::now();

    double decode_ms = Ms(t6 - t5).count();
    int decode_tokens = output_len;
    double decode_tok_s = (decode_tokens > 0 && decode_ms > 0)
        ? decode_tokens / (decode_ms / 1000.0) : 0.0;

    std::printf("decode_ms=%.1f\n", decode_ms);
    std::printf("decode_tok_s=%.2f\n", decode_tok_s);
    std::printf("threads=%d\n", opts.n_threads);
    std::printf("generated=%d\n", decode_tokens);
    std::printf("first_tokens=");
    for (int i = 0; i < std::min(5, output_len); ++i) {
        if (i > 0) std::printf(",");
        std::printf("%d", output_ids[i]);
    }
    std::printf("\n");
    return 0;
}
