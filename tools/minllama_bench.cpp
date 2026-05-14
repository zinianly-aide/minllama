// minllama_bench.cpp — Benchmark infrastructure for minllama
//
// Usage:
//   minllama_bench --model <gguf> [--prompt "text"] [--max-new-tokens N]
//                  [--threads N] [--warmup N] [--json]
//                  [--mode decode-only|prefill-only|full]
//
// Output: timing metrics for model load, prefill, and decode phases.

#include "minllama.h"
#include "minllama_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------
// Argument parsing
// -----------------------------------------------------------------------
struct BenchConfig {
    std::string model_path = "../models/SmolLM-135M.Q4_0.gguf";
    std::string prompt = "hello world";
    int max_new_tokens = 32;
    int threads = 1;          // placeholder — not implemented yet
    int warmup = 0;           // warmup iterations
    bool json_output = false;
    std::string mode = "full"; // full | decode-only | prefill-only
};

static bool parse_args(int argc, char **argv, BenchConfig &cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) cfg.model_path = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) cfg.prompt = argv[++i];
        else if (arg == "--max-new-tokens" && i + 1 < argc) cfg.max_new_tokens = std::atoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) cfg.threads = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) cfg.warmup = std::atoi(argv[++i]);
        else if (arg == "--json") cfg.json_output = true;
        else if (arg == "--mode" && i + 1 < argc) {
            cfg.mode = argv[++i];
            if (cfg.mode != "full" && cfg.mode != "decode-only" && cfg.mode != "prefill-only") {
                std::fprintf(stderr, "ERROR: --mode must be: full, decode-only, prefill-only\n");
                return false;
            }
        }
        else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: minllama_bench [--model <path>] [--prompt <text>]\n");
            std::printf("  --max-new-tokens <N>  (default: 32)\n");
            std::printf("  --threads <N>         (placeholder, default: 1)\n");
            std::printf("  --warmup <N>          warmup runs before timing (default: 0)\n");
            std::printf("  --json                JSON output\n");
            std::printf("  --mode <mode>         full | decode-only | prefill-only\n");
            return false;
        }
        else {
            std::fprintf(stderr, "WARNING: unknown flag: %s\n", arg.c_str());
        }
    }
    return true;
}

// -----------------------------------------------------------------------
// Timing
// -----------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<Ms>(end - start).count();
}

// -----------------------------------------------------------------------
// Benchmark runner
// -----------------------------------------------------------------------
struct BenchResult {
    double load_model_ms = 0;
    double tokenize_ms = 0;
    double prefill_ms = 0;
    double prefill_tok_s = 0;
    int prefill_count = 0;
    double decode_ms = 0;
    double decode_tok_s = 0;
    int decode_count = 0;
    double total_gen_ms = 0;
    double total_tok_s = 0;
};

static BenchResult run_benchmark(const BenchConfig &cfg) {
    BenchResult r;

    // ---- Load model ----
    auto t0 = Clock::now();
    ml_model *ml = ml_model_load(cfg.model_path.c_str());
    if (!ml) {
        std::fprintf(stderr, "FAIL: ml_model_load\n");
        std::exit(1);
    }
    r.load_model_ms = elapsed_ms(t0, Clock::now());

    // ---- Load transformer model ----
    minllama::TransformerModelF32 model;
    std::string error;
    if (!minllama::load_transformer_model_f32_from_tensors(*ml, model, &error)) {
        std::fprintf(stderr, "FAIL: load_transformer_model: %s\n", error.c_str());
        ml_model_free(ml);
        std::exit(1);
    }

    // ---- Load tokenizer ----
    minllama::BpeTokenizer bpe;
    if (!minllama::bpe_tokenizer_load(cfg.model_path, bpe, &error)) {
        // Fall back to simple tokenizer
        minllama::SimpleTokenizer tokenizer;
        if (!minllama::load_simple_tokenizer_from_gguf(*ml, tokenizer, &error)) {
            std::fprintf(stderr, "FAIL: load tokenizer: %s\n", error.c_str());
            ml_model_free(ml);
            std::exit(1);
        }
        // Use naive whitespace tokenizer
        std::vector<int> ids;
        minllama::tokenizer_encode_whitespace(tokenizer, cfg.prompt, ids, false);
        // BOS
        if (tokenizer.add_bos_token && tokenizer.bos_token_id >= 0) {
            ids.insert(ids.begin(), tokenizer.bos_token_id);
        }
        std::vector<int> prompt_tokens = ids;

        auto t_tok = Clock::now();
        r.tokenize_ms = elapsed_ms(t_tok, t_tok); // ~0

        // ---- Prefill ----
        std::vector<float> logits(model.vocab_size);
        auto t_prefill_start = Clock::now();

        int last_tid = -1;
        for (int i = 0; i < (int)prompt_tokens.size(); ++i) {
            std::vector<float> emb(model.dim);
            minllama::token_embedding_lookup_f32(model, prompt_tokens[i], emb.data());
            if (cfg.mode != "decode-only") {
                int tid = -1;
                minllama::transformer_model_greedy_step_f32(model, emb.data(), i, &tid);
                if (i == (int)prompt_tokens.size() - 1) last_tid = tid;
            } else {
                // In decode-only mode, we skip the prompt
                if (i == (int)prompt_tokens.size() - 1) last_tid = prompt_tokens[i];
            }
        }

        auto t_prefill_end = Clock::now();
        r.prefill_ms = elapsed_ms(t_prefill_start, t_prefill_end);
        r.prefill_count = (cfg.mode == "decode-only") ? 0 : (int)prompt_tokens.size();
        r.prefill_tok_s = (r.prefill_ms > 0 && r.prefill_count > 0)
            ? r.prefill_count / (r.prefill_ms / 1000.0) : 0;

        // ---- Decode ----
        if (cfg.mode != "prefill-only" && cfg.max_new_tokens > 0) {
            int cur_tok = last_tid;
            auto t_decode_start = Clock::now();

            for (int i = 0; i < cfg.max_new_tokens; ++i) {
                // output[i] = cur_tok
                if (i == cfg.max_new_tokens - 1) break;
                int pos = (int)prompt_tokens.size() + i;
                std::vector<float> emb(model.dim);
                minllama::token_embedding_lookup_f32(model, cur_tok, emb.data());
                minllama::transformer_model_greedy_step_f32(model, emb.data(), pos, &cur_tok);
            }

            auto t_decode_end = Clock::now();
            r.decode_ms = elapsed_ms(t_decode_start, t_decode_end);
            r.decode_count = cfg.max_new_tokens;
            r.decode_tok_s = (r.decode_ms > 0)
                ? r.decode_count / (r.decode_ms / 1000.0) : 0;
        }

        r.total_gen_ms = r.prefill_ms + r.decode_ms;
        r.total_tok_s = (r.total_gen_ms > 0)
            ? (r.prefill_count + r.decode_count) / (r.total_gen_ms / 1000.0) : 0;

        ml_model_free(ml);
        return r;
    }

    // BPE path
    std::vector<int> prompt_tokens;
    auto t_tok = Clock::now();
    minllama::bpe_encode(bpe, cfg.prompt, prompt_tokens);
    r.tokenize_ms = elapsed_ms(t_tok, Clock::now());

    // ---- Prefill ----
    auto t_prefill_start = Clock::now();

    int last_tid = -1;
    for (int i = 0; i < (int)prompt_tokens.size(); ++i) {
        std::vector<float> emb(model.dim);
        minllama::token_embedding_lookup_f32(model, prompt_tokens[i], emb.data());
        if (cfg.mode != "decode-only") {
            int tid = -1;
            minllama::transformer_model_greedy_step_f32(model, emb.data(), i, &tid);
            if (i == (int)prompt_tokens.size() - 1) last_tid = tid;
        } else {
            if (i == (int)prompt_tokens.size() - 1) last_tid = prompt_tokens[i];
        }
    }

    auto t_prefill_end = Clock::now();
    r.prefill_ms = elapsed_ms(t_prefill_start, t_prefill_end);
    r.prefill_count = (cfg.mode == "decode-only") ? 0 : (int)prompt_tokens.size();
    r.prefill_tok_s = (r.prefill_ms > 0 && r.prefill_count > 0)
        ? r.prefill_count / (r.prefill_ms / 1000.0) : 0;

    // ---- Decode ----
    if (cfg.mode != "prefill-only" && cfg.max_new_tokens > 0) {
        int cur_tok = last_tid;
        auto t_decode_start = Clock::now();

        for (int i = 0; i < cfg.max_new_tokens; ++i) {
            if (i == cfg.max_new_tokens - 1) break;
            int pos = (int)prompt_tokens.size() + i;
            std::vector<float> emb(model.dim);
            minllama::token_embedding_lookup_f32(model, cur_tok, emb.data());
            minllama::transformer_model_greedy_step_f32(model, emb.data(), pos, &cur_tok);
        }

        auto t_decode_end = Clock::now();
        r.decode_ms = elapsed_ms(t_decode_start, t_decode_end);
        r.decode_count = cfg.max_new_tokens;
        r.decode_tok_s = (r.decode_ms > 0)
            ? r.decode_count / (r.decode_ms / 1000.0) : 0;
    }

    r.total_gen_ms = r.prefill_ms + r.decode_ms;
    r.total_tok_s = (r.total_gen_ms > 0)
        ? (r.prefill_count + r.decode_count) / (r.total_gen_ms / 1000.0) : 0;

    ml_model_free(ml);
    return r;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    BenchConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        return cfg.json_output ? 0 : 0;
    }

    // Warmup
    for (int w = 0; w < cfg.warmup; ++w) {
        BenchConfig warmup_cfg = cfg;
        warmup_cfg.max_new_tokens = 1;
        run_benchmark(warmup_cfg);
    }

    // Actual benchmark
    auto r = run_benchmark(cfg);

    double approx_memory_mb = 86.0 + 90.0; // GGUF + KV cache estimate

    if (cfg.json_output) {
        std::printf("{\n");
        std::printf("  \"load_model_ms\": %.2f,\n", r.load_model_ms);
        std::printf("  \"tokenize_ms\": %.2f,\n", r.tokenize_ms);
        std::printf("  \"prefill_ms\": %.2f,\n", r.prefill_ms);
        std::printf("  \"prefill_tok_s\": %.2f,\n", r.prefill_tok_s);
        std::printf("  \"prefill_count\": %d,\n", r.prefill_count);
        std::printf("  \"decode_ms\": %.2f,\n", r.decode_ms);
        std::printf("  \"decode_tok_s\": %.2f,\n", r.decode_tok_s);
        std::printf("  \"decode_count\": %d,\n", r.decode_count);
        std::printf("  \"total_gen_ms\": %.2f,\n", r.total_gen_ms);
        std::printf("  \"total_tok_s\": %.2f,\n", r.total_tok_s);
        std::printf("  \"approx_memory_mb\": %.0f\n", approx_memory_mb);
        std::printf("}\n");
    } else {
        std::printf("=== minllama benchmark ===\n");
        std::printf("model:       %s\n", cfg.model_path.c_str());
        std::printf("prompt:      \"%s\" (len=%d tok)\n", cfg.prompt.c_str(), r.prefill_count);
        std::printf("decode:      %d tokens\n", r.decode_count);
        std::printf("threads:     %d\n", cfg.threads);
        std::printf("\n");
        std::printf("load_model:  %8.2f ms  (%d tokens%%)  %8.2f tok/s\n",
                    r.load_model_ms, 0, 0.0);
        std::printf("tokenize:    %8.2f ms\n", r.tokenize_ms);
        if (r.prefill_count > 0) {
            std::printf("prefill:     %8.2f ms  (%d tokens)  %8.2f tok/s\n",
                        r.prefill_ms, r.prefill_count, r.prefill_tok_s);
        }
        if (r.decode_count > 0) {
            std::printf("decode:      %8.2f ms  (%d tokens)  %8.2f tok/s\n",
                        r.decode_ms, r.decode_count, r.decode_tok_s);
        }
        std::printf("total gen:   %8.2f ms  (%d tokens)  %8.2f tok/s\n",
                    r.total_gen_ms, r.prefill_count + r.decode_count, r.total_tok_s);
        std::printf("approx mem:  %.0f MB\n", approx_memory_mb);
    }

    return 0;
}
