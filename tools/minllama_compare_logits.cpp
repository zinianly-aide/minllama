// minllama_compare_logits: consolidated diagnostic tool for P0 correctness debugging.
//
// Combines three progressive diagnostic modes into one tool with CLI flags:
//   --mode logits        →  top-k logits (from old logits_dump.cpp)
//   --mode diagnostics   →  weight/layer activation norms (from old logits_dump2.cpp)
//   --mode layer-trace   →  per-component layer 0 trace (from old logits_dump3.cpp)
//   --mode all           →  runs all above (default)
//
// Usage:
//   minllama_compare_logits --model <gguf_path> [--mode <mode>] [--top-k <n>]
//
// Build: cmake -S . -B build && cmake --build build
// Run:   ./build/minllama_compare_logits --model models/SmolLM-135M.Q4_0.gguf

#include "minllama.h"
#include "minllama_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float vec_norm(const float *v, int n) {
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += (double)v[i] * v[i];
    return std::sqrt(sum);
}

static float vec_norm(const std::vector<float> &v) {
    return v.empty() ? 0.0f : vec_norm(v.data(), (int)v.size());
}

static float vec_mean(const float *v, int n) {
    if (n <= 0) return 0.0f;
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += v[i];
    return (float)(sum / n);
}

static float vec_mean(const std::vector<float> &v) {
    return v.empty() ? 0.0f : vec_mean(v.data(), (int)v.size());
}

struct CliConfig {
    std::string model_path = "../models/SmolLM-135M.Q4_0.gguf";
    std::string mode = "all";
    int top_k = 20;
};

static bool parse_args(int argc, char **argv, CliConfig &cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            cfg.mode = argv[++i];
            if (cfg.mode != "logits" && cfg.mode != "diagnostics" &&
                cfg.mode != "layer-trace" && cfg.mode != "all") {
                std::fprintf(stderr, "ERROR: --mode must be: logits, diagnostics, layer-trace, all\n");
                return false;
            }
        } else if (arg == "--top-k" && i + 1 < argc) {
            cfg.top_k = std::atoi(argv[++i]);
            if (cfg.top_k < 1) cfg.top_k = 1;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: minllama_compare_logits [--model <path>] [--mode <mode>] [--top-k <n>]\n");
            std::printf("  --model       GGUF model path (default: ../models/SmolLM-135M.Q4_0.gguf)\n");
            std::printf("  --mode        logits | diagnostics | layer-trace | all (default: all)\n");
            std::printf("  --top-k       number of top logits to print (default: 20)\n");
            return false;
        } else {
            std::fprintf(stderr, "WARNING: unknown flag: %s\n", arg.c_str());
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Mode 1: logits — top-k logits with token strings
// ---------------------------------------------------------------------------

static bool mode_logits(minllama::TransformerModelF32 &model,
                        const minllama::SimpleTokenizer &tokenizer,
                        int top_k) {
    int bos_id = tokenizer.bos_token_id;
    if (bos_id < 0) {
        std::fprintf(stderr, "WARN: no BOS token; using id 0\n");
        bos_id = 0;
    }

    std::vector<float> emb(model.dim);
    if (!minllama::token_embedding_lookup_f32(model, bos_id, emb.data())) {
        std::fprintf(stderr, "FAIL: token_embedding_lookup\n");
        return false;
    }

    std::vector<float> hidden(model.dim);
    if (!minllama::transformer_model_decode_f32(model, emb.data(), 0, hidden.data())) {
        std::fprintf(stderr, "FAIL: model decode at pos 0\n");
        return false;
    }

    std::vector<float> norm_hidden(model.dim);
    if (!minllama::rmsnorm_f32(hidden.data(), model.final_norm_weight.data(),
                                model.dim, model.rms_norm_eps,
                                norm_hidden.data(), model.dim)) {
        std::fprintf(stderr, "FAIL: final rmsnorm\n");
        return false;
    }

    std::vector<float> logits(model.vocab_size);
    if (!minllama::matvec_f32_f32(model.lm_head.data(), model.vocab_size, model.dim,
                                   norm_hidden.data(), model.dim,
                                   logits.data(), model.vocab_size)) {
        std::fprintf(stderr, "FAIL: lm_head matvec\n");
        return false;
    }

    // Sort top-k
    std::vector<std::pair<float, int>> sorted;
    sorted.reserve(model.vocab_size);
    for (int i = 0; i < model.vocab_size; ++i)
        sorted.emplace_back(logits[i], i);
    int k = std::min(top_k, model.vocab_size);
    std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

    std::printf("Top-%d logits (minllama):\n", k);
    for (int i = 0; i < k; ++i) {
        int id = sorted[i].second;
        float val = sorted[i].first;
        std::string tok = (id >= 0 && id < (int)tokenizer.id_to_token.size())
            ? tokenizer.id_to_token[id] : "<OOR>";
        for (auto &c : tok) if (c == '\n' || c == '\r') c = '?';
        std::printf("  #%2d: id=%5d  logit=%12.6f  token=<%s>\n",
                    i + 1, id, val, tok.c_str());
    }

    // Softmax probabilities
    float max_logit = sorted[0].first;
    double sum_exp = 0;
    std::vector<float> probs(model.vocab_size);
    for (int i = 0; i < model.vocab_size; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    std::printf("Top-%d probabilities:\n", k);
    for (int i = 0; i < k; ++i) {
        int id = sorted[i].second;
        float p = probs[id] / (float)sum_exp;
        std::string tok = (id >= 0 && id < (int)tokenizer.id_to_token.size())
            ? tokenizer.id_to_token[id] : "<OOR>";
        for (auto &c : tok) if (c == '\n' || c == '\r') c = '?';
        std::printf("  #%2d: id=%5d  p=%.6f  token=<%s>\n",
                    i + 1, id, p, tok.c_str());
    }

    return true;
}

// ---------------------------------------------------------------------------
// Mode 2: diagnostics — weight norms + per-layer activation tracking
// ---------------------------------------------------------------------------

static bool mode_diagnostics(minllama::TransformerModelF32 &model,
                             const minllama::SimpleTokenizer &tokenizer) {
    int bos_id = tokenizer.bos_token_id;
    if (bos_id < 0) bos_id = 0;

    // Weight norms
    std::printf("\n--- Weight norms (layer 0) ---\n");
    std::printf("token_embd norm=%.4f mean=%.6f\n",
                vec_norm(model.token_embedding), vec_mean(model.token_embedding));
    std::printf("wq norm=%.4f  wk norm=%.4f  wv norm=%.4f  wo norm=%.4f\n",
                vec_norm(model.layers[0].wq), vec_norm(model.layers[0].wk),
                vec_norm(model.layers[0].wv), vec_norm(model.layers[0].wo));
    std::printf("w1 norm=%.4f  w2 norm=%.4f  w3 norm=%.4f\n",
                vec_norm(model.layers[0].w1), vec_norm(model.layers[0].w2),
                vec_norm(model.layers[0].w3));
    std::printf("rms_att norm=%.4f  rms_ffn norm=%.4f  final_norm norm=%.4f  lm_head norm=%.4f\n",
                vec_norm(model.layers[0].rms_att_weight),
                vec_norm(model.layers[0].rms_ffn_weight),
                vec_norm(model.final_norm_weight),
                vec_norm(model.lm_head));

    // BOS embedding
    std::vector<float> emb(model.dim);
    if (!minllama::token_embedding_lookup_f32(model, bos_id, emb.data())) {
        std::fprintf(stderr, "FAIL: embedding lookup\n");
        return false;
    }
    std::printf("\n--- BOS token (id=%d) embedding ---\n", bos_id);
    std::printf("norm=%.4f  mean=%.6f  min=%.6f  max=%.6f\n",
                vec_norm(emb), vec_mean(emb),
                *std::min_element(emb.begin(), emb.end()),
                *std::max_element(emb.begin(), emb.end()));

    // Per-layer activation norms
    std::vector<float> x = emb;
    for (int layer = 0; layer < model.n_layers; ++layer) {
        std::vector<float> out(model.dim);
        if (!minllama::transformer_layer_decode_f32(model.layers[layer], x.data(),
                                                     model.kv_caches[layer], 0,
                                                     out.data())) {
            std::fprintf(stderr, "FAIL: layer %d decode\n", layer);
            return false;
        }
        x = out;
        // Print first 5, last 3, and groups of 5-10 in between
        bool print_layer = false;
        if (layer < 5 || layer >= model.n_layers - 3) {
            print_layer = true;
        } else if (layer < model.n_layers - 3 && layer % 5 == 0) {
            print_layer = true;
        }
        if (print_layer) {
            std::printf("layer %2d: norm=%.2f  mean=%.4f  min=%.4f  max=%.4f\n",
                        layer, vec_norm(x), vec_mean(x),
                        *std::min_element(x.begin(), x.end()),
                        *std::max_element(x.begin(), x.end()));
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Mode 3: layer-trace — per-component breakdown of layer 0
// ---------------------------------------------------------------------------

static bool mode_layer_trace(minllama::TransformerModelF32 &model,
                             const minllama::SimpleTokenizer &tokenizer) {
    using minllama::KvCacheF32;

    int bos_id = tokenizer.bos_token_id;
    if (bos_id < 0) bos_id = 0;

    const auto &L = model.layers[0];
    auto &cache = model.kv_caches[0];
    const int dim = model.dim;
    const int n_heads = L.n_heads;
    const int n_kv_heads = L.n_kv_heads;
    const int head_dim = L.head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    std::printf("\n=== Layer 0 per-component trace ===\n");
    std::printf("dim=%d n_heads=%d n_kv_heads=%d head_dim=%d kv_dim=%d hidden_dim=%d\n",
                dim, n_heads, n_kv_heads, head_dim, kv_dim, L.hidden_dim);

    std::vector<float> x(dim);
    minllama::token_embedding_lookup_f32(model, bos_id, x.data());

    auto trace_vec = [](const char *label, const std::vector<float> &v) {
        std::printf("  %s: norm=%.4f mean=%.6f min=%.6f max=%.6f\n",
                    label, vec_norm(v), vec_mean(v),
                    *std::min_element(v.begin(), v.end()),
                    *std::max_element(v.begin(), v.end()));
    };

    // Input
    std::printf("\n--- Step 0: Input x ---\n");
    trace_vec("x", x);

    // RMSNorm
    std::vector<float> xn(dim);
    minllama::rmsnorm_f32(x.data(), L.rms_att_weight.data(), dim, L.rms_norm_eps, xn.data(), dim);
    std::printf("--- Step 1: RMSNorm(x) ---\n");
    trace_vec("xn", xn);

    // Q projection
    std::vector<float> q(dim);
    minllama::matvec_f32_f32(L.wq.data(), dim, dim, xn.data(), dim, q.data(), dim);
    std::printf("--- Step 2a: Q = Wq * xn ---\n");
    trace_vec("q", q);

    // K projection
    std::vector<float> k(kv_dim);
    minllama::matvec_f32_f32(L.wk.data(), kv_dim, dim, xn.data(), dim, k.data(), kv_dim);
    std::printf("--- Step 2b: K = Wk * xn ---\n");
    trace_vec("k", k);

    // V projection
    std::vector<float> v(kv_dim);
    minllama::matvec_f32_f32(L.wv.data(), kv_dim, dim, xn.data(), dim, v.data(), kv_dim);
    std::printf("--- Step 2c: V = Wv * xn ---\n");
    trace_vec("v", v);

    // Q per-head norms (before RoPE)
    std::printf("--- Step 3a: Q per-head norms (before RoPE) ---\n");
    for (int h = 0; h < n_heads && h < 9; ++h) {
        std::printf("  head %d: norm=%.4f\n", h,
                    vec_norm(q.data() + h * head_dim, head_dim));
    }
    if (n_heads > 9) std::printf("  ... (%d heads total)\n", n_heads);

    // K per-head norms (before RoPE)
    std::printf("--- Step 3b: K per-head norms (before RoPE) ---\n");
    for (int h = 0; h < n_kv_heads && h < 9; ++h) {
        std::printf("  kv_head %d: norm=%.4f\n", h,
                    vec_norm(k.data() + h * head_dim, head_dim));
    }
    if (n_kv_heads > 9) std::printf("  ... (%d kv_heads total)\n", n_kv_heads);

    // Apply RoPE
    for (int h = 0; h < n_heads; ++h)
        minllama::rope_apply_f32(q.data() + h * head_dim, head_dim, 0, L.rope_theta);
    for (int h = 0; h < n_kv_heads; ++h)
        minllama::rope_apply_f32(k.data() + h * head_dim, head_dim, 0, L.rope_theta);

    std::printf("--- Step 3c: Q per-head norms (after RoPE, pos=0) ---\n");
    for (int h = 0; h < n_heads && h < 9; ++h) {
        std::printf("  head %d: norm=%.4f\n", h,
                    vec_norm(q.data() + h * head_dim, head_dim));
    }

    // Write KV to cache
    minllama::kv_cache_write_f32(cache, 0, k.data(), v.data());

    // Attention (position 0, only 1 key/value)
    std::printf("--- Step 4: Attention (pos=0, 1 token in KV) ---\n");
    const int gph = n_heads / n_kv_heads;
    std::vector<float> att_out(dim);
    for (int h = 0; h < n_heads; ++h) {
        const int kv_h = h / gph;
        const float *q_h = q.data() + h * head_dim;
        const float *k_h = cache.keys.data() + kv_h * head_dim;
        const float *v_h = cache.values.data() + kv_h * head_dim;

        float score = 0;
        for (int j = 0; j < head_dim; ++j) score += q_h[j] * k_h[j];
        score /= std::sqrt((float)head_dim);

        float *out_h = att_out.data() + h * head_dim;
        for (int j = 0; j < head_dim; ++j) out_h[j] = v_h[j];

        if (h < 3 || h == n_heads - 1) {
            std::printf("  head %d -> kv_head %d: score=%.4f  val_norm=%.4f\n",
                        h, kv_h, score,
                        vec_norm(v_h, head_dim));
        }
    }
    trace_vec("att_out", att_out);

    // Wo projection
    std::vector<float> projected(dim);
    minllama::matvec_f32_f32(L.wo.data(), dim, dim, att_out.data(), dim, projected.data(), dim);
    std::printf("--- Step 5: projected = Wo * att ---\n");
    trace_vec("projected", projected);

    // Residual: h = x + projected
    std::vector<float> h(dim);
    for (int i = 0; i < dim; ++i) h[i] = x[i] + projected[i];
    std::printf("--- Step 6: h = x + projected ---\n");
    trace_vec("h", h);

    // FFN path
    std::vector<float> ffn_input(dim);
    minllama::rmsnorm_f32(h.data(), L.rms_ffn_weight.data(), dim, L.rms_norm_eps, ffn_input.data(), dim);
    std::printf("--- Step 7: RMSNorm(h) for FFN ---\n");
    trace_vec("ffn_input", ffn_input);

    int hdim = L.hidden_dim;
    if (hdim <= 0) {
        std::printf("--- FFN skipped (hidden_dim=%d) ---\n", hdim);
        h_dim_check:  // can't use goto, skip
        std::vector<float> output(dim);
        for (int i = 0; i < dim; ++i) output[i] = h[i];
        return true;
    }

    std::vector<float> gate(hdim);
    minllama::matvec_f32_f32(L.w1.data(), hdim, dim, ffn_input.data(), dim, gate.data(), hdim);
    std::printf("--- Step 8: gate = W1 * ffn_input ---\n");
    trace_vec("gate", gate);

    std::vector<float> up(hdim);
    minllama::matvec_f32_f32(L.w3.data(), hdim, dim, ffn_input.data(), dim, up.data(), hdim);
    std::printf("--- Step 9: up = W3 * ffn_input ---\n");
    trace_vec("up", up);

    std::vector<float> hidden(hdim);
    minllama::swiglu_f32(gate.data(), up.data(), hdim, hidden.data(), hdim);
    std::printf("--- Step 10: hidden = silu(gate) * up ---\n");
    trace_vec("hidden", hidden);

    std::vector<float> ffn_out(dim);
    minllama::matvec_f32_f32(L.w2.data(), dim, hdim, hidden.data(), hdim, ffn_out.data(), dim);
    std::printf("--- Step 11: ffn_out = W2 * hidden ---\n");
    trace_vec("ffn_out", ffn_out);

    // Final residual
    std::vector<float> output(dim);
    for (int i = 0; i < dim; ++i) output[i] = h[i] + ffn_out[i];
    std::printf("--- Step 12: output = h + ffn_out ---\n");
    trace_vec("output", output);

    // Compare with reference
    std::vector<float> ref_out(dim);
    minllama::transformer_layer_decode_f32(L, x.data(), cache, 0, ref_out.data());
    std::printf("\n--- Reference: transformer_layer_decode_f32 ---\n");
    trace_vec("ref_out", ref_out);

    float max_diff = 0;
    for (int i = 0; i < dim; ++i) {
        float d = std::abs(output[i] - ref_out[i]);
        if (d > max_diff) max_diff = d;
    }
    std::printf("max difference between manual and ref: %.10f\n", max_diff);

    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    CliConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        return 0;  // --help or parse error already printed
    }

    // 1. Load GGUF
    std::printf("=== minllama_compare_logits ===\n");
    std::printf("model: %s\n", cfg.model_path.c_str());

    ml_model *ml = ml_model_load(cfg.model_path.c_str());
    if (!ml) {
        std::fprintf(stderr, "FAIL: ml_model_load(%s)\n", cfg.model_path.c_str());
        return 1;
    }

    const auto &ml_cfg = ml->config;
    std::printf("dim=%d  n_layers=%d  vocab=%d  ctx=%d  n_head=%d  n_head_kv=%d\n",
                ml_cfg.n_embd, ml_cfg.n_layer, ml_cfg.n_vocab, ml_cfg.n_ctx_train,
                ml_cfg.n_head, ml_cfg.n_head_kv);

    // 2. Build model
    minllama::TransformerModelF32 model;
    std::string error;
    if (!minllama::load_transformer_model_f32_from_tensors(*ml, model, &error)) {
        std::fprintf(stderr, "FAIL: load_transformer_model: %s\n", error.c_str());
        ml_model_free(ml);
        return 1;
    }

    // 3. Load tokenizer
    minllama::SimpleTokenizer tokenizer;
    if (!minllama::load_simple_tokenizer_from_gguf(*ml, tokenizer, &error)) {
        std::fprintf(stderr, "FAIL: load tokenizer: %s\n", error.c_str());
        ml_model_free(ml);
        return 1;
    }

    ml_model_free(ml);

    std::printf("BOS id: %d  EOS id: %d  UNK id: %d  vocab_size: %d\n\n",
                tokenizer.bos_token_id, tokenizer.eos_token_id,
                tokenizer.unk_token_id, model.vocab_size);

    // 4. Run requested modes
    bool ok = true;
    if (cfg.mode == "logits" || cfg.mode == "all") {
        std::printf("=== MODE: logits (top-%d) ===\n", cfg.top_k);
        ok = mode_logits(model, tokenizer, cfg.top_k) && ok;
        std::printf("\n");
    }
    if (cfg.mode == "diagnostics" || cfg.mode == "all") {
        std::printf("=== MODE: diagnostics ===\n");
        ok = mode_diagnostics(model, tokenizer) && ok;
        std::printf("\n");
    }
    if (cfg.mode == "layer-trace" || cfg.mode == "all") {
        std::printf("=== MODE: layer-trace ===\n");
        ok = mode_layer_trace(model, tokenizer) && ok;
        std::printf("\n");
    }

    return ok ? 0 : 1;
}
