// minllama_dump_layer: dump per-layer hidden states for comparison with llama.cpp
// Usage: ./minllama_dump_layer --model <gguf> --prompt-tokens <ids> [--max-layers N]
#include "minllama_internal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void dump_vec(const char *s, const float *v, int n, int k) {
    (void)s; (void)v; (void)n; (void)k;
}

int main(int argc, char **argv) {
    std::string model_path;
    std::string prompt_tokens_raw;
    int max_layers = 3;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--prompt-tokens" && i + 1 < argc) prompt_tokens_raw = argv[++i];
        else if (a == "--max-layers" && i + 1 < argc) max_layers = std::atoi(argv[++i]);
        else if (a == "--help") {
            std::printf("Usage: minllama_dump_layer --model <gguf> --prompt-tokens <ids> [--max-layers N]\n");
            return 0;
        }
    }

    if (model_path.empty() || prompt_tokens_raw.empty()) {
        std::fprintf(stderr, "Usage: minllama_dump_layer --model <gguf> --prompt-tokens <ids>\n");
        return 1;
    }

    // Parse token IDs.
    std::vector<int> tokens;
    {
        const char *p = prompt_tokens_raw.c_str();
        while (*p) {
            char *end;
            long v = std::strtol(p, &end, 10);
            if (end == p) break;
            tokens.push_back((int)v);
            p = end;
            if (*p == ',') ++p;
        }
    }
    if (tokens.empty()) { std::fprintf(stderr, "No tokens parsed\n"); return 1; }

    // Load model.
    ml_model *ml = ml_model_load(model_path.c_str());
    if (!ml) { std::fprintf(stderr, "Model load failed\n"); return 1; }

    minllama::TransformerModelF32 model;
    std::string error;
    if (!minllama::load_transformer_model_f32_from_tensors(*ml, model, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 1;
    }

    int dim = model.dim;
    int n_layers = model.n_layers;
    if (n_layers > max_layers) n_layers = max_layers;

    std::fprintf(stderr, "Model: dim=%d n_layers=%d n_heads=%d n_kv_heads=%d\n",
                 dim, model.n_layers,
                 model.layers[0].n_heads, model.layers[0].n_kv_heads);

    // ---- Single token forward: position 0 ----
    int token_id = tokens[0];
    std::fprintf(stderr, "Token: %d, position: 0\n", token_id);

    std::vector<float> x(dim);
    if (!minllama::token_embedding_lookup_f32(model, token_id, x.data())) {
        std::fprintf(stderr, "Embedding lookup failed\n"); return 1;
    }

    std::printf("# minllama dump: token=%d\n", token_id);

    // DUMP macro for first 5 values
#define DUMP_(name, vec, n) do { \
    float amax = 0, nrm = 0; \
    for (int _i = 0; _i < (n); ++_i) { \
        float a = std::fabs((vec)[_i]); if (a > amax) amax = a; \
        nrm += (vec)[_i] * (vec)[_i]; \
    } \
    std::printf("%-30s norm=%.6e absmax=%.6e", (name), (double)std::sqrt(nrm), (double)amax); \
    int _k = 5 < (n) ? 5 : (n); \
    std::printf(" ["); \
    for (int _i = 0; _i < _k; ++_i) std::printf(" %.6f", (double)(vec)[_i]); \
    if (_k < (n)) std::printf(" ..."); \
    std::printf(" ]\n"); \
} while(0)

    DUMP_("embedding", x.data(), dim);

    // Per-layer dump with attention/FFN separation.
    for (int layer = 0; layer < n_layers; ++layer) {
        auto &l = model.layers[layer];
        int dim = l.dim;
        int n_heads = l.n_heads, n_kv_heads = l.n_kv_heads, head_dim = l.head_dim;
        int kv_dim = n_kv_heads * head_dim;

        // --- RMSNorm (pre-attention) ---
        std::vector<float> xn(dim);
        minllama::rmsnorm_f32(x.data(), l.rms_att_weight.data(), dim,
                               l.rms_norm_eps, xn.data(), dim);

        // --- Q/K/V ---
        std::vector<float> q(dim), k(kv_dim), v(kv_dim);
        minllama::matvec_f32_f32(l.wq.data(), dim, dim, xn.data(), dim, q.data(), dim);
        minllama::matvec_f32_f32(l.wk.data(), kv_dim, dim, xn.data(), dim, k.data(), kv_dim);
        minllama::matvec_f32_f32(l.wv.data(), kv_dim, dim, xn.data(), dim, v.data(), kv_dim);

        // --- RoPE ---
        for (int h = 0; h < n_heads; ++h)
            minllama::rope_apply_f32(q.data() + h*head_dim, head_dim, 0, l.rope_theta);
        for (int h = 0; h < n_kv_heads; ++h)
            minllama::rope_apply_f32(k.data() + h*head_dim, head_dim, 0, l.rope_theta);

        // --- Write KV cache ---
        minllama::kv_cache_write_f32(model.kv_caches[layer], 0, k.data(), v.data());

        // --- Attention ---
        std::vector<float> att(dim);
        minllama::attention_decode_gqa_f32(q.data(), model.kv_caches[layer], 0,
                                            n_heads, n_kv_heads, head_dim, att.data());

        // --- Wo projection ---
        std::vector<float> wo_out(dim);
        minllama::matvec_f32_f32(l.wo.data(), dim, dim, att.data(), dim, wo_out.data(), dim);

        // --- Residual after attention ---
        std::vector<float> h(dim);
        for (int i = 0; i < dim; ++i) h[i] = x[i] + wo_out[i];

        // --- FFN ---
        std::vector<float> ffn_out(dim);
        if (l.hidden_dim > 0) {
            int hd = l.hidden_dim;
            std::vector<float> norm_h(dim);
            minllama::rmsnorm_f32(h.data(), l.rms_ffn_weight.data(), dim,
                                   l.rms_norm_eps, norm_h.data(), dim);
            std::vector<float> gate(hd), up(hd);
            minllama::matvec_f32_f32(l.w1.data(), hd, dim, norm_h.data(), dim, gate.data(), hd);
            minllama::matvec_f32_f32(l.w3.data(), hd, dim, norm_h.data(), dim, up.data(), hd);
            std::vector<float> hidden(hd);
            minllama::swiglu_f32(gate.data(), up.data(), hd, hidden.data(), hd);
            minllama::matvec_f32_f32(l.w2.data(), dim, hd, hidden.data(), hd, ffn_out.data(), dim);
            for (int i = 0; i < dim; ++i) ffn_out[i] += h[i];
        } else {
            ffn_out = h;
        }

        char buf[64];
        std::snprintf(buf, sizeof(buf), "L%d_rms_att_norm", layer);
        DUMP_(buf, xn.data(), dim);
        std::snprintf(buf, sizeof(buf), "L%d_q", layer);
        DUMP_(buf, q.data(), dim);
        std::snprintf(buf, sizeof(buf), "L%d_k", layer);
        DUMP_(buf, k.data(), kv_dim);
        std::snprintf(buf, sizeof(buf), "L%d_att_out", layer);
        DUMP_(buf, att.data(), dim);
        std::snprintf(buf, sizeof(buf), "L%d_wo_out", layer);
        DUMP_(buf, wo_out.data(), dim);
        std::snprintf(buf, sizeof(buf), "L%d_post_attn", layer);
        DUMP_(buf, h.data(), dim);
        std::snprintf(buf, sizeof(buf), "L%d_output", layer);
        DUMP_(buf, ffn_out.data(), dim);

        x.swap(ffn_out);
    }

    // Final norm + lm_head.
    {
        std::vector<float> norm_h(dim);
        minllama::rmsnorm_f32(x.data(), model.final_norm_weight.data(),
                               dim, model.rms_norm_eps, norm_h.data(), dim);
        DUMP_("final_norm", norm_h.data(), dim);

        std::vector<float> logits(model.vocab_size);
        minllama::matvec_f32_f32(model.lm_head.data(),
                                  model.vocab_size, dim,
                                  norm_h.data(), dim,
                                  logits.data(), model.vocab_size);

        int top = minllama::argmax_f32(logits.data(), model.vocab_size);
        std::printf("argmax_token=%d val=%.6f\n", top, (double)logits[top]);
    }

    ml_model_free(ml);
    return 0;
}
