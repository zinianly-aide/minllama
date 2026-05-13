// minllama_dump_weight_layout: inspect GGUF weight layout vs loaded layout
#include "minllama_internal.h"
#include <cstdio>

static void dump_weights(const char *path) {
    ml_model *ml = ml_model_load(path);
    if (!ml) return;

    // Raw loads
    auto raw_load = [&](const char *name, std::vector<float> &v) {
        minllama::load_tensor_as_f32(path, ml->tensor_index, name, &v);
    };

    auto print_info = [&](const char *name, const std::vector<float> &loaded, int rows, int cols,
                           const char *llama_semantics) {
        const auto *ti = ml->tensor_index.find(name);
        if (!ti) { std::printf("%s: NOT FOUND\n", name); return; }

        std::vector<float> raw;
        raw_load(name, raw);

        bool same = (loaded == raw);
        std::printf("%-40s GGUF[%llu,%llu] target[%d,%d] %s\n",
                    name, (unsigned long long)ti->dims[0], (unsigned long long)ti->dims[1],
                    rows, cols, same ? "NO-TRANSPOSE" : "TRANSPOSED");
        std::printf("  llama.cpp: %s\n", llama_semantics);
        std::printf("  loaded[0..7]: ");
        for (int i=0;i<8&&i<(int)loaded.size();i++) std::printf("%.4f ", (double)loaded[i]);
        std::printf("\n  raw[0..7]:    ");
        for (int i=0;i<8&&i<(int)raw.size();i++) std::printf("%.4f ", (double)raw[i]);
        std::printf("\n");
    };

    minllama::TransformerModelF32 model;
    minllama::load_transformer_model_f32_from_tensors(*ml, model, nullptr);

    int dim = model.dim, vocab = model.vocab_size;
    auto &l0 = model.layers[0];
    int hd = l0.hidden_dim, kv_dim = l0.n_kv_heads * l0.head_dim;

    std::printf("=== Token Embedding ===\n");
    print_info("token_embd.weight", model.token_embedding, vocab, dim,
               "ggml_get_rows: emb[i] = weight[tok*dim+i]");

    std::printf("\n=== Attention Weights (layer 0) ===\n");
    print_info("blk.0.attn_q.weight", l0.wq, dim, dim,
               "mul_mat(q, x): q[r] = sum_c wq[r*dim+c]*x[c]");
    print_info("blk.0.attn_k.weight", l0.wk, kv_dim, dim,
               "mul_mat(k, x_norm): k[r] = sum_c wk[r*dim+c]*xn[c]");
    print_info("blk.0.attn_v.weight", l0.wv, kv_dim, dim,
               "mul_mat(v, x_norm): v[r] = sum_c wv[r*dim+c]*xn[c]");
    print_info("blk.0.attn_output.weight", l0.wo, dim, dim,
               "mul_mat(wo, att): out[r] = sum_c wo[r*dim+c]*att[c]");

    std::printf("\n=== FFN Weights (layer 0) ===\n");
    print_info("blk.0.ffn_gate.weight", l0.w1, hd, dim,
               "mul_mat(w1, x): gate[r] = sum_c w1[r*dim+c]*x[c]");
    print_info("blk.0.ffn_down.weight", l0.w2, dim, hd,
               "mul_mat(w2, h): out[r] = sum_c w2[r*hd+c]*h[c]");
    print_info("blk.0.ffn_up.weight", l0.w3, hd, dim,
               "mul_mat(w3, x): up[r] = sum_c w3[r*dim+c]*x[c]");

    ml_model_free(ml);
}

int main(int argc, char **argv) {
    const char *path = "../models/SmolLM-135M.Q4_0.gguf";
    for (int i=1; i<argc; i++) {
        if (std::string(argv[i])=="--model" && i+1<argc) path = argv[++i];
    }
    dump_weights(path);
    return 0;
}
