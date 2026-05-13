// minllama_ffn_probe: test FFN semantic transpose correctness.
// Usage: ./minllama_ffn_probe --model <gguf> [--transpose]
#include "minllama_internal.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static float norm(const float *v, int n) {
    float s = 0; for (int i=0;i<n;i++) s+=v[i]*v[i]; return std::sqrt(s);
}
static float amax(const float *v, int n) {
    float m=0; for(int i=0;i<n;i++){float a=std::fabs(v[i]);if(a>m)m=a;} return m;
}

static void manual_matvec(const float *w, int rows, int cols,
                          const float *in, float *out) {
    for (int r=0; r<rows; r++) {
        float s=0;
        for (int c=0; c<cols; c++) s += w[r*cols+c] * in[c];
        out[r] = s;
    }
}

int main(int argc, char **argv) {
    std::string model_path;
    bool do_transpose = false;

    for (int i=1; i<argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i+1<argc) model_path = argv[++i];
        else if (a == "--transpose") do_transpose = true;
    }
    if (model_path.empty()) { fprintf(stderr,"--model required\n"); return 1; }

    ml_model *ml = ml_model_load(model_path.c_str());
    if (!ml) return 1;
    minllama::TransformerModelF32 model;
    minllama::load_transformer_model_f32_from_tensors(*ml, model, nullptr);

    auto &l = model.layers[0];
    int dim = l.dim, hd = l.hidden_dim;

    // Create a known hidden state: unit basis vector e[0] = [1,0,0,...]
    std::vector<float> x(dim, 0); x[0] = 1.0f;

    // RMSNorm
    std::vector<float> xn(dim);
    minllama::rmsnorm_f32(x.data(), l.rms_ffn_weight.data(), dim, 1e-5f, xn.data(), dim);

    std::printf("=== FFN Probe (layer 0, unit input e[0]=1) ===\n");
    std::printf("input norm=%.6e absmax=%.6e\n", norm(x.data(),dim), amax(x.data(),dim));
    std::printf("rms_norm norm=%.6e absmax=%.6e\n", norm(xn.data(),dim), amax(xn.data(),dim));

    // ============================================================
    // W1 (gate): [hidden_dim, dim] = [1536, 576]
    // ============================================================
    std::vector<float> gate(hd), gate_t(hd);
    minllama::matvec_f32_f32(l.w1.data(), hd, dim, xn.data(), dim, gate.data(), hd);

    // Transpose version: interpret w1 as [dim, hidden_dim] → matvec gives [dim]
    // then we'd need a different interpretation. Actually, to test transpose:
    // Use W1^T instead of W1: matvec with W1 treated as [dim, hd] giving [dim]
    // But we want [hd] output still. The experiment is:
    // Interpret W1 as having shape [dim, hd] (swap in/out dims)
    // Then gate'[c] = sum_r w1_data[c*dim + r] * input[r] (c in 0..hd-1)
    // This is: gate' = W1_colmajor × input, where W1_colmajor has hd rows, dim cols
    //
    // Original: gate[r] = sum_c w1[r*dim + c] * input[c]  (r in 0..hd-1)
    // Transpose: gate[r] = sum_c w1[c*hd + r] * input[c]  (r in 0..hd-1)
    //
    // The transpose effectively reinterprets W1's layout.

    for (int r = 0; r < hd; r++) {
        float s = 0;
        for (int c = 0; c < dim; c++)
            s += l.w1[c * hd + r] * xn[c];  // swapped: c=in_dim, r=out_dim
        gate_t[r] = s;
    }

    // ============================================================
    // W3 (up): same shape [hidden_dim, dim]
    // ============================================================
    std::vector<float> up(hd), up_t(hd);
    minllama::matvec_f32_f32(l.w3.data(), hd, dim, xn.data(), dim, up.data(), hd);
    for (int r = 0; r < hd; r++) {
        float s = 0;
        for (int c = 0; c < dim; c++)
            s += l.w3[c * hd + r] * xn[c];
        up_t[r] = s;
    }

    // ============================================================
    // SwiGLU
    // ============================================================
    std::vector<float> swiglu(hd), swiglu_t(hd);
    minllama::swiglu_f32(gate.data(), up.data(), hd, swiglu.data(), hd);
    minllama::swiglu_f32(gate_t.data(), up_t.data(), hd, swiglu_t.data(), hd);

    std::printf("\n--- Current layout (no transpose) ---\n");
    std::printf("gate norm=%.4f absmax=%.4f [%.4f %.4f %.4f %.4f %.4f ...]\n",
                norm(gate.data(),hd), amax(gate.data(),hd),
                gate[0], gate[1], gate[2], gate[3], gate[4]);
    std::printf("up   norm=%.4f absmax=%.4f [%.4f %.4f %.4f %.4f %.4f ...]\n",
                norm(up.data(),hd), amax(up.data(),hd),
                up[0], up[1], up[2], up[3], up[4]);
    std::printf("swiglu norm=%.4f absmax=%.4f [%.4f %.4f %.4f %.4f %.4f ...]\n",
                norm(swiglu.data(),hd), amax(swiglu.data(),hd),
                swiglu[0], swiglu[1], swiglu[2], swiglu[3], swiglu[4]);

    std::printf("\n--- Transpose layout ---\n");
    std::printf("gate_t norm=%.4f absmax=%.4f [%.4f %.4f %.4f %.4f %.4f ...]\n",
                norm(gate_t.data(),hd), amax(gate_t.data(),hd),
                gate_t[0], gate_t[1], gate_t[2], gate_t[3], gate_t[4]);
    std::printf("up_t   norm=%.4f absmax=%.4f [%.4f %.4f %.4f %.4f %.4f ...]\n",
                norm(up_t.data(),hd), amax(up_t.data(),hd),
                up_t[0], up_t[1], up_t[2], up_t[3], up_t[4]);
    std::printf("swiglu_t norm=%.4f absmax=%.4f [%.4f %.4f %.4f %.4f %.4f ...]\n",
                norm(swiglu_t.data(),hd), amax(swiglu_t.data(),hd),
                swiglu_t[0], swiglu_t[1], swiglu_t[2], swiglu_t[3], swiglu_t[4]);

    // ============================================================
    // W2 (down): [dim, hidden_dim] = [576, 1536]
    // ============================================================
    std::vector<float> down(dim), down_t(dim);
    minllama::matvec_f32_f32(l.w2.data(), dim, hd, swiglu.data(), hd, down.data(), dim);
    for (int r = 0; r < dim; r++) {
        float s = 0;
        for (int c = 0; c < hd; c++)
            s += l.w2[c * dim + r] * swiglu_t[c];  // swapped interpretation
        down_t[r] = s;
    }

    // Residual
    std::vector<float> out(dim), out_t(dim);
    for (int i=0; i<dim; i++) { out[i] = x[i] + down[i]; out_t[i] = x[i] + down_t[i]; }

    std::printf("\n--- W2 + residual ---\n");
    std::printf("down      norm=%.4f absmax=%.4f\n", norm(down.data(),dim), amax(down.data(),dim));
    std::printf("down_t    norm=%.4f absmax=%.4f\n", norm(down_t.data(),dim), amax(down_t.data(),dim));
    std::printf("output    norm=%.4f absmax=%.4f\n", norm(out.data(),dim), amax(out.data(),dim));
    std::printf("output_t  norm=%.4f absmax=%.4f\n", norm(out_t.data(),dim), amax(out_t.data(),dim));

    // ============================================================
    // Full layer test with real embedding
    // ============================================================
    std::printf("\n=== Full layer test (token 28120, real embedding) ===\n");

    std::vector<float> emb(dim);
    minllama::token_embedding_lookup_f32(model, 28120, emb.data());

    // Run full layer
    std::vector<float> layer_out(dim);
    minllama::transformer_layer_decode_f32(l, emb.data(), model.kv_caches[0], 0, layer_out.data());

    // Run with transposed FFN
    // (repeat the layer but with transposed FFN weights)
    // First: do attention part normally
    std::vector<float> xn2(dim);
    minllama::rmsnorm_f32(emb.data(), l.rms_att_weight.data(), dim, 1e-5f, xn2.data(), dim);
    std::vector<float> q(dim), k(192), v(192);
    minllama::matvec_f32_f32(l.wq.data(), dim, dim, xn2.data(), dim, q.data(), dim);
    minllama::matvec_f32_f32(l.wk.data(), 192, dim, xn2.data(), dim, k.data(), 192);
    minllama::matvec_f32_f32(l.wv.data(), 192, dim, xn2.data(), dim, v.data(), 192);
    for (int h=0; h<9; h++) minllama::rope_apply_f32(q.data()+h*64, 64, 0, 10000.0f);
    for (int h=0; h<3; h++) minllama::rope_apply_f32(k.data()+h*64, 64, 0, 10000.0f);
    minllama::KvCacheF32 cache_t;
    minllama::kv_cache_init_gqa_f32(cache_t, 8, 3, 64);
    minllama::kv_cache_write_f32(cache_t, 0, k.data(), v.data());
    std::vector<float> att(dim);
    minllama::attention_decode_gqa_f32(q.data(), cache_t, 0, 9, 3, 64, att.data());
    std::vector<float> wo_out(dim);
    minllama::matvec_f32_f32(l.wo.data(), dim, dim, att.data(), dim, wo_out.data(), dim);
    std::vector<float> h(dim);
    for (int i=0; i<dim; i++) h[i] = emb[i] + wo_out[i];

    // FFN with transposed weights
    std::vector<float> ffn_norm(dim);
    minllama::rmsnorm_f32(h.data(), l.rms_ffn_weight.data(), dim, 1e-5f, ffn_norm.data(), dim);
    std::vector<float> gate2(hd), up2(hd);
    for (int r=0; r<hd; r++) {
        float sg=0, su=0;
        for (int c=0; c<dim; c++) { sg+=l.w1[c*hd+r]*ffn_norm[c]; su+=l.w3[c*hd+r]*ffn_norm[c]; }
        gate2[r]=sg; up2[r]=su;
    }
    std::vector<float> swi2(hd);
    minllama::swiglu_f32(gate2.data(), up2.data(), hd, swi2.data(), hd);
    std::vector<float> down2(dim);
    for (int r=0; r<dim; r++) {
        float s=0;
        for (int c=0; c<hd; c++) s+=l.w2[c*dim+r]*swi2[c];
        down2[r]=s;
    }
    std::vector<float> layer_out_t(dim);
    for (int i=0; i<dim; i++) layer_out_t[i] = h[i] + down2[i];

    std::printf("layer_out (current)  norm=%.4f absmax=%.4f\n",
                norm(layer_out.data(),dim), amax(layer_out.data(),dim));
    std::printf("layer_out (transpose) norm=%.4f absmax=%.4f\n",
                norm(layer_out_t.data(),dim), amax(layer_out_t.data(),dim));

    ml_model_free(ml);
    return 0;
}
