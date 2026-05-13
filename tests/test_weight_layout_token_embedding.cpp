// test_weight_layout_token_embedding: verify token embedding layout
#include "minllama_internal.h"
#include <cmath>
#include <cstdio>

static float cosine(const float *a, const float *b, int n) {
    float dot=0, na=0, nb=0;
    for(int i=0;i<n;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return dot/(std::sqrt(na)*std::sqrt(nb));
}

int main() {
    const char *path = "../models/SmolLM-135M.Q4_0.gguf";
    ml_model *ml = ml_model_load(path);
    if (!ml) { std::fprintf(stderr,"load fail\n"); return 1; }

    minllama::TransformerModelF32 model;
    std::string err;
    if (!minllama::load_transformer_model_f32_from_tensors(*ml, model, &err)) {
        std::fprintf(stderr,"%s\n",err.c_str()); return 1;
    }

    int dim = model.dim, vocab = model.vocab_size;
    std::printf("dim=%d vocab=%d\n", dim, vocab);

    // Load raw reference
    std::vector<float> raw;
    minllama::load_tensor_as_f32(path, ml->tensor_index, "token_embd.weight", &raw);
    std::printf("raw.size=%zu model.size=%zu\n", raw.size(), model.token_embedding.size());

    // Test 1: verify model.token_embedding == raw (no transpose corruption)
    bool identical = (model.token_embedding == raw);
    std::printf("Test 1 - model == raw: %s\n", identical ? "PASS" : "FAIL");

    // Test 2: token 28120 embedding values
    std::vector<float> emb(dim);
    minllama::token_embedding_lookup_f32(model, 28120, emb.data());
    std::printf("Test 2 - token 28120 first 8: ");
    for (int i=0;i<8;i++) std::printf("%.4f ", (double)emb[i]);
    std::printf("\n  raw first 8: ");
    for (int i=0;i<8;i++) std::printf("%.4f ", (double)raw[28120*dim+i]);

    bool match = true;
    for (int i=0;i<dim;i++) if(std::fabs(emb[i]-raw[28120*dim+i])>1e-5f){match=false;break;}
    std::printf("\n  match: %s\n", match ? "PASS" : "FAIL");

    // Test 3: embedding norms for first 10 tokens
    std::printf("Test 3 - token norms: ");
    for (int t=0;t<10;t++) {
        float n=0;
        for(int i=0;i<dim;i++) n+=model.token_embedding[t*dim+i]*model.token_embedding[t*dim+i];
        std::printf("%.2f ", std::sqrt(n));
    }
    std::printf("\n");

    // Test 4: cosine similarity between different tokens (should not be 1.0)
    std::vector<float> e0(dim), e1(dim);
    minllama::token_embedding_lookup_f32(model, 0, e0.data());
    minllama::token_embedding_lookup_f32(model, 1, e1.data());
    float c01 = cosine(e0.data(), e1.data(), dim);
    std::printf("Test 4 - cosine(e0,e1)=%.4f (should be < 0.5, not 1.0): %s\n",
                (double)c01, c01 < 0.5f ? "PASS" : "FAIL");

    // Test 5: token 28120 embedding should differ from token 0
    std::vector<float> e28120(dim);
    minllama::token_embedding_lookup_f32(model, 28120, e28120.data());
    float c0 = cosine(e0.data(), e28120.data(), dim);
    std::printf("Test 5 - cosine(e0,e28120)=%.4f (should not be 1.0): %s\n",
                (double)c0, std::fabs(c0-1.0f) > 0.01f ? "PASS" : "FAIL");

    ml_model_free(ml);
    return identical && match && !(c01 > 0.5f) ? 0 : 1;
}
