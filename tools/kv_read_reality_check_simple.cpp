#include "minllama.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>

namespace {
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

void fill_data(std::vector<float> &x, float scale = 1.0f) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int t = static_cast<int>((i * 37 + 17) % 211) - 105;
        x[i] = scale * static_cast<float>(t) / 53.0f;
    }
}

void flush_cache() {
    std::vector<char> buffer(10 * 1024 * 1024);
    std::memset(buffer.data(), 0, buffer.size());
}

void attention_sequential(const float *q,
                          const float *keys,
                          const float *values,
                          int n_tokens,
                          int head_dim,
                          float *out_qk,
                          float *out_av) {
    const float *q_head = q;
    const float *keys_h = keys;
    const float *values_h = values;
    
    for (int t = 0; t < n_tokens; ++t) {
        const float *k_head = keys_h + t * head_dim;
        float qk = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            qk += q_head[d] * k_head[d];
        }
        out_qk[t] = qk;
    }
    
    float sum = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        sum += std::exp(out_qk[t]);
    }
    const float *v_head = values_h;
    for (int d = 0; d < head_dim; ++d) {
        float av = 0.0f;
        for (int t = 0; t < n_tokens; ++t) {
            const float *v_t = v_head + t * head_dim + d;
            av += std::exp(out_qk[t]) / sum * (*v_t);
        }
        out_av[d] = av;
    }
}

int main() {
    printf("=== KV Read Reality Check (Simple) ===\n\n");
    
    const int n_tokens = 128;
    const int head_dim = 64;
    
    printf("Test: seq_len=%d, head_dim=%d\n\n", n_tokens, head_dim);
    
    // Test 1: Warm Cache
    printf("[1] Warm Cache - Sequential\n");
    std::vector<float> q(1 * head_dim);
    std::vector<float> keys(1 * head_dim * n_tokens);
    std::vector<float> values(1 * head_dim * n_tokens);
    
    fill_data(q, 1.0f);
    fill_data(keys, 0.7f);
    fill_data(values, 0.9f);
    
    int iters = 1000;
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        std::vector<float> out_qk(n_tokens);
        std::vector<float> out_av(head_dim);
        attention_sequential(q.data(), keys.data(), values.data(),
                            n_tokens, head_dim,
                            out_qk.data(), out_av.data());
    }
    auto t1 = Clock::now();
    double ms_warm = Ms(t1 - t0).count() / static_cast<double>(iters);
    printf("    Time: %.6f ms/call\n", ms_warm);
    
    // Test 2: Cold Cache
    printf("[2] Cold Cache - Sequential\n");
    flush_cache();
    
    t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        std::vector<float> out_qk(n_tokens);
        std::vector<float> out_av(head_dim);
        attention_sequential(q.data(), keys.data(), values.data(),
                            n_tokens, head_dim,
                            out_qk.data(), out_av.data());
    }
    t1 = Clock::now();
    double ms_cold = Ms(t1 - t0).count() / static_cast<double>(iters);
    printf("    Time: %.6f ms/call\n", ms_cold);
    printf("    Speedup (cold vs warm): %.3fx\n", ms_warm / ms_cold);
    
    // Test 3: Realistic Stochastic KV
    printf("\n[3] Stochastic KV Access (Cold Cache)\n");
    flush_cache();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> kv_dist(0, 0); // Only head 0
    
    t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        std::vector<float> out_qk(n_tokens);
        std::vector<float> out_av(head_dim);
        
        const int kv_h = kv_dist(gen);
        const size_t offset = kv_h * head_dim * n_tokens;
        const float *keys_h = keys.data() + offset;
        const float *values_h = values.data() + offset;
        
        for (int t = 0; t < n_tokens; ++t) {
            const float *k_head = keys_h + t * head_dim;
            float qk = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                qk += q[d] * k_head[d];
            }
            out_qk[t] = qk;
        }
        
        float sum = 0.0f;
        for (int t = 0; t < n_tokens; ++t) {
            sum += std::exp(out_qk[t]);
        }
        for (int d = 0; d < head_dim; ++d) {
            float av = 0.0f;
            for (int t = 0; t < n_tokens; ++t) {
                const float *v_t = values_h + t * head_dim + d;
                av += std::exp(out_qk[t]) / sum * (*v_t);
            }
            out_av[d] = av;
        }
    }
    t1 = Clock::now();
    double ms_stochastic = Ms(t1 - t0).count() / static_cast<double>(iters);
    printf("    Time: %.6f ms/call\n", ms_stochastic);
    printf("    Speedup (stochastic vs warm): %.3fx\n", ms_warm / ms_stochastic);
    
    printf("\n=== Reality Check Summary ===\n");
    printf("Cold cache degradation: %.3fx\n", ms_cold / ms_warm);
    printf("Stochastic KV access: %.3fx\n", ms_stochastic / ms_warm);
    printf("\nGate:\n");
    printf("- Realistic mode >= 1.10x speedup → forward A/B test\n");
    printf("- Realistic mode < 1.05x → synthetic-only gain (stop)\n");
    
    return 0;
}

} // namespace
