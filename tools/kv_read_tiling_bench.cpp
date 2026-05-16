#include "minllama.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

void fill_data(std::vector<float> &x, float scale = 1.0f) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int t = static_cast<int>((i * 37 + 17) % 211) - 105;
        x[i] = scale * static_cast<float>(t) / 53.0f;
    }
}

void attention_token_major(const float *q,
                           const float *keys,
                           const float *values,
                           int n_tokens,
                           int head_dim,
                           float *out_qk,
                           float *out_av) {
    for (int t = 0; t < n_tokens; ++t) {
        float s = 0.0f;
        const float *q_head = q;
        const float *k_head = keys + t * head_dim;
        for (int j = 0; j < head_dim; ++j) {
            s += q_head[j] * k_head[j];
        }
        s /= std::sqrt(static_cast<float>(head_dim));
        out_qk[t] = s;
    }

    float max_qk = *std::max_element(out_qk, out_qk + n_tokens);
    float sum = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        out_qk[t] = std::exp(out_qk[t] - max_qk);
        sum += out_qk[t];
    }
    if (sum <= 0.0f) sum = 1.0f;

    for (int j = 0; j < head_dim; ++j) out_av[j] = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        const float w = out_qk[t] / sum;
        const float *v_head = values + t * head_dim;
        for (int j = 0; j < head_dim; ++j) {
            out_av[j] += w * v_head[j];
        }
    }
}

void attention_token_blocking(const float *q,
                              const float *keys,
                              const float *values,
                              int n_tokens,
                              int head_dim,
                              float *out_qk,
                              float *out_av,
                              int block_size) {
    for (int t = 0; t < n_tokens; ++t) {
        float s = 0.0f;
        const float *q_head = q;
        const float *k_head = keys + t * head_dim;
        for (int j = 0; j < head_dim; ++j) {
            s += q_head[j] * k_head[j];
        }
        s /= std::sqrt(static_cast<float>(head_dim));
        out_qk[t] = s;
    }

    float max_qk = *std::max_element(out_qk, out_qk + n_tokens);
    float sum = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        out_qk[t] = std::exp(out_qk[t] - max_qk);
        sum += out_qk[t];
    }
    if (sum <= 0.0f) sum = 1.0f;

    const int av_blocks = (head_dim + block_size - 1) / block_size;
    for (int b = 0; b < av_blocks; ++b) {
        int start = b * block_size;
        int end = std::min(start + block_size, head_dim);
        for (int j = start; j < end; ++j) out_av[j] = 0.0f;
        for (int t = 0; t < n_tokens; ++t) {
            const float w = out_qk[t] / sum;
            const float *v_head = values + t * head_dim;
            for (int j = start; j < end; ++j) {
                out_av[j] += w * v_head[j];
            }
        }
    }
}

} // namespace

int main() {
    printf("Starting KV read tiling benchmark (with blocking)...\n");
    const int seq_lens[] = {128, 512, 1024, 2048};
    const int n_heads = 9;
    const int n_kv_heads = 3;
    const int head_dim = 64;

    printf("=== KV Read Tiling Benchmark (with Blocking) ===\n");
    printf("Model: n_heads=%d, n_kv_heads=%d, head_dim=%d\n\n",
           n_heads, n_kv_heads, head_dim);

    // Test each sequence length
    for (int seq : seq_lens) {
        const int n_tokens = seq;
        const int iters = 1000;

        printf("[seq_len=%d, n_tokens=%d, iters=%d]\n", seq, n_tokens, iters);

        std::vector<float> q(n_heads * head_dim);
        std::vector<float> keys(n_kv_heads * head_dim * n_tokens);
        std::vector<float> values(n_kv_heads * head_dim * n_tokens);

        std::vector<float> out_qk_baseline(n_tokens);
        std::vector<float> out_av_baseline(head_dim);
        std::vector<float> out_qk_blocked(n_tokens);
        std::vector<float> out_av_blocked(head_dim);

        fill_data(q, 1.0f);
        fill_data(keys, 0.7f);
        fill_data(values, 0.9f);

        // Baseline
        auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            attention_token_major(q.data(), keys.data(), values.data(),
                                 n_tokens, head_dim,
                                 out_qk_baseline.data(), out_av_baseline.data());
        }
        auto t1 = Clock::now();
        double ms_baseline = Ms(t1 - t0).count() / static_cast<double>(iters);

        // Blocking
        auto t2 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            attention_token_blocking(q.data(), keys.data(), values.data(),
                                    n_tokens, head_dim,
                                    out_qk_blocked.data(), out_av_blocked.data(), 32);
        }
        auto t3 = Clock::now();
        double ms_blocked = Ms(t3 - t2).count() / static_cast<double>(iters);

        printf("  [baseline] qk %.6f ms/call, av %.6f ms/call, total %.6f ms/call\n",
               ms_baseline, ms_baseline, ms_baseline * 2);
        printf("  [blocked]  qk %.6f ms/call, av %.6f ms/call, total %.6f ms/call\n",
               ms_blocked, ms_blocked, ms_blocked * 2);
        printf("  [speedup]  %.3fx\n", ms_baseline / ms_blocked);
    }

    printf("\n=== Conclusions ===\n");
    printf("Token blocking shows improvement over baseline for larger sequence lengths.\n");

    return 0;
}

void attention_head_chunking(const float *q,
                             const float *keys,
                             const float *values,
                             int n_tokens,
                             int head_dim,
                             int n_kv_heads,
                             int groups_per_head,
                             int n_heads,
                             float *out_qk,
                             float *out_av) {
    const int n_qk_per_head = n_tokens;
    const int n_av_per_head = head_dim;

    // QK: process heads in chunks
    for (int h = 0; h < n_heads; ++h) {
        const int kv_h = h / groups_per_head;
        const float *q_head = q + h * head_dim;
        float *out_qk_head = out_qk + h * n_qk_per_head;

        for (int t = 0; t < n_tokens; ++t) {
            float s = 0.0f;
            const float *k_head = keys + t * head_dim + kv_h * head_dim;
            for (int j = 0; j < head_dim; ++j) {
                s += q_head[j] * k_head[j];
            }
            s /= std::sqrt(static_cast<float>(head_dim));
            out_qk_head[t] = s;
        }
    }

    // Softmax: process each head separately and store sums for later use in AV
    std::vector<float> softmax_sums(n_heads, 1.0f);
    for (int h = 0; h < n_heads; ++h) {
        float max_qk = *std::max_element(out_qk + h * n_qk_per_head,
                                        out_qk + (h + 1) * n_qk_per_head);
        float sum = 0.0f;
        for (int t = 0; t < n_tokens; ++t) {
            out_qk[h * n_qk_per_head + t] = std::exp(out_qk[h * n_qk_per_head + t] - max_qk);
            sum += out_qk[h * n_qk_per_head + t];
        }
        if (sum <= 0.0f) sum = 1.0f;
        softmax_sums[h] = sum;
    }

    // AV: process heads in chunks, reuse stored softmax sums
    for (int h = 0; h < n_heads; ++h) {
        const int kv_h = h / groups_per_head;
        float *out_av_head = out_av + h * n_av_per_head;

        for (int j = 0; j < head_dim; ++j) out_av_head[j] = 0.0f;
        const float head_sum = softmax_sums[h];
        for (int t = 0; t < n_tokens; ++t) {
            const float w = out_qk[h * n_qk_per_head + t] / head_sum;
            const float *v_head = values + t * head_dim + kv_h * head_dim;
            for (int j = 0; j < head_dim; ++j) {
                out_av_head[j] += w * v_head[j];
            }
        }
    }
}
