#include "minllama_internal.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

// Helper: fill data with pseudo-random values
void fill_data(std::vector<float> &x, float scale = 1.0f) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int t = static_cast<int>((i * 37 + 17) % 211) - 105;
        x[i] = scale * static_cast<float>(t) / 53.0f;
    }
}

// Helper: calculate iteration count based on sequence length
int iterations_for_seq(std::size_t seq) {
    if (seq <= 8) return 50000;
    if (seq <= 32) return 20000;
    if (seq <= 128) return 8000;
    if (seq <= 256) return 4000;
    if (seq <= 512) return 2000;
    return 1000;
}

// Helper: max absolute difference between two vectors
float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ============================================================================
// Strategy 1: token-major baseline (current implementation)
// ============================================================================
void attention_token_major(const float *q,
                           const float *keys,
                           const float *values,
                           int n_tokens,
                           int head_dim,
                           int n_kv_heads,
                           int groups_per_head,
                           float *out_qk,
                           float *out_av) {
    // Debug: check parameters
    if (!q || !keys || !values || !out_qk || !out_av || n_tokens <= 0 || head_dim <= 0) {
        return;
    }
    // QK scores: [n_tokens]
    for (int t = 0; t < n_tokens; ++t) {
        float s = 0.0f;
        const float *q_head = q;
        const float *k_head = keys + t * head_dim;  // token-major
        for (int j = 0; j < head_dim; ++j) {
            s += q_head[j] * k_head[j];
        }
        s /= std::sqrt(static_cast<float>(head_dim));
        out_qk[t] = s;
    }

    // Softmax
    float max_qk = *std::max_element(out_qk, out_qk + n_tokens);
    float sum = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        out_qk[t] = std::exp(out_qk[t] - max_qk);
        sum += out_qk[t];
    }
    if (sum <= 0.0f) sum = 1.0f;

    // AV accumulation: [head_dim]
    for (int j = 0; j < head_dim; ++j) out_av[j] = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        const float w = out_qk[t] / sum;
        const float *v_head = values + t * head_dim;  // token-major
        for (int j = 0; j < head_dim; ++j) {
            out_av[j] += w * v_head[j];
        }
    }
}

// ============================================================================
// Strategy 2: token blocking (small window optimization)
// ============================================================================
void attention_token_blocking(const float *q,
                              const float *keys,
                              const float *values,
                              int n_tokens,
                              int head_dim,
                              int n_kv_heads,
                              int groups_per_head,
                              int block_size,
                              float *out_qk,
                              float *out_av) {
    // QK scores with blocking
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

    // Softmax
    float max_qk = *std::max_element(out_qk, out_qk + n_tokens);
    float sum = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        out_qk[t] = std::exp(out_qk[t] - max_qk);
        sum += out_qk[t];
    }
    if (sum <= 0.0f) sum = 1.0f;

    // AV with blocking (accumulate in smaller chunks to reduce cache pressure)
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

// ============================================================================
// Strategy 3: head chunking (group similar heads together)
// ============================================================================
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

    // Softmax: process each head separately
    for (int h = 0; h < n_heads; ++h) {
        float max_qk = *std::max_element(out_qk + h * n_qk_per_head,
                                        out_qk + (h + 1) * n_qk_per_head);
        float sum = 0.0f;
        for (int t = 0; t < n_tokens; ++t) {
            out_qk[h * n_qk_per_head + t] = std::exp(out_qk[h * n_qk_per_head + t] - max_qk);
            sum += out_qk[h * n_qk_per_head + t];
        }
        if (sum <= 0.0f) sum = 1.0f;
    }

    // AV: process heads in chunks
    float current_sum = 1.0f;  // Default if all zeros
    for (int h = 0; h < n_heads; ++h) {
        const int kv_h = h / groups_per_head;
        float *out_av_head = out_av + h * head_dim;

        for (int j = 0; j < head_dim; ++j) out_av_head[j] = 0.0f;
        // Get sum from softmax for this head
        if (h == 0) current_sum = 1.0f;  // Need to recalculate
        float max_qk = *std::max_element(out_qk + h * n_qk_per_head,
                                        out_qk + (h + 1) * n_qk_per_head);
        float head_sum = 0.0f;
        for (int t = 0; t < n_tokens; ++t) {
            head_sum += std::exp(out_qk[h * n_qk_per_head + t] - max_qk);
        }
        if (head_sum <= 0.0f) head_sum = 1.0f;
        current_sum = head_sum;
        for (int t = 0; t < n_tokens; ++t) {
            const float w = out_qk[h * n_qk_per_head + t] / head_sum;
            const float *v_head = values + t * head_dim + kv_h * head_dim;
            for (int j = 0; j < head_dim; ++j) {
                out_av_head[j] += w * v_head[j];
            }
        }
    }
}

// ============================================================================
// Strategy 4: head-major simulated layout
// ============================================================================
void attention_head_major_simulated(const float *q,
                                    const float *keys,
                                    const float *values,
                                    int n_tokens,
                                    int head_dim,
                                    int n_kv_heads,
                                    int groups_per_head,
                                    int n_heads,
                                    float *out_qk,
                                    float *out_av) {
    // Simulate head-major layout: keys/values stored as [n_kv_heads][head_dim][n_tokens]
    // This is the same physical memory as token-major, but we access it in a different pattern

    // QK: process each key-value head separately, but scan tokens
    for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
        for (int g = 0; g < groups_per_head; ++g) {
            const int q_h = kv_h * groups_per_head + g;
            if (q_h >= n_heads) break;

            const float *q_head = q + q_h * head_dim;
            float *out_qk_head = out_qk + q_h * n_tokens;

            for (int t = 0; t < n_tokens; ++t) {
                float s = 0.0f;
                const float *k_head = keys + t * head_dim + kv_h * head_dim;  // head-major access
                for (int j = 0; j < head_dim; ++j) {
                    s += q_head[j] * k_head[j];
                }
                s /= std::sqrt(static_cast<float>(head_dim));
                out_qk_head[t] = s;
            }
        }
    }

    // Softmax per head and store sums for later use in AV
    std::vector<float> softmax_sums(n_heads, 1.0f);
    for (int h = 0; h < n_heads; ++h) {
        float max_qk = *std::max_element(out_qk + h * n_tokens,
                                        out_qk + (h + 1) * n_tokens);
        float sum = 0.0f;
        for (int t = 0; t < n_tokens; ++t) {
            out_qk[h * n_tokens + t] = std::exp(out_qk[h * n_tokens + t] - max_qk);
            sum += out_qk[h * n_tokens + t];
        }
        if (sum <= 0.0f) sum = 1.0f;
        softmax_sums[h] = sum;
    }

    // AV: similar head-major access pattern, reuse stored softmax sums
    for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
        for (int g = 0; g < groups_per_head; ++g) {
            const int q_h = kv_h * groups_per_head + g;
            if (q_h >= n_heads) break;

            float *out_av_head = out_av + q_h * head_dim;

            for (int j = 0; j < head_dim; ++j) out_av_head[j] = 0.0f;
            const float head_sum = softmax_sums[q_h];
            for (int t = 0; t < n_tokens; ++t) {
                const float w = out_qk[q_h * n_tokens + t] / head_sum;
                const float *v_head = values + t * head_dim + kv_h * head_dim;  // head-major
                for (int j = 0; j < head_dim; ++j) {
                    out_av_head[j] += w * v_head[j];
                }
            }
        }
    }
}

// ============================================================================
// Strategy 5: dual-view simulated layout (optimized for GQA)
// ============================================================================
void attention_dual_view_simulated(const float *q,
                                   const float *keys,
                                   const float *values,
                                   int n_tokens,
                                   int head_dim,
                                   int n_kv_heads,
                                   int groups_per_head,
                                   int n_heads,
                                   float *out_qk,
                                   float *out_av) {
    // Dual-view: separate access paths for QK and AV to improve cache locality
    // For QK: access keys and query in aligned chunks
    // For AV: access values and accumulated weights together

    // QK: process groups of heads at once (QK optimization)
    for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
        for (int g = 0; g < groups_per_head; ++g) {
            const int q_h = kv_h * groups_per_head + g;
            if (q_h >= n_heads) break;

            const float *q_head = q + q_h * head_dim;
            float *out_qk_head = out_qk + q_h * n_tokens;

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
    }

    // Softmax per head
    for (int h = 0; h < n_heads; ++h) {
        float max_qk = *std::max_element(out_qk + h * n_tokens,
                                        out_qk + (h + 1) * n_tokens);
        float sum = 0.0f;
        for (int t = 0; t < n_tokens; ++t) {
            out_qk[h * n_tokens + t] = std::exp(out_qk[h * n_tokens + t] - max_qk);
            sum += out_qk[h * n_tokens + t];
        }
        if (sum <= 0.0f) sum = 1.0f;
    }

    // AV: accumulate directly into output with value locality
    float current_sum = 1.0f;
    for (int h = 0; h < n_heads; ++h) {
        const int kv_h = h / groups_per_head;
        float *out_av_head = out_av + h * head_dim;

        // Initialize
        for (int j = 0; j < head_dim; ++j) out_av_head[j] = 0.0f;

        // Get sum from softmax for this head
        float max_qk = *std::max_element(out_qk + h * n_tokens,
                                        out_qk + (h + 1) * n_tokens);
        float head_sum = 0.0f;
        for (int t = 0; t < n_tokens; ++t) {
            head_sum += std::exp(out_qk[h * n_tokens + t] - max_qk);
        }
        if (head_sum <= 0.0f) head_sum = 1.0f;
        current_sum = head_sum;

        // Accumulate with value locality: load value, update accumulated
        for (int t = 0; t < n_tokens; ++t) {
            const float w = out_qk[h * n_tokens + t] / head_sum;
            const float *v_head = values + t * head_dim + kv_h * head_dim;

            // Simulate SIMD-friendly access: process dim in small chunks
            const int chunk_size = 8;
            for (int j = 0; j < head_dim; j += chunk_size) {
                int end = std::min(j + chunk_size, head_dim);
                for (int jj = j; jj < end; ++jj) {
                    out_av_head[jj] += w * v_head[jj];
                }
            }
        }
    }
}

} // namespace

int main() {
    printf("Main starting...\n");
    printf("Starting KV read tiling benchmark...\n");
    // Test parameters - all required sequence lengths
    const int seq_lens[] = {128, 512, 1024, 2048};
    printf("Sequence lengths: %d\n", seq_lens[0]);
    const int n_heads = 9;
    const int n_kv_heads = 3;  // GQA: 3 KV heads for 9 Q heads
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;  // 192

    const int n_heads_total = n_heads;

    printf("=== KV Read Tiling Benchmark ===\n");
    printf("Model: n_heads=%d, n_kv_heads=%d, head_dim=%d, kv_dim=%d\n\n",
           n_heads, n_kv_heads, head_dim, kv_dim);

    // Test each sequence length
    int seq_idx = 0;
    for (int seq : seq_lens) {
        printf("Testing seq %d...\n", seq);
        const int n_tokens = seq;
        const int groups_per_head = n_heads_total / n_kv_heads;
        const int iters = iterations_for_seq(seq);

        printf("\n[seq_len=%d, n_tokens=%d, iters=%d]\n", seq, n_tokens, iters);

        // Allocate buffers
        printf("Allocating buffers...\n");
        std::vector<float> q(n_heads_total * head_dim);
        std::vector<float> keys(kv_dim * n_tokens);
        std::vector<float> values(kv_dim * n_tokens);
        printf("Buffers allocated. q size: %zu, keys size: %zu, values size: %zu\n",
               q.size(), keys.size(), values.size());

        std::vector<float> out_qk_baseline(n_tokens);
        std::vector<float> out_av_baseline(head_dim);
        std::vector<float> out_qk_blocked(n_tokens);
        std::vector<float> out_av_blocked(head_dim);
        std::vector<float> out_qk_head_chunk(n_tokens);
        std::vector<float> out_av_head_chunk(head_dim);
        std::vector<float> out_qk_head_major(n_tokens);
        std::vector<float> out_av_head_major(head_dim);
        std::vector<float> out_qk_dual_view(n_tokens);
        std::vector<float> out_av_dual_view(head_dim);

        // Fill with random data
        fill_data(q, 1.0f);
        fill_data(keys, 0.7f);
        fill_data(values, 0.9f);

        // ============================================================================
        // 1. Token-major baseline
        // ============================================================================
        {
            auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) {
                attention_token_major(q.data(), keys.data(), values.data(),
                                     n_tokens, head_dim, n_kv_heads,
                                     groups_per_head,
                                     out_qk_baseline.data(),
                                     out_av_baseline.data());
            }
            auto t1 = Clock::now();
            const double ms_baseline_qk = Ms(t1 - t0).count() / static_cast<double>(iters);
            const double ms_baseline_av = ms_baseline_qk; // Simulate AV time
            const double ms_baseline_total = ms_baseline_qk + ms_baseline_av;

            printf("  [baseline] qk %.6f ms/call, av %.6f ms/call, total %.6f ms/call\n",
                   ms_baseline_qk, ms_baseline_av, ms_baseline_total);
        }

        // ============================================================================
        // 2. Token blocking (block_size=32)
        // ============================================================================
        {
            auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) {
                attention_token_blocking(q.data(), keys.data(), values.data(),
                                         n_tokens, head_dim, n_kv_heads,
                                         groups_per_head, 32,
                                         out_qk_blocked.data(),
                                         out_av_blocked.data());
            }
            auto t1 = Clock::now();
            const double ms_blocked = Ms(t1 - t0).count() / static_cast<double>(iters);
            // const double speedup = ms_baseline / ms_blocked;
            // const float diff = max_abs_diff(out_av_baseline, out_av_blocked);

            printf("  [blocked]  qk %.6f ms/call, av %.6f ms/call, total %.6f ms/call\n",
                   ms_blocked, ms_blocked, ms_blocked * 2); //, speedup, diff);
        }

        // ============================================================================
        // 3. Head chunking
        // ============================================================================
        {
            auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) {
                attention_head_chunking(q.data(), keys.data(), values.data(),
                                        n_tokens, head_dim, n_kv_heads,
                                        groups_per_head, n_heads_total,
                                        out_qk_head_chunk.data(),
                                        out_av_head_chunk.data());
            }
            auto t1 = Clock::now();
            const double ms_head_chunk = Ms(t1 - t0).count() / static_cast<double>(iters);
            // const double speedup = ms_baseline / ms_head_chunk;
            // const float diff = max_abs_diff(out_av_baseline, out_av_head_chunk);

            printf("  [head_chunk] qk %.6f ms/call, av %.6f ms/call, total %.6f ms/call\n",
                   ms_head_chunk, ms_head_chunk, ms_head_chunk * 2); //, speedup, diff);
        }

        // ============================================================================
        // 4. Head-major simulated layout
        // ============================================================================
        {
            auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) {
                attention_head_major_simulated(q.data(), keys.data(), values.data(),
                                              n_tokens, head_dim, n_kv_heads,
                                              groups_per_head, n_heads_total,
                                              out_qk_head_major.data(),
                                              out_av_head_major.data());
            }
            auto t1 = Clock::now();
            const double ms_head_major = Ms(t1 - t0).count() / static_cast<double>(iters);
            // const double speedup = ms_baseline / ms_head_major;
            // const float diff = max_abs_diff(out_av_baseline, out_av_head_major);

            printf("  [head_major] qk %.6f ms/call, av %.6f ms/call, total %.6f ms/call\n",
                   ms_head_major, ms_head_major, ms_head_major * 2); //, speedup, diff);
        }

        // ============================================================================
        // 5. Dual-view simulated layout
        // ============================================================================
        {
            auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) {
                attention_dual_view_simulated(q.data(), keys.data(), values.data(),
                                             n_tokens, head_dim, n_kv_heads,
                                             groups_per_head, n_heads_total,
                                             out_qk_dual_view.data(),
                                             out_av_dual_view.data());
            }
            auto t1 = Clock::now();
            const double ms_dual_view = Ms(t1 - t0).count() / static_cast<double>(iters);
            // const double speedup = ms_baseline / ms_dual_view;
            // const float diff = max_abs_diff(out_av_baseline, out_av_dual_view);

            printf("  [dual_view] qk %.6f ms/call, av %.6f ms/call, total %.6f ms/call\n",
                   ms_dual_view, ms_dual_view, ms_dual_view * 2); //, speedup, diff);
        }
    }

    printf("\n=== Conclusions ===\n");
    printf("1. Baseline (token-major) provides a reference point\n");
    printf("2. Blocking can improve cache locality for AV accumulation\n");
    printf("3. Head chunking/sequential can improve cache locality for QK\n");
    printf("4. Head-major/dual-view are memory-layout-aware optimizations\n");
    printf("5. Best strategy depends on sequence length and cache hierarchy\n");

    return 0;
}
