#include "minllama_internal.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

void fill_data(std::vector<float> &x, float scale = 1.0f) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int t = static_cast<int>((i * 37 + 17) % 211) - 105;
        x[i] = scale * static_cast<float>(t) / 53.0f;
    }
}

int iterations_for_seq(std::size_t seq) {
    if (seq <= 8) return 50000;
    if (seq <= 32) return 20000;
    if (seq <= 128) return 8000;
    if (seq <= 256) return 4000;
    return 1000;
}

} // namespace

int main() {
    const std::size_t head_dim = 64;
    const std::size_t seqs[] = {1, 8, 32, 128, 256, 1024};

    std::printf("attention_simd_bench head_dim=%zu\n", head_dim);
    std::printf("[qk] seq_len ms/call_scalar ms/call_neon speedup max_diff\n");

    for (std::size_t seq : seqs) {
        const int iters = iterations_for_seq(seq);
        std::vector<float> q(head_dim);
        std::vector<float> k(seq * head_dim);
        std::vector<float> out_scalar(seq, 0.0f);
        std::vector<float> out_neon(seq, 0.0f);
        fill_data(q, 1.0f);
        fill_data(k, 0.7f);

        auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            minllama::attention_qk_scores_scalar_f32(q.data(), head_dim, k.data(), seq, out_scalar.data(), out_scalar.size());
        }
        auto t1 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            minllama::attention_qk_scores_neon_f32(q.data(), head_dim, k.data(), seq, out_neon.data(), out_neon.size());
        }
        auto t2 = Clock::now();

        const double ms_scalar = Ms(t1 - t0).count() / static_cast<double>(iters);
        const double ms_neon = Ms(t2 - t1).count() / static_cast<double>(iters);
        const double speedup = (ms_neon > 0.0) ? (ms_scalar / ms_neon) : 0.0;
        const float diff = max_abs_diff(out_scalar, out_neon);

        std::printf("[qk] %4zu %.6f %.6f %.3fx %.8f\n", seq, ms_scalar, ms_neon, speedup, diff);
    }

    std::printf("[av] seq_len ms/call_scalar ms/call_neon speedup max_diff\n");

    for (std::size_t seq : seqs) {
        const int iters = iterations_for_seq(seq);
        std::vector<float> probs(seq, 1.0f / static_cast<float>(seq));
        std::vector<float> v(seq * head_dim);
        std::vector<float> out_scalar(head_dim, 0.0f);
        std::vector<float> out_neon(head_dim, 0.0f);
        fill_data(v, 0.9f);

        auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            minllama::attention_av_accum_scalar_f32(probs.data(), v.data(), seq, head_dim, out_scalar.data(), out_scalar.size());
        }
        auto t1 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            minllama::attention_av_accum_neon_f32(probs.data(), v.data(), seq, head_dim, out_neon.data(), out_neon.size());
        }
        auto t2 = Clock::now();

        const double ms_scalar = Ms(t1 - t0).count() / static_cast<double>(iters);
        const double ms_neon = Ms(t2 - t1).count() / static_cast<double>(iters);
        const double speedup = (ms_neon > 0.0) ? (ms_scalar / ms_neon) : 0.0;
        const float diff = max_abs_diff(out_scalar, out_neon);

        std::printf("[av] %4zu %.6f %.6f %.3fx %.8f\n", seq, ms_scalar, ms_neon, speedup, diff);
    }

    return 0;
}
