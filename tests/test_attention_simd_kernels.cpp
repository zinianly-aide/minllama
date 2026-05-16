#include "minllama_internal.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

float l2_norm(const std::vector<float> &x) {
    double s = 0.0;
    for (float v : x) s += static_cast<double>(v) * static_cast<double>(v);
    return static_cast<float>(std::sqrt(s));
}

float cosine_sim(const std::vector<float> &a, const std::vector<float> &b) {
    assert(a.size() == b.size());
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na == 0.0 || nb == 0.0) return 1.0f;
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    assert(a.size() == b.size());
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
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

void check_case(std::size_t seq_len, std::size_t head_dim) {
    std::vector<float> q(head_dim);
    std::vector<float> k(seq_len * head_dim);
    std::vector<float> probs(seq_len);
    std::vector<float> v(seq_len * head_dim);

    fill_data(q, 1.0f);
    fill_data(k, 0.7f);
    fill_data(v, 0.9f);
    for (std::size_t i = 0; i < seq_len; ++i) {
        probs[i] = 1.0f / static_cast<float>(seq_len);
    }

    std::vector<float> qk_scalar(seq_len, 0.0f);
    std::vector<float> qk_neon(seq_len, 0.0f);
    assert(minllama::attention_qk_scores_scalar_f32(q.data(), head_dim, k.data(), seq_len, qk_scalar.data(), qk_scalar.size()));
    assert(minllama::attention_qk_scores_neon_f32(q.data(), head_dim, k.data(), seq_len, qk_neon.data(), qk_neon.size()));

    const float qk_diff = max_abs_diff(qk_scalar, qk_neon);
    const float qk_cos = cosine_sim(qk_scalar, qk_neon);
    assert(qk_diff < 1e-4f);
    assert(qk_cos > 0.99999f);

    std::vector<float> av_scalar(head_dim, 0.0f);
    std::vector<float> av_neon(head_dim, 0.0f);
    assert(minllama::attention_av_accum_scalar_f32(probs.data(), v.data(), seq_len, head_dim, av_scalar.data(), av_scalar.size()));
    assert(minllama::attention_av_accum_neon_f32(probs.data(), v.data(), seq_len, head_dim, av_neon.data(), av_neon.size()));

    const float av_diff = max_abs_diff(av_scalar, av_neon);
    const float av_cos = cosine_sim(av_scalar, av_neon);
    assert(av_diff < 1e-4f);
    assert(av_cos > 0.99999f);
}

} // namespace

int main() {
    const std::size_t head_dim = 64;
    const std::size_t seqs[] = {1, 8, 32, 128, 256};
    for (std::size_t s : seqs) {
        check_case(s, head_dim);
    }
    return 0;
}
