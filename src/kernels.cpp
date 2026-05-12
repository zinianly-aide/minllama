#include "minllama_internal.h"

#include <cmath>
#include <limits>
#include <vector>

namespace minllama {

namespace {
constexpr std::uint32_t kGgmlTypeF32 = 0;
constexpr std::uint32_t kGgmlTypeF16 = 1;
constexpr std::uint32_t kGgmlTypeQ4_0 = 2;

bool valid_matvec_args(std::size_t rows,
                       std::size_t cols,
                       const void *matrix,
                       const float *input,
                       std::size_t input_len,
                       float *out,
                       std::size_t out_len) {
    return rows > 0 &&
           cols > 0 &&
           matrix != nullptr &&
           input != nullptr &&
           out != nullptr &&
           input_len == cols &&
           out_len == rows &&
           rows <= std::numeric_limits<std::size_t>::max() / cols;
}

bool matvec_decoded_f32(const std::vector<float> &matrix,
                        std::size_t rows,
                        std::size_t cols,
                        const std::vector<float> &input,
                        std::vector<float> *out) {
    if (!out ||
        rows == 0 ||
        cols == 0 ||
        input.size() != cols ||
        matrix.size() != rows * cols) {
        return false;
    }

    out->assign(rows, 0.0f);
    return matvec_f32_f32(matrix.data(), rows, cols, input.data(), input.size(), out->data(), out->size());
}
} // namespace

bool matvec_f32_f32(const float *matrix,
                    std::size_t rows,
                    std::size_t cols,
                    const float *input,
                    std::size_t input_len,
                    float *out,
                    std::size_t out_len) {
    if (!valid_matvec_args(rows, cols, matrix, input, input_len, out, out_len)) {
        return false;
    }

    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        const std::size_t base = row * cols;
        for (std::size_t col = 0; col < cols; ++col) {
            sum += matrix[base + col] * input[col];
        }
        out[row] = sum;
    }
    return true;
}

bool matvec_f16_f32(const std::uint16_t *matrix,
                    std::size_t rows,
                    std::size_t cols,
                    const float *input,
                    std::size_t input_len,
                    float *out,
                    std::size_t out_len) {
    if (!valid_matvec_args(rows, cols, matrix, input, input_len, out, out_len)) {
        return false;
    }

    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        const std::size_t base = row * cols;
        for (std::size_t col = 0; col < cols; ++col) {
            sum += ml_fp16_to_fp32(matrix[base + col]) * input[col];
        }
        out[row] = sum;
    }
    return true;
}

bool matvec_tensor_as_f32(const char *path,
                          const TensorIndex &tensor_index,
                          const std::string &name,
                          const std::vector<float> &input,
                          std::vector<float> *out) {
    if (!path || !out) {
        return false;
    }

    const TensorInfo *info = tensor_index.find(name);
    if (!info ||
        info->n_dims != 2 ||
        (info->gguf_type != kGgmlTypeF32 &&
         info->gguf_type != kGgmlTypeF16 &&
         info->gguf_type != kGgmlTypeQ4_0) ||
        info->dims[0] > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        info->dims[1] > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }

    // GGUF/ggml stores matrix shape as ne[0] columns, ne[1] rows.
    const std::size_t cols = static_cast<std::size_t>(info->dims[0]);
    const std::size_t rows = static_cast<std::size_t>(info->dims[1]);
    if (rows == 0 ||
        cols == 0 ||
        input.size() != cols ||
        rows > std::numeric_limits<std::size_t>::max() / cols) {
        return false;
    }

    std::vector<float> matrix;
    if (!load_tensor_as_f32(path, tensor_index, name, &matrix)) {
        return false;
    }

    return matvec_decoded_f32(matrix, rows, cols, input, out);
}

bool matvec_q40_f32(const char *path,
                    const TensorIndex &tensor_index,
                    const std::string &name,
                    const std::vector<float> &input,
                    std::vector<float> *out) {
    const TensorInfo *info = tensor_index.find(name);
    if (!info || info->gguf_type != kGgmlTypeQ4_0) {
        return false;
    }
    return matvec_tensor_as_f32(path, tensor_index, name, input, out);
}

bool rmsnorm_f32(const float *x,
                 const float *weight,
                 std::size_t len,
                 float eps,
                 float *out,
                 std::size_t out_len) {
    if (!x || !weight || !out || len == 0 || out_len != len) {
        return false;
    }

    // Reference scalar RMSNorm:
    // y[i] = weight[i] * x[i] / sqrt(mean(x^2) + eps).
    float sum_sq = 0.0f;
    for (std::size_t i = 0; i < len; ++i) {
        sum_sq += x[i] * x[i];
    }

    const float mean_sq = sum_sq / static_cast<float>(len);
    const float scale = 1.0f / std::sqrt(mean_sq + eps);
    for (std::size_t i = 0; i < len; ++i) {
        out[i] = weight[i] * x[i] * scale;
    }
    return true;
}

bool softmax_f32(const float *input,
                 std::size_t len,
                 float *out,
                 std::size_t out_len) {
    if (!input || !out || len == 0 || out_len != len) {
        return false;
    }

    // Reference scalar softmax with max subtraction for numerical stability.
    float max_value = input[0];
    for (std::size_t i = 1; i < len; ++i) {
        if (input[i] > max_value) {
            max_value = input[i];
        }
    }

    float sum = 0.0f;
    for (std::size_t i = 0; i < len; ++i) {
        out[i] = std::exp(input[i] - max_value);
        sum += out[i];
    }

    if (sum == 0.0f) {
        return false;
    }
    const float inv_sum = 1.0f / sum;
    for (std::size_t i = 0; i < len; ++i) {
        out[i] *= inv_sum;
    }
    return true;
}

float silu_f32(float x) {
    // Reference scalar SiLU: x * sigmoid(x).
    return x / (1.0f + std::exp(-x));
}

bool swiglu_f32(const float *gate,
                const float *up,
                std::size_t len,
                float *out,
                std::size_t out_len) {
    if (!gate || !up || !out || len == 0 || out_len != len) {
        return false;
    }

    // Minimal reference SwiGLU: silu(gate) * up.
    for (std::size_t i = 0; i < len; ++i) {
        out[i] = silu_f32(gate[i]) * up[i];
    }
    return true;
}

bool rope_apply_f32(float *vec,
                    std::size_t len,
                    std::size_t position,
                    float rope_theta) {
    if (!vec || len == 0 || (len % 2) != 0 || rope_theta <= 0.0f) {
        return false;
    }

    // Reference Llama-style RoPE over even/odd pairs.
    const float half_len = static_cast<float>(len) / 2.0f;
    for (std::size_t i = 0; i < len; i += 2) {
        const float pair_index = static_cast<float>(i / 2);
        const float freq = std::pow(rope_theta, -pair_index / half_len);
        const float angle = static_cast<float>(position) * freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float x0 = vec[i];
        const float x1 = vec[i + 1];
        vec[i] = x0 * c - x1 * s;
        vec[i + 1] = x0 * s + x1 * c;
    }
    return true;
}

bool dot_f32(const float *a,
             const float *b,
             std::size_t len,
             float *out) {
    if (!a || !b || !out || len == 0) {
        return false;
    }

    float sum = 0.0f;
    for (std::size_t i = 0; i < len; ++i) {
        sum += a[i] * b[i];
    }
    *out = sum;
    return true;
}

bool scale_f32(float *vec,
               std::size_t len,
               float scale) {
    if (!vec || len == 0) {
        return false;
    }

    for (std::size_t i = 0; i < len; ++i) {
        vec[i] *= scale;
    }
    return true;
}

bool attention_scores_f32(const float *query,
                          std::size_t dim,
                          const float *keys,
                          std::size_t n_keys,
                          float *out_scores,
                          std::size_t out_len) {
    if (!query || !keys || !out_scores || dim == 0 || n_keys == 0 || out_len != n_keys) {
        return false;
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (std::size_t i = 0; i < n_keys; ++i) {
        float score = 0.0f;
        if (!dot_f32(query, keys + i * dim, dim, &score)) {
            return false;
        }
        out_scores[i] = score * scale;
    }
    return true;
}

bool attention_single_head_f32(const float *query,
                               const float *keys,
                               const float *values,
                               int n_tokens,
                               int dim,
                               float *output) {
    if (!query || !keys || !values || !output || n_tokens <= 0 || dim <= 0) {
        return false;
    }

    const std::size_t n = static_cast<std::size_t>(n_tokens);
    const std::size_t d = static_cast<std::size_t>(dim);

    // Step 1: scores[i] = dot(query, keys[i]) / sqrt(dim)
    std::vector<float> scores(n);
    if (!attention_scores_f32(query, d, keys, n, scores.data(), n)) {
        return false;
    }

    // Step 2: softmax over scores (in-place, safe because softmax_f32 reads
    // input before writing output element-by-element).
    if (!softmax_f32(scores.data(), n, scores.data(), n)) {
        return false;
    }

    // Step 3: output = sum_i softmax(scores)[i] * values[i]
    for (std::size_t j = 0; j < d; ++j) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            sum += scores[i] * values[i * d + j];
        }
        output[j] = sum;
    }

    return true;
}

bool kv_cache_init_f32(KvCacheF32 &cache, int max_tokens, int dim) {
    if (max_tokens <= 0 || dim <= 0) {
        return false;
    }

    cache.keys.assign(static_cast<std::size_t>(max_tokens) * static_cast<std::size_t>(dim), 0.0f);
    cache.values.assign(static_cast<std::size_t>(max_tokens) * static_cast<std::size_t>(dim), 0.0f);
    cache.max_tokens = max_tokens;
    cache.dim = dim;
    return true;
}

bool kv_cache_write_f32(KvCacheF32 &cache, int position, const float *key, const float *value) {
    if (!key || !value || position < 0 || position >= cache.max_tokens ||
        cache.max_tokens <= 0 || cache.dim <= 0) {
        return false;
    }

    const std::size_t d = static_cast<std::size_t>(cache.dim);
    const std::size_t offset = static_cast<std::size_t>(position) * d;
    std::copy(key, key + d, cache.keys.begin() + static_cast<long>(offset));
    std::copy(value, value + d, cache.values.begin() + static_cast<long>(offset));
    return true;
}

bool attention_decode_single_head_f32(const float *query,
                                      KvCacheF32 &cache,
                                      int position,
                                      float *output) {
    if (!query || !output || position < 0 || position >= cache.max_tokens ||
        cache.max_tokens <= 0 || cache.dim <= 0) {
        return false;
    }

    const int n_tokens = position + 1;
    return attention_single_head_f32(query,
                                     cache.keys.data(),
                                     cache.values.data(),
                                     n_tokens,
                                     cache.dim,
                                     output);
}

bool transformer_layer_decode_f32(const TransformerLayerF32 &layer,
                                  const float *x,
                                  KvCacheF32 &cache,
                                  int position,
                                  float *output) {
    if (!x || !output || layer.dim <= 0 || position < 0 ||
        position >= cache.max_tokens || cache.max_tokens <= 0 ||
        cache.dim <= 0 || cache.dim != layer.dim) {
        return false;
    }

    const int dim = layer.dim;
    const std::size_t d = static_cast<std::size_t>(dim);
    const std::size_t dd = d * d;

    // Validate attention weight sizes.
    if (layer.rms_att_weight.size() != d ||
        layer.wq.size() != dd ||
        layer.wk.size() != dd ||
        layer.wv.size() != dd ||
        layer.wo.size() != dd) {
        return false;
    }

    // -------------------------------------------------------------
    // Attention block (steps 1-7)
    // -------------------------------------------------------------

    // 1. RMSNorm
    std::vector<float> xn(dim);
    if (!rmsnorm_f32(x, layer.rms_att_weight.data(), d,
                      layer.rms_norm_eps, xn.data(), d)) {
        return false;
    }

    // 2. Q/K/V linear projections
    std::vector<float> q(dim), k(dim), v(dim);
    if (!matvec_f32_f32(layer.wq.data(), d, d, xn.data(), d, q.data(), d) ||
        !matvec_f32_f32(layer.wk.data(), d, d, xn.data(), d, k.data(), d) ||
        !matvec_f32_f32(layer.wv.data(), d, d, xn.data(), d, v.data(), d)) {
        return false;
    }

    // 3. RoPE on q and k
    if (!rope_apply_f32(q.data(), d, static_cast<std::size_t>(position),
                         layer.rope_theta) ||
        !rope_apply_f32(k.data(), d, static_cast<std::size_t>(position),
                         layer.rope_theta)) {
        return false;
    }

    // 4. Write k/v to cache
    if (!kv_cache_write_f32(cache, position, k.data(), v.data())) {
        return false;
    }

    // 5. Decode attention
    std::vector<float> att(dim);
    if (!attention_decode_single_head_f32(q.data(), cache, position,
                                          att.data())) {
        return false;
    }

    // 6. Output projection
    std::vector<float> projected(dim);
    if (!matvec_f32_f32(layer.wo.data(), d, d, att.data(), d,
                         projected.data(), d)) {
        return false;
    }

    // 7. Residual: h = x + projected (attention residual)
    std::vector<float> h(dim);
    for (int i = 0; i < dim; ++i) {
        h[i] = x[i] + projected[i];
    }

    // -------------------------------------------------------------
    // FFN block (steps 8-13, optional: skipped when hidden_dim <= 0)
    // -------------------------------------------------------------
    const int hdim = layer.hidden_dim;
    if (hdim <= 0) {
        for (int i = 0; i < dim; ++i) {
            output[i] = h[i];
        }
        return true;
    }

    const std::size_t hd = static_cast<std::size_t>(hdim);
    const std::size_t hd_d = hd * d;
    const std::size_t d_hd = d * hd;

    // Validate FFN weight sizes.
    if (layer.rms_ffn_weight.size() != d ||
        layer.w1.size() != hd_d ||
        layer.w2.size() != d_hd ||
        layer.w3.size() != hd_d) {
        return false;
    }

    // 8. Second RMSNorm on h
    std::vector<float> norm_h(dim);
    if (!rmsnorm_f32(h.data(), layer.rms_ffn_weight.data(), d,
                      layer.rms_norm_eps, norm_h.data(), d)) {
        return false;
    }

    // 9. gate = W1 * norm_h
    std::vector<float> gate(hdim);
    if (!matvec_f32_f32(layer.w1.data(), hd, d, norm_h.data(), d,
                         gate.data(), hd)) {
        return false;
    }

    // 10. up = W3 * norm_h
    std::vector<float> up(hdim);
    if (!matvec_f32_f32(layer.w3.data(), hd, d, norm_h.data(), d,
                         up.data(), hd)) {
        return false;
    }

    // 11. hidden = silu(gate) * up
    std::vector<float> hidden(hdim);
    if (!swiglu_f32(gate.data(), up.data(), hd, hidden.data(), hd)) {
        return false;
    }

    // 12. ffn_out = W2 * hidden
    std::vector<float> ffn_out(dim);
    if (!matvec_f32_f32(layer.w2.data(), d, hd, hidden.data(), hd,
                         ffn_out.data(), d)) {
        return false;
    }

    // 13. output = h + ffn_out
    for (int i = 0; i < dim; ++i) {
        output[i] = h[i] + ffn_out[i];
    }

    return true;
}

bool transformer_model_decode_f32(TransformerModelF32 &model,
                                  const float *x,
                                  int position,
                                  float *output) {
    if (!x || !output || model.dim <= 0 || model.n_layers <= 0) {
        return false;
    }

    const int dim = model.dim;
    const int n = model.n_layers;

    if (static_cast<int>(model.layers.size()) != n ||
        static_cast<int>(model.kv_caches.size()) != n) {
        return false;
    }

    // Pre-validate layers and caches.
    for (int i = 0; i < n; ++i) {
        if (model.layers[i].dim != dim ||
            model.kv_caches[i].dim != dim) {
            return false;
        }
        if (position < 0 || position >= model.kv_caches[i].max_tokens ||
            model.kv_caches[i].max_tokens <= 0 ||
            model.kv_caches[i].dim <= 0) {
            return false;
        }
    }

    std::vector<float> curr(dim);
    std::vector<float> next(dim);

    // Layer 0 input = x.
    for (int j = 0; j < dim; ++j) {
        curr[j] = x[j];
    }

    for (int i = 0; i < n; ++i) {
        float *layer_out = (i == n - 1) ? output : next.data();
        if (!transformer_layer_decode_f32(model.layers[i], curr.data(),
                                          model.kv_caches[i], position,
                                          layer_out)) {
            return false;
        }
        if (i < n - 1) {
            curr.swap(next);
        }
    }

    return true;
}

bool transformer_model_logits_f32(TransformerModelF32 &model,
                                  const float *x,
                                  int position,
                                  float *logits) {
    if (!x || !logits || model.dim <= 0 || model.vocab_size <= 0) {
        return false;
    }

    const int dim = model.dim;
    const int vocab = model.vocab_size;
    const std::size_t d = static_cast<std::size_t>(dim);
    const std::size_t v = static_cast<std::size_t>(vocab);

    if (model.final_norm_weight.size() != d ||
        model.lm_head.size() != v * d) {
        return false;
    }

    // 1. Forward through all transformer layers.
    std::vector<float> hidden(dim);
    if (!transformer_model_decode_f32(model, x, position, hidden.data())) {
        return false;
    }

    // 2. Final RMSNorm.
    std::vector<float> norm_hidden(dim);
    if (!rmsnorm_f32(hidden.data(), model.final_norm_weight.data(), d,
                      model.rms_norm_eps, norm_hidden.data(), d)) {
        return false;
    }

    // 3. LM head projection: logits = lm_head * norm_hidden.
    if (!matvec_f32_f32(model.lm_head.data(), v, d, norm_hidden.data(), d,
                         logits, v)) {
        return false;
    }

    return true;
}

int argmax_f32(const float *values, int len) {
    if (!values || len <= 0) {
        return -1;
    }

    int best = 0;
    float best_val = values[0];
    for (int i = 1; i < len; ++i) {
        if (values[i] > best_val) {
            best = i;
            best_val = values[i];
        }
    }
    return best;
}

bool transformer_model_greedy_step_f32(TransformerModelF32 &model,
                                       const float *x,
                                       int position,
                                       int *token_id) {
    if (!token_id) {
        return false;
    }

    const int vocab = model.vocab_size;
    if (vocab <= 0) {
        return false;
    }

    std::vector<float> logits(vocab);
    if (!transformer_model_logits_f32(model, x, position, logits.data())) {
        return false;
    }

    const int tid = argmax_f32(logits.data(), vocab);
    if (tid < 0) {
        return false;
    }

    *token_id = tid;
    return true;
}

uint32_t rng_next_u32(uint32_t *state) {
    if (!state) return 0;
    // Minimal LCG: glibc-style constants.
    *state = *state * 1103515245u + 12345u;
    return *state;
}

float rng_uniform01(uint32_t *state) {
    if (!state) return 0.0f;
    const uint32_t v = rng_next_u32(state);
    // Convert to [0, 1) using high bits for better uniformity.
    return static_cast<float>(v >> 8) / 16777216.0f; // 2^24
}

int sample_temperature_f32(const float *logits,
                           int vocab_size,
                           float temperature,
                           uint32_t *rng_state) {
    if (!logits || !rng_state || vocab_size <= 0) {
        return -1;
    }

    // temperature <= 0: greedy.
    if (temperature <= 0.0f) {
        return argmax_f32(logits, vocab_size);
    }

    // Find max for numerical stability.
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Compute scaled probabilities.
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        const float v = std::exp((logits[i] - max_logit) / temperature);
        probs[i] = v;
        sum += v;
    }

    if (sum <= 0.0f) {
        return argmax_f32(logits, vocab_size);
    }

    // Normalize.
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    // Sample via CDF.
    const float u = rng_uniform01(rng_state);
    float cumulative = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cumulative += probs[i];
        if (u < cumulative) {
            return i;
        }
    }

    // Fallback (floating-point edge).
    return vocab_size - 1;
}

bool transformer_model_sample_step_f32(TransformerModelF32 &model,
                                       const float *x,
                                       int position,
                                       float temperature,
                                       uint32_t *rng_state,
                                       int *token_id) {
    if (!token_id || !rng_state) {
        return false;
    }

    const int vocab = model.vocab_size;
    if (vocab <= 0) {
        return false;
    }

    std::vector<float> logits(vocab);
    if (!transformer_model_logits_f32(model, x, position, logits.data())) {
        return false;
    }

    const int tid = sample_temperature_f32(logits.data(), vocab, temperature,
                                            rng_state);
    if (tid < 0) {
        return false;
    }

    *token_id = tid;
    return true;
}

bool token_embedding_lookup_f32(const TransformerModelF32 &model,
                                int token_id,
                                float *output) {
    if (!output) {
        return false;
    }

    const int dim = model.dim;
    const int vocab_size = model.vocab_size;

    if (dim <= 0 || vocab_size <= 0) {
        return false;
    }

    const std::size_t d = static_cast<std::size_t>(dim);
    const std::size_t v = static_cast<std::size_t>(vocab_size);

    if (model.token_embedding.size() != v * d) {
        return false;
    }

    if (token_id < 0 || token_id >= vocab_size) {
        return false;
    }

    const float *src = model.token_embedding.data() + static_cast<std::size_t>(token_id) * d;
    for (std::size_t i = 0; i < d; ++i) {
        output[i] = src[i];
    }

    return true;
}

bool transformer_model_greedy_token_step_f32(TransformerModelF32 &model,
                                             int token_id,
                                             int position,
                                             int *next_token_id) {
    if (!next_token_id) {
        return false;
    }

    const int dim = model.dim;
    if (dim <= 0) {
        return false;
    }

    // 1. Lookup token embedding.
    std::vector<float> x(dim);
    if (!token_embedding_lookup_f32(model, token_id, x.data())) {
        return false;
    }

    // 2. Forward through transformer model.
    if (!transformer_model_greedy_step_f32(model, x.data(), position,
                                            next_token_id)) {
        return false;
    }

    return true;
}

bool transformer_model_generate_greedy_f32(TransformerModelF32 &model,
                                           const int *prompt_tokens,
                                           int prompt_len,
                                           int max_new_tokens,
                                           int *output_tokens,
                                           int output_capacity,
                                           int *output_len,
                                           int eos_token_id) {
    if (!prompt_tokens || !output_tokens || !output_len) {
        return false;
    }

    if (prompt_len <= 0) {
        return false;
    }

    if (max_new_tokens < 0) {
        return false;
    }

    if (output_capacity < max_new_tokens) {
        return false;
    }

    // Validate eos_token_id.
    if (eos_token_id >= model.vocab_size) {
        return false;
    }

    // KV cache is assumed to be already initialized by the caller.
    // Prefill: process all prompt tokens in order.
    int next_token = -1;
    for (int i = 0; i < prompt_len; ++i) {
        int tid = -1;
        if (!transformer_model_greedy_token_step_f32(
                model, prompt_tokens[i], i, &tid)) {
            return false;
        }
        if (i == prompt_len - 1) {
            next_token = tid;
        }
    }

    // If max_new_tokens == 0, we only prefill and return.
    if (max_new_tokens == 0) {
        *output_len = 0;
        return true;
    }

    // Generate: produce new tokens one by one.
    int generated = 0;
    int cur_token = next_token;
    for (int i = 0; i < max_new_tokens; ++i) {
        output_tokens[generated] = cur_token;
        ++generated;

        // EOS check: stop after writing the eos token.
        if (eos_token_id >= 0 && cur_token == eos_token_id) {
            *output_len = generated;
            return true;
        }

        if (i == max_new_tokens - 1) {
            break;
        }

        int pos = prompt_len + i;
        int nid = -1;
        if (!transformer_model_greedy_token_step_f32(
                model, cur_token, pos, &nid)) {
            return false;
        }
        cur_token = nid;
    }

    *output_len = generated;
    return true;
}

bool transformer_model_generate_sample_f32(TransformerModelF32 &model,
                                           const int *prompt_tokens,
                                           int prompt_len,
                                           int max_new_tokens,
                                           int *output_tokens,
                                           int output_capacity,
                                           int *output_len,
                                           int eos_token_id,
                                           float temperature,
                                           uint32_t *rng_state) {
    if (!prompt_tokens || !output_tokens || !output_len || !rng_state) {
        return false;
    }

    if (prompt_len <= 0) {
        return false;
    }

    if (max_new_tokens < 0) {
        return false;
    }

    if (output_capacity < max_new_tokens) {
        return false;
    }

    if (eos_token_id >= model.vocab_size) {
        return false;
    }

    // Prefill: process all prompt tokens in order (use greedy for KV cache fill).
    int next_token = -1;
    for (int i = 0; i < prompt_len; ++i) {
        int tid = -1;
        if (!transformer_model_greedy_token_step_f32(
                model, prompt_tokens[i], i, &tid)) {
            return false;
        }
        if (i == prompt_len - 1) {
            next_token = tid;
        }
    }

    if (max_new_tokens == 0) {
        *output_len = 0;
        return true;
    }

    // Generate: sample new tokens one by one.
    int generated = 0;
    int cur_token = next_token;
    for (int i = 0; i < max_new_tokens; ++i) {
        output_tokens[generated] = cur_token;
        ++generated;

        // EOS check.
        if (eos_token_id >= 0 && cur_token == eos_token_id) {
            *output_len = generated;
            return true;
        }

        if (i == max_new_tokens - 1) {
            break;
        }

        int pos = prompt_len + i;
        int nid = -1;
        // Sample next token using the current token's embedding.
        std::vector<float> x(model.dim);
        if (!token_embedding_lookup_f32(model, cur_token, x.data())) {
            return false;
        }
        if (!transformer_model_sample_step_f32(model, x.data(), pos,
                                                temperature, rng_state, &nid)) {
            return false;
        }
        cur_token = nid;
    }

    *output_len = generated;
    return true;
}

bool load_transformer_model_f32_from_tensors(const ml_model &src,
                                             TransformerModelF32 &model,
                                             std::string *error) {
    auto set_error = [error](const std::string &msg) {
        if (error) *error = msg;
    };

    const auto &config = src.config;
    const auto &tensor_index = src.tensor_index;
    const char *path = src.path.c_str();

    // Validate metadata.
    if (config.n_embd == 0 || config.n_layer == 0 || config.n_vocab == 0 ||
        config.n_ctx_train == 0) {
        set_error("Missing required metadata (n_embd, n_layer, n_vocab, n_ctx_train)");
        return false;
    }

    const int dim = static_cast<int>(config.n_embd);
    const int n_layers = static_cast<int>(config.n_layer);
    const int vocab_size = static_cast<int>(config.n_vocab);
    const int context_length = static_cast<int>(config.n_ctx_train);
    const float rope_theta = config.rope_theta > 0.0f ? config.rope_theta : 10000.0f;

    // Derive hidden_dim from ffn_gate.weight dimensions.
    int hidden_dim = 0;
    {
        const TensorInfo *info = tensor_index.find("blk.0.ffn_gate.weight");
        if (info && info->n_dims == 2 && info->dims[1] == static_cast<std::uint64_t>(dim)) {
            hidden_dim = static_cast<int>(info->dims[0]);
        }
    }
    if (hidden_dim <= 0) {
        hidden_dim = dim; // fallback
    }

    // Initialize model.
    model.dim = dim;
    model.n_layers = n_layers;
    model.vocab_size = vocab_size;
    model.rms_norm_eps = config.rms_norm_eps > 0.0f ? config.rms_norm_eps : 1e-6f;
    model.layers.clear();
    model.layers.resize(n_layers);
    model.kv_caches.clear();
    model.kv_caches.resize(n_layers);

    for (int i = 0; i < n_layers; ++i) {
        model.layers[i].dim = dim;
        model.layers[i].hidden_dim = hidden_dim;
        model.layers[i].rope_theta = rope_theta;
        model.layers[i].rms_norm_eps = model.rms_norm_eps;
        if (!kv_cache_init_f32(model.kv_caches[i], context_length, dim)) {
            set_error("Failed to init KV cache for layer " + std::to_string(i));
            return false;
        }
    }

    // Helper to load a tensor by name.
    auto load_tensor = [&](const std::string &name,
                           std::vector<float> *out) -> bool {
        if (!load_tensor_as_f32(path, tensor_index, name, out)) {
            set_error("Missing or unreadable tensor: " + name);
            return false;
        }
        return true;
    };

    // Helper to validate tensor shape.
    auto check_shape = [&](const std::string &name,
                           const std::vector<float> &data,
                           const std::vector<std::uint64_t> &expected,
                           const std::string &desc) -> bool {
        const TensorInfo *info = tensor_index.find(name);
        if (!info) {
            set_error("Tensor not in index: " + name);
            return false;
        }
        if (info->n_dims != expected.size()) {
            set_error("Bad ndims for " + name + ": expected " +
                      std::to_string(expected.size()) + " got " +
                      std::to_string(info->n_dims));
            return false;
        }
        for (std::size_t j = 0; j < expected.size(); ++j) {
            if (info->dims[j] != expected[j]) {
                set_error("Bad shape for " + name + " dim " +
                          std::to_string(j) + ": expected " +
                          std::to_string(expected[j]) + " got " +
                          std::to_string(info->dims[j]));
                return false;
            }
        }
        return true;
    };

    // Load global tensors.
    if (!load_tensor("token_embd.weight", &model.token_embedding)) return false;
    if (!check_shape("token_embd.weight", model.token_embedding,
                     {static_cast<std::uint64_t>(vocab_size), static_cast<std::uint64_t>(dim)},
                     "token_embd.weight")) return false;

    if (!load_tensor("output_norm.weight", &model.final_norm_weight)) return false;
    if (!check_shape("output_norm.weight", model.final_norm_weight,
                     {static_cast<std::uint64_t>(dim)},
                     "output_norm.weight")) return false;

    if (!load_tensor("output.weight", &model.lm_head)) return false;
    if (!check_shape("output.weight", model.lm_head,
                     {static_cast<std::uint64_t>(vocab_size), static_cast<std::uint64_t>(dim)},
                     "output.weight")) return false;

    // Load per-layer tensors.
    for (int i = 0; i < n_layers; ++i) {
        auto &layer = model.layers[i];
        const std::string prefix = "blk." + std::to_string(i);

        if (!load_tensor(prefix + ".attn_norm.weight", &layer.rms_att_weight)) return false;
        if (!check_shape(prefix + ".attn_norm.weight", layer.rms_att_weight,
                         {static_cast<std::uint64_t>(dim)},
                         prefix + ".attn_norm.weight")) return false;

        if (!load_tensor(prefix + ".attn_q.weight", &layer.wq)) return false;
        if (!check_shape(prefix + ".attn_q.weight", layer.wq,
                         {static_cast<std::uint64_t>(dim), static_cast<std::uint64_t>(dim)},
                         prefix + ".attn_q.weight")) return false;

        if (!load_tensor(prefix + ".attn_k.weight", &layer.wk)) return false;
        if (!check_shape(prefix + ".attn_k.weight", layer.wk,
                         {static_cast<std::uint64_t>(dim), static_cast<std::uint64_t>(dim)},
                         prefix + ".attn_k.weight")) return false;

        if (!load_tensor(prefix + ".attn_v.weight", &layer.wv)) return false;
        if (!check_shape(prefix + ".attn_v.weight", layer.wv,
                         {static_cast<std::uint64_t>(dim), static_cast<std::uint64_t>(dim)},
                         prefix + ".attn_v.weight")) return false;

        if (!load_tensor(prefix + ".attn_output.weight", &layer.wo)) return false;
        if (!check_shape(prefix + ".attn_output.weight", layer.wo,
                         {static_cast<std::uint64_t>(dim), static_cast<std::uint64_t>(dim)},
                         prefix + ".attn_output.weight")) return false;

        if (!load_tensor(prefix + ".ffn_norm.weight", &layer.rms_ffn_weight)) return false;
        if (!check_shape(prefix + ".ffn_norm.weight", layer.rms_ffn_weight,
                         {static_cast<std::uint64_t>(dim)},
                         prefix + ".ffn_norm.weight")) return false;

        if (!load_tensor(prefix + ".ffn_gate.weight", &layer.w1)) return false;
        if (!check_shape(prefix + ".ffn_gate.weight", layer.w1,
                         {static_cast<std::uint64_t>(hidden_dim), static_cast<std::uint64_t>(dim)},
                         prefix + ".ffn_gate.weight")) return false;

        if (!load_tensor(prefix + ".ffn_down.weight", &layer.w2)) return false;
        if (!check_shape(prefix + ".ffn_down.weight", layer.w2,
                         {static_cast<std::uint64_t>(dim), static_cast<std::uint64_t>(hidden_dim)},
                         prefix + ".ffn_down.weight")) return false;

        if (!load_tensor(prefix + ".ffn_up.weight", &layer.w3)) return false;
        if (!check_shape(prefix + ".ffn_up.weight", layer.w3,
                         {static_cast<std::uint64_t>(hidden_dim), static_cast<std::uint64_t>(dim)},
                         prefix + ".ffn_up.weight")) return false;
    }

    return true;
}
}
