// test_q4_fused_matvec.cpp
// Verify that matvec_q4_0_fused_f32 produces identical results to the
// scalar reference (dequant → f32 matvec).

#include "minllama.h"
#include "minllama_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

// Forward declare: helper to load raw Q4_0 data bytes
static bool load_q4_0_raw_bytes(const char *path,
                                 const TensorIndex &tensor_index,
                                 const std::string &name,
                                 std::vector<unsigned char> *out,
                                 std::size_t *out_rows,
                                 std::size_t *out_cols);

// Helper: load a Q4_0 tensor as f32 via scalar dequant path
static bool load_as_f32_and_matvec(const char *path,
                                    const TensorIndex &tensor_index,
                                    const std::string &name,
                                    const std::vector<float> &input,
                                    std::vector<float> *out) {
    const auto *info = tensor_index.find(name);
    if (!info || info->n_dims != 2) return false;
    const std::size_t cols = static_cast<std::size_t>(info->dims[0]);
    const std::size_t rows = static_cast<std::size_t>(info->dims[1]);

    std::vector<float> matrix;
    if (!minllama::load_tensor_as_f32(path, tensor_index, name, &matrix))
        return false;

    out->resize(rows);
    return minllama::matvec_f32_f32(matrix.data(), rows, cols,
                                     input.data(), input.size(),
                                     out->data(), out->size());
}

int main(int argc, char **argv) {
    const char *model_path = "../models/SmolLM-135M.Q4_0.gguf";
    if (argc > 1) model_path = argv[1];
    if (argc > 1 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        std::printf("Usage: test_q4_fused_matvec [model_path]\n");
        return 0;
    }

    ml_model *ml = ml_model_load(model_path);
    if (!ml) { std::fprintf(stderr, "FAIL: load model\n"); return 1; }

    // Load one Q4_0 weight tensor to test
    // token_embd.weight is Q8_0 (not Q4_0), use a layer weight
    const std::string test_tensor = "blk.0.attn_q.weight";
    const auto *info = ml->tensor_index.find(test_tensor);
    if (!info || info->gguf_type != 2) {  // 2 = Q4_0
        std::fprintf(stderr, "FAIL: tensor %s not Q4_0 (type=%u)\n",
                     test_tensor.c_str(), info ? info->gguf_type : 999);
        ml_model_free(ml);
        return 1;
    }

    const std::size_t cols = static_cast<std::size_t>(info->dims[0]);
    const std::size_t rows = static_cast<std::size_t>(info->dims[1]);
    std::printf("Tensor: %s  shape=[%zu, %zu]  type=Q4_0\n",
                test_tensor.c_str(), rows, cols);

    // Create random input
    std::vector<float> input(cols);
    unsigned int seed = 42;
    for (auto &v : input) {
        seed = seed * 1103515245u + 12345u;
        v = (static_cast<float>(seed >> 16) / 65536.0f) * 2.0f - 1.0f;
    }

    // Path A: scalar reference (dequant → f32 matvec)
    std::vector<float> ref_out;
    if (!load_as_f32_and_matvec(model_path, ml->tensor_index, test_tensor, input, &ref_out)) {
        std::fprintf(stderr, "FAIL: reference matvec\n");
        ml_model_free(ml);
        return 1;
    }

    // Path B: fused Q4_0 matvec
    std::vector<unsigned char> raw_q4;
    std::size_t raw_rows = 0, raw_cols = 0;
    if (!load_q4_0_raw_bytes(model_path, ml->tensor_index, test_tensor,
                              &raw_q4, &raw_rows, &raw_cols)) {
        std::fprintf(stderr, "FAIL: load raw Q4_0 bytes\n");
        ml_model_free(ml);
        return 1;
    }

    std::vector<float> fused_out(raw_rows);
    if (!minllama::matvec_q4_0_fused_f32(raw_q4.data(), raw_rows, raw_cols,
                                          input.data(), input.size(),
                                          fused_out.data(), fused_out.size())) {
        std::fprintf(stderr, "FAIL: fused matvec\n");
        ml_model_free(ml);
        return 1;
    }

    // Compare
    double sum_sq = 0.0, sum_ref_sq = 0.0;
    float max_diff = 0.0f;
    int max_diff_idx = -1;
    bool all_pass = true;
    for (std::size_t i = 0; i < rows; ++i) {
        float d = std::abs(ref_out[i] - fused_out[i]);
        sum_sq += static_cast<double>(d) * d;
        sum_ref_sq += static_cast<double>(ref_out[i]) * ref_out[i];
        if (d > max_diff) { max_diff = d; max_diff_idx = static_cast<int>(i); }
        if (d > 0.05f) {
            if (all_pass) {
                std::printf("  FAIL at [%zu]: ref=%.6f fused=%.6f diff=%.6f\n", i, ref_out[i], fused_out[i], d);
            }
            all_pass = false;
        }
    }

    double rmse = std::sqrt(sum_sq / rows);
    double norm_ref = std::sqrt(sum_ref_sq);
    double cosine = (norm_ref > 0) ? (1.0 - sum_sq / sum_ref_sq) : 1.0;

    std::printf("\n--- Results ---\n");
    std::printf("RMSE:         %.10f\n", rmse);
    std::printf("Max diff:     %.6f at index %d\n", max_diff, max_diff_idx);
    std::printf("Cosine sim:   %.10f\n", cosine);

    bool pass = true;
    if (max_diff > 0.1f) {
        std::printf("FAIL: max diff > 0.1\n");
        pass = false;
    }
    if (cosine < 0.99) {
        std::printf("FAIL: cosine similarity < 0.99\n");
        pass = false;
    }

    if (pass) {
        std::printf("\nALL PASS: fused matvec matches reference within tolerance\n");
    }

    ml_model_free(ml);
    return pass ? 0 : 1;
}

// Implementation of load_q4_0_raw_bytes
static bool load_q4_0_raw_bytes(const char *path,
                                 const TensorIndex &tensor_index,
                                 const std::string &name,
                                 std::vector<unsigned char> *out,
                                 std::size_t *out_rows,
                                 std::size_t *out_cols) {
    const auto *info = tensor_index.find(name);
    const auto *view = tensor_index.find_view(name);
    if (!info || !view || info->gguf_type != 2) return false;

    *out_cols = static_cast<std::size_t>(info->dims[0]);
    *out_rows = static_cast<std::size_t>(info->dims[1]);

    std::size_t byte_size = static_cast<std::size_t>(view->byte_size);
    out->resize(byte_size);

    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    file.seekg(static_cast<std::streamoff>(view->data_begin));
    file.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(byte_size));
    return file.good();
}
