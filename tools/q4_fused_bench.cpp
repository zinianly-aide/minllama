// q4_fused_bench.cpp — benchmark fused vs scalar Q4_0 matvec
#include "minllama.h"
#include "minllama_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<Ms>(end - start).count();
}

int main(int argc, char **argv) {
    const char *model_path = "models/SmolLM-135M.Q4_0.gguf";
    if (argc > 1) model_path = argv[1];

    ml_model *ml = ml_model_load(model_path);
    if (!ml) { std::fprintf(stderr, "FAIL: load model\n"); return 1; }

    const char *tensors[] = {
        "blk.0.attn_q.weight",    // [576, 576] = 331,776 elements
        "blk.0.ffn_gate.weight",  // [1536, 576] = 884,736 elements
        "blk.0.attn_output.weight" // [576, 576] = 331,776 elements
    };

    for (const char *tname : tensors) {
        const auto *info = ml->tensor_index.find(tname);
        if (!info || info->gguf_type != 2) continue;

        const std::size_t cols = static_cast<std::size_t>(info->dims[0]);
        const std::size_t rows = static_cast<std::size_t>(info->dims[1]);
        const auto *view = ml->tensor_index.find_view(tname);

        // Load raw Q4_0 bytes
        std::vector<unsigned char> raw_q4(static_cast<std::size_t>(view->byte_size));
        {
            std::ifstream file(model_path, std::ios::binary);
            if (!file) { std::fprintf(stderr, "FAIL: open file\n"); continue; }
            file.seekg(static_cast<std::streamoff>(view->data_begin));
            file.read(reinterpret_cast<char*>(raw_q4.data()),
                      static_cast<std::streamsize>(raw_q4.size()));
        }

        // Generate random input
        std::vector<float> input(cols);
        unsigned int seed = 42;
        for (auto &v : input) {
            seed = seed * 1103515245u + 12345u;
            v = (static_cast<float>(seed >> 16) / 65536.0f) * 2.0f - 1.0f;
        }

        std::vector<float> out(rows);
        int warmup = 3;
        int iters = 20;

        // Warmup
        for (int w = 0; w < warmup; ++w) {
            minllama::matvec_q4_0_fused_f32(raw_q4.data(), rows, cols,
                                              input.data(), input.size(),
                                              out.data(), out.size());
        }

        // Benchmark fused
        double fused_total = 0;
        for (int i = 0; i < iters; ++i) {
            auto t0 = Clock::now();
            minllama::matvec_q4_0_fused_f32(raw_q4.data(), rows, cols,
                                              input.data(), input.size(),
                                              out.data(), out.size());
            fused_total += elapsed_ms(t0, Clock::now());
        }

        // Benchmark scalar (dequant + f32 matvec)
        double scalar_total = 0;
        for (int i = 0; i < iters; ++i) {
            std::vector<float> matrix;
            minllama::load_tensor_as_f32(model_path, ml->tensor_index, tname, &matrix);
            auto t0 = Clock::now();
            minllama::matvec_f32_f32(matrix.data(), rows, cols,
                                      input.data(), input.size(),
                                      out.data(), out.size());
            scalar_total += elapsed_ms(t0, Clock::now());
        }

        double fused_ms = fused_total / iters;
        double scalar_ms = scalar_total / iters;
        double ratio = scalar_ms / fused_ms;

        std::printf("%-30s [%4zux%-4zu] scalar=%7.2fms  fused=%7.2fms  speedup=%.2fx\n",
                    tname, rows, cols, scalar_ms, fused_ms, ratio);
    }

    ml_model_free(ml);
    return 0;
}
