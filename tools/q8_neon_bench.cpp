// q8_neon_bench.cpp — benchmark NEON vs scalar Q8_0 matvec
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

    // Test on token_embd.weight which is Q8_0 in this model
    struct TestCase {
        const char *name;
        std::size_t rows;
        std::size_t cols;
        const unsigned char *data;
    };
    std::vector<TestCase> cases;

    {
        const auto *info = ml->tensor_index.find("token_embd.weight");
        const auto *view = ml->tensor_index.find_view("token_embd.weight");
        if (!info || !view) {
            std::fprintf(stderr, "FAIL: token_embd.weight not found\n");
            ml_model_free(ml);
            return 1;
        }
        const std::size_t cols = static_cast<std::size_t>(info->dims[0]);
        const std::size_t rows = static_cast<std::size_t>(info->dims[1]);
        std::vector<unsigned char> raw(static_cast<std::size_t>(view->byte_size));
        {
            std::ifstream file(model_path, std::ios::binary);
            file.seekg(static_cast<std::streamoff>(view->data_begin));
            file.read(reinterpret_cast<char*>(raw.data()), raw.size());
        }
        // Only take a portion for shorter benchmark
        const std::size_t subset_rows = std::min(rows, (std::size_t)576);
        std::printf("token_embd.weight: [%zu x %zu] (%zu KB raw)\n",
                    rows, cols, raw.size()/1024);

        // Copy subset of rows
        auto *subset = new unsigned char[subset_rows * (cols/32)*34];
        std::memcpy(subset, raw.data(), subset_rows * (cols/32)*34);
        cases.push_back({"token_embd.weight (subset 576 rows)", subset_rows, cols, subset});
    }

    for (auto &tc : cases) {
        std::vector<float> input(tc.cols);
        unsigned int seed = 42;
        for (auto &v : input) {
            seed = seed * 1103515245u + 12345u;
            v = (static_cast<float>(seed >> 16) / 65536.0f) * 2.0f - 1.0f;
        }
        std::vector<float> out(tc.rows);

        // Check correctness first
        std::vector<float> ref_out(tc.rows);
        minllama::matvec_q8_0_fused_f32(tc.data, tc.rows, tc.cols,
                                         input.data(), input.size(),
                                         ref_out.data(), ref_out.size());
        minllama::matvec_q8_0_neon_f32(tc.data, tc.rows, tc.cols,
                                        input.data(), input.size(),
                                        out.data(), out.size());

        double max_diff = 0;
        for (std::size_t i = 0; i < tc.rows; ++i) {
            double d = std::abs((double)ref_out[i] - (double)out[i]);
            if (d > max_diff) max_diff = d;
        }
        std::printf("\n%s: correctness max_diff=%.10f  %s\n",
                    tc.name, max_diff,
                    max_diff < 1e-4 ? "PASS" : "FAIL");

        // Warmup
        for (int w = 0; w < 3; ++w) {
            minllama::matvec_q8_0_neon_f32(tc.data, tc.rows, tc.cols,
                                            input.data(), input.size(),
                                            out.data(), out.size());
            minllama::matvec_q8_0_fused_f32(tc.data, tc.rows, tc.cols,
                                             input.data(), input.size(),
                                             out.data(), out.size());
        }

        int iters = 10;

        // Benchmark NEON
        double neon_total = 0;
        for (int i = 0; i < iters; ++i) {
            auto t0 = Clock::now();
            minllama::matvec_q8_0_neon_f32(tc.data, tc.rows, tc.cols,
                                            input.data(), input.size(),
                                            out.data(), out.size());
            neon_total += elapsed_ms(t0, Clock::now());
        }

        // Benchmark scalar fused
        double fused_total = 0;
        for (int i = 0; i < iters; ++i) {
            auto t0 = Clock::now();
            minllama::matvec_q8_0_fused_f32(tc.data, tc.rows, tc.cols,
                                             input.data(), input.size(),
                                             out.data(), out.size());
            fused_total += elapsed_ms(t0, Clock::now());
        }

        double neon_ms = neon_total / iters;
        double fused_ms = fused_total / iters;
        double ratio = fused_ms / neon_ms;

        std::printf("  scalar fused: %7.2f ms\n", fused_ms);
        std::printf("  NEON fused:   %7.2f ms\n", neon_ms);
        std::printf("  speedup:      %.2fx\n", ratio);
    }

    // Cleanup
    for (auto &tc : cases) {
        delete[] tc.data;
    }

    ml_model_free(ml);
    return 0;
}
