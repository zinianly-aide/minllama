// q8_neon_bench2.cpp — benchmark NEON vs scalar Q8_0 matvec on ALL sizes
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

    // Collect Q8_0 tensors
    struct TensorData {
        std::string name;
        std::size_t rows, cols;
        std::vector<unsigned char> raw;
    };
    std::vector<TensorData> tensors;

    for (std::size_t i = 0; i < ml->tensor_index.size(); ++i) {
        const auto &info = ml->tensor_index.tensors[i];
        if (info.gguf_type == 8) { // Q8_0
            const auto *view = ml->tensor_index.find_view(info.name);
            if (!view) continue;
            const std::size_t cols = static_cast<std::size_t>(info.dims[0]);
            const std::size_t rows = static_cast<std::size_t>(info.dims[1]);
            TensorData td;
            td.name = info.name;
            td.rows = rows;
            td.cols = cols;
            td.raw.resize(static_cast<std::size_t>(view->byte_size));
            std::ifstream file(model_path, std::ios::binary);
            file.seekg(static_cast<std::streamoff>(view->data_begin));
            file.read(reinterpret_cast<char*>(td.raw.data()), td.raw.size());
            tensors.push_back(std::move(td));
        }
    }

    if (tensors.empty()) {
        std::printf("No Q8_0 tensors found in model\n");
        ml_model_free(ml);
        return 0;
    }

    std::printf("Found %zu Q8_0 tensor(s):\n\n", tensors.size());

    for (auto &td : tensors) {
        std::vector<float> input(td.cols);
        unsigned int seed = 42;
        for (auto &v : input) {
            seed = seed * 1103515245u + 12345u;
            v = (static_cast<float>(seed >> 16) / 65536.0f) * 2.0f - 1.0f;
        }
        std::vector<float> out(td.rows);

        // Correctness
        std::vector<float> ref(td.rows);
        minllama::matvec_q8_0_fused_f32(td.raw.data(), td.rows, td.cols, input.data(), input.size(), ref.data(), ref.size());
        minllama::matvec_q8_0_neon_f32(td.raw.data(), td.rows, td.cols, input.data(), input.size(), out.data(), out.size());

        double max_d = 0;
        for (std::size_t i = 0; i < td.rows; ++i) {
            double d = std::abs((double)ref[i] - (double)out[i]);
            if (d > max_d) max_d = d;
        }

        // Warmup
        for (int w = 0; w < 2; ++w) {
            minllama::matvec_q8_0_neon_f32(td.raw.data(), td.rows, td.cols, input.data(), input.size(), out.data(), out.size());
            minllama::matvec_q8_0_fused_f32(td.raw.data(), td.rows, td.cols, input.data(), input.size(), out.data(), out.size());
        }

        int iters = 5;
        double t_fused = 0, t_neon = 0;
        for (int i = 0; i < iters; ++i) {
            auto t0 = Clock::now();
            minllama::matvec_q8_0_fused_f32(td.raw.data(), td.rows, td.cols, input.data(), input.size(), out.data(), out.size());
            t_fused += elapsed_ms(t0, Clock::now());
            auto t1 = Clock::now();
            minllama::matvec_q8_0_neon_f32(td.raw.data(), td.rows, td.cols, input.data(), input.size(), out.data(), out.size());
            t_neon += elapsed_ms(t1, Clock::now());
        }

        t_fused /= iters;
        t_neon /= iters;

        std::printf("%-35s [%5zux%4zu]  scalar=%7.2fms  neon=%7.2fms  speedup=%5.2fx  max_diff=%.2e %s\n",
                    td.name.c_str(), td.rows, td.cols, t_fused, t_neon,
                    t_fused / t_neon, max_d,
                    max_d < 1e-4 ? "PASS" : "FAIL");
    }

    ml_model_free(ml);
    return 0;
}
