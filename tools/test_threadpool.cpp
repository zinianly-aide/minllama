#include "minllama_internal.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    // Stress test: 1000 iterations of parallel_for with 576-row matvec work at threads=2
    const int N = 1000;
    const int n_threads = 2;
    const std::size_t rows = 576;
    const std::size_t cols = 576;
    std::vector<float> matrix(rows * cols, 0.5f);
    std::vector<float> input(cols, 1.0f);
    std::vector<float> out(rows, 0.0f);

    minllama::ThreadPool &pool = minllama::get_thread_pool(n_threads);

    auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < N; ++iter) {
        pool.parallel_for(0, rows, [&](std::size_t s, std::size_t e) {
            for (std::size_t row = s; row < e; ++row) {
                float sum = 0.0f;
                std::size_t base = row * cols;
                for (std::size_t col = 0; col < cols; ++col) {
                    sum += matrix[base + col] * input[col];
                }
                out[row] = sum;
            }
        });
        // Verify correctness every 100 iterations
        if ((iter + 1) % 100 == 0) {
            float expected = cols * 0.5f;
            bool ok = true;
            for (std::size_t r = 0; r < rows; ++r) {
                if (out[r] != expected) {
                    std::fprintf(stderr, "FAIL: iter=%d row=%zu got=%f expected=%f\n",
                                 iter, r, out[r], expected);
                    ok = false;
                    break;
                }
            }
            if (ok) {
                std::printf("iter %d: OK (sum=%.1f)\n", iter + 1, out[0]);
            } else {
                std::exit(1);
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("\n=== STRESS TEST RESULTS ===\n");
    std::printf("threads=%d: %d calls in %.1fms = %.1fus per call\n",
                n_threads, N, ms, (ms*1000.0)/N);
    std::printf("All 1000 iterations passed correctness verification.\n");
    std::printf("DONE\n");
    return 0;
}
