#include "minllama_internal.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace minllama {
namespace {

constexpr std::size_t kMinChunkRows = 4;

struct GlobalPool {
    std::mutex mutex;
    std::unique_ptr<ThreadPool> pool;
};

GlobalPool &global_pool() {
    static GlobalPool gp;
    return gp;
}

} // namespace

ThreadPool::ThreadPool(int n_threads)
    : n_threads_(n_threads) {
    if (n_threads_ > 1) {
        workers_.reserve(static_cast<std::size_t>(n_threads_ - 1));
        for (int i = 0; i < n_threads_ - 1; ++i) {
            workers_.emplace_back(&ThreadPool::worker_loop, this, i);
        }
    }
}

ThreadPool::~ThreadPool() {
    if (n_threads_ > 1) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
            current_fn_ = nullptr;
            chunks_ = nullptr;
        }
        cv_.notify_all();
        for (auto &w : workers_) {
            if (w.joinable()) w.join();
        }
    }
}

void ThreadPool::parallel_for(std::size_t start, std::size_t end,
                               std::function<void(std::size_t, std::size_t)> fn) {
    if (start >= end) return;
    if (n_threads_ <= 1) {
        fn(start, end);
        return;
    }

    // Pre-assign chunk ranges to all threads (main + workers).
    const std::size_t total = end - start;
    const int n_workers = n_threads_;  // includes main thread
    const std::size_t base_chunk = total / static_cast<std::size_t>(n_workers);
    const std::size_t remainder = total % static_cast<std::size_t>(n_workers);

    // Store fn for workers.
    std::function<void(std::size_t, std::size_t)> shared_fn = std::move(fn);

    // Build chunk assignments on the stack (max 1024 workers, should be plenty).
    struct ChunkAssign { std::size_t start, end; };
    ChunkAssign chunk_assignments[1024];

    std::size_t cur = start;
    for (int i = 0; i < n_workers && i < 1024; ++i) {
        std::size_t size = base_chunk + (static_cast<std::size_t>(i) < remainder ? 1 : 0);
        chunk_assignments[i].start = cur;
        chunk_assignments[i].end = cur + size;
        cur += size;
    }

    // Publish the work under the lock. Workers will wait on cv_ until
    // current_fn_ and chunks_ are set.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_fn_ = &shared_fn;
        chunks_ = reinterpret_cast<Chunk *>(chunk_assignments);
        n_chunks_ = n_workers;
        chunks_done_ = 0;
        generation_++;  // signal new generation to workers
    }
    cv_.notify_all();

    // Main thread does its chunk (index 0).
    if (chunk_assignments[0].start < chunk_assignments[0].end) {
        shared_fn(chunk_assignments[0].start, chunk_assignments[0].end);
    }

    // Wait for all workers to finish.
    {
        std::unique_lock<std::mutex> lock(mutex_);
        auto start_wait = std::chrono::steady_clock::now();
        while (chunks_done_ < n_workers - 1) {
            auto now = std::chrono::steady_clock::now();
            if (now - start_wait > std::chrono::seconds(5)) {
                std::fprintf(stderr, "[tp FATAL] parallel_for hang: done=%d/%d\n",
                             chunks_done_, n_workers - 1);
                std::fflush(stderr);
                std::exit(1);
            }
            cv_.wait_for(lock, std::chrono::milliseconds(100));
        }
        // Workers are all done, clear the work state.
        current_fn_ = nullptr;
        chunks_ = nullptr;
        n_chunks_ = 0;
    }
}

void ThreadPool::worker_loop(int worker_id) {
    int my_gen = 0;  // last generation we processed
    while (true) {
        Chunk chunk;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this, &my_gen]() {
                return generation_ != my_gen || shutdown_;
            });
            if (shutdown_) return;

            my_gen = generation_;
            // Get assigned chunk for this worker.
            // worker_id maps to chunk index (worker_id+1 since main is 0).
            chunk = chunks_[worker_id + 1];
        }

        // Do work.
        if (chunk.start < chunk.end) {
            (*current_fn_)(chunk.start, chunk.end);
        }

        // Signal completion.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            chunks_done_++;
        }
        cv_.notify_one();
    }
}

ThreadPool &get_thread_pool(int n_threads) {
    auto &gp = global_pool();
    std::lock_guard<std::mutex> lock(gp.mutex);
    if (!gp.pool || gp.pool->num_threads() != n_threads) {
        gp.pool = std::make_unique<ThreadPool>(n_threads);
    }
    return *gp.pool;
}

} // namespace minllama

void thread_pool_skeleton_anchor() {}
