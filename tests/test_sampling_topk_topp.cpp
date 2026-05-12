#include "minllama_internal.h"

#include <cassert>
#include <set>
#include <vector>

int main() {
    // ----------------------------------------------------------------
    // Test 1: top_k=1 equivalent to greedy.
    // ----------------------------------------------------------------
    {
        const float logits[] = {1.0f, 3.0f, 2.0f};
        uint32_t rng = 42;
        int r = minllama::sample_top_k_top_p_f32(logits, 3, 1.0f, 1, 1.0f, &rng);
        assert(r == 1); // argmax
    }

    // ----------------------------------------------------------------
    // Test 2: top_k=1 with T=0 also greedy.
    // ----------------------------------------------------------------
    {
        const float logits[] = {1.0f, 3.0f, 2.0f};
        uint32_t rng = 42;
        int r = minllama::sample_top_k_top_p_f32(logits, 3, 0.0f, 1, 1.0f, &rng);
        assert(r == 1);
    }

    // ----------------------------------------------------------------
    // Test 3: top_k=2 result only in top 2.
    // logits=[0.1, 0.4, 0.3, 0.2], T=1, top_k=2
    // Top 2: index 1 (0.4) and index 2 (0.3)
    // Result must be 1 or 2.
    // ----------------------------------------------------------------
    {
        const float logits[] = {0.1f, 0.4f, 0.3f, 0.2f};
        uint32_t rng = 1;
        for (int i = 0; i < 50; ++i) {
            int r = minllama::sample_top_k_top_p_f32(logits, 4, 1.0f, 2, 1.0f, &rng);
            assert(r == 1 || r == 2);
        }
    }

    // ----------------------------------------------------------------
    // Test 4: top_p very small keeps at least the highest prob token.
    // logits=[10, 1, 1], T=1 → probs ≈ [0.9998, 0.0001, 0.0001]
    // top_p=0.5 only keeps token 0.
    // ----------------------------------------------------------------
    {
        const float logits[] = {10.0f, 1.0f, 1.0f};
        uint32_t rng = 1;
        for (int i = 0; i < 20; ++i) {
            int r = minllama::sample_top_k_top_p_f32(logits, 3, 1.0f, 0, 0.5f, &rng);
            assert(r == 0);
        }
    }

    // ----------------------------------------------------------------
    // Test 5: top_p preserves multiple tokens.
    // logits=[5, 3, 2], T=1 → probs ≈ [0.844, 0.114, 0.042]
    // top_p=0.96 keeps tokens 0 and 1 (cumul 0.958 ≤ 0.96).
    // ----------------------------------------------------------------
    {
        const float logits[] = {5.0f, 3.0f, 2.0f};
        uint32_t rng = 1;
        bool saw_0 = false, saw_1 = false;
        for (int i = 0; i < 100; ++i) {
            int r = minllama::sample_top_k_top_p_f32(logits, 3, 1.0f, 0, 0.96f, &rng);
            assert(r == 0 || r == 1);
            if (r == 0) saw_0 = true;
            if (r == 1) saw_1 = true;
        }
        assert(saw_0);
        assert(saw_1); // 100 samples with p≈0.12 should hit token 1
    }

    // ----------------------------------------------------------------
    // Test 6: top_k + top_p combined works.
    // logits=[5, 3, 2, 1], T=1, top_k=3, top_p=0.9
    // ----------------------------------------------------------------
    {
        const float logits[] = {5.0f, 3.0f, 2.0f, 1.0f};
        uint32_t rng = 1;
        for (int i = 0; i < 30; ++i) {
            int r = minllama::sample_top_k_top_p_f32(logits, 4, 1.0f, 3, 0.9f, &rng);
            assert(r >= 0 && r < 4);
        }
    }

    // ----------------------------------------------------------------
    // Test 7: same seed → same result with top_k/top_p.
    // ----------------------------------------------------------------
    {
        const float logits[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        uint32_t rng1 = 42, rng2 = 42;
        int r1 = minllama::sample_top_k_top_p_f32(logits, 5, 1.0f, 3, 0.8f, &rng1);
        int r2 = minllama::sample_top_k_top_p_f32(logits, 5, 1.0f, 3, 0.8f, &rng2);
        assert(r1 == r2);
    }

    // ----------------------------------------------------------------
    // Test 8: invalid — nullptr.
    // ----------------------------------------------------------------
    {
        const float logits[] = {1.0f};
        uint32_t rng = 1;
        assert(minllama::sample_top_k_top_p_f32(nullptr, 1, 1.0f, 0, 1.0f, &rng) == -1);
        assert(minllama::sample_top_k_top_p_f32(logits, 1, 1.0f, 0, 1.0f, nullptr) == -1);
    }

    // ----------------------------------------------------------------
    // Test 9: invalid — top_k < 0.
    // ----------------------------------------------------------------
    {
        const float logits[] = {1.0f, 2.0f};
        uint32_t rng = 1;
        assert(minllama::sample_top_k_top_p_f32(logits, 2, 1.0f, -1, 1.0f, &rng) == -1);
    }

    // ----------------------------------------------------------------
    // Test 10: invalid — top_p <= 0.
    // ----------------------------------------------------------------
    {
        const float logits[] = {1.0f, 2.0f};
        uint32_t rng = 1;
        assert(minllama::sample_top_k_top_p_f32(logits, 2, 1.0f, 0, 0.0f, &rng) == -1);
        assert(minllama::sample_top_k_top_p_f32(logits, 2, 1.0f, 0, -0.1f, &rng) == -1);
    }

    // ----------------------------------------------------------------
    // Test 11: top_k >= vocab_size → no filtering.
    // ----------------------------------------------------------------
    {
        const float logits[] = {0.1f, 0.4f, 0.3f};
        uint32_t rng = 1;
        int r = minllama::sample_top_k_top_p_f32(logits, 3, 1.0f, 10, 1.0f, &rng);
        assert(r >= 0 && r < 3);
    }

    // ----------------------------------------------------------------
    // Test 12: top_p=1.0 → no filtering.
    // ----------------------------------------------------------------
    {
        const float logits[] = {1.0f, 2.0f, 3.0f};
        uint32_t rng = 1;
        int r = minllama::sample_top_k_top_p_f32(logits, 3, 1.0f, 0, 1.0f, &rng);
        assert(r >= 0 && r < 3);
    }

    return 0;
}
