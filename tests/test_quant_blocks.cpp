#include "minllama_internal.h"
#include "test_gguf_util.h"

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

namespace {

void assert_close(float actual, float expected, float tolerance = 1e-5f) {
    assert(std::fabs(actual - expected) < tolerance);
}

void test_q40_scale(std::uint16_t scale_bits, float scale_value) {
    std::vector<int> q = {
        -8, -4, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -3, -2,
         7,  6,  5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8,
    };
    std::array<unsigned char, 16> packed = {};
    {
        std::ofstream out("/tmp/q40_block.bin", std::ios::binary | std::ios::trunc);
        {
            bool ok = minllama_test::write_tensor_payload_q4_0_block(out, scale_bits, q);
            assert(ok);
        }
    }
    {
        std::ifstream in("/tmp/q40_block.bin", std::ios::binary);
        std::uint16_t scale_read = 0;
        in.read(reinterpret_cast<char*>(&scale_read), 2);
        in.read(reinterpret_cast<char*>(packed.data()), packed.size());
        assert(scale_read == scale_bits);
    }
    std::array<float, 32> out = {};
    assert(minllama::decode_q4_0_block_f32(scale_bits, packed.data(), packed.size(), out.data(), out.size()));
    for (std::size_t i = 0; i < out.size(); ++i) {
        assert_close(out[i], scale_value * static_cast<float>(q[i]), std::max(1e-5f, std::fabs(scale_value) * 1e-5f));
    }
}

void test_q80_scale(std::uint16_t scale_bits, float scale_value) {
    std::vector<int> q(32);
    for (int i = 0; i < 32; ++i) q[i] = i - 16;
    std::array<unsigned char, 32> qs = {};
    {
        std::ofstream out("/tmp/q80_block.bin", std::ios::binary | std::ios::trunc);
        {
            bool ok = minllama_test::write_tensor_payload_q8_0_block(out, scale_bits, q);
            assert(ok);
        }
    }
    {
        std::ifstream in("/tmp/q80_block.bin", std::ios::binary);
        std::uint16_t scale_read = 0;
        in.read(reinterpret_cast<char*>(&scale_read), 2);
        in.read(reinterpret_cast<char*>(qs.data()), qs.size());
        assert(scale_read == scale_bits);
    }
    std::array<float, 32> out = {};
    assert(minllama::decode_q8_0_block_f32(scale_bits, qs.data(), qs.size(), out.data(), out.size()));
    for (std::size_t i = 0; i < out.size(); ++i) {
        assert_close(out[i], scale_value * static_cast<float>(q[i]), std::max(1e-5f, std::fabs(scale_value) * 1e-5f));
    }
    assert_close(out[0], scale_value * -16.0f);
    assert_close(out[31], scale_value * 15.0f);
}

} // namespace

int main() {
    test_q40_scale(0x3c00u, 1.0f);
    test_q40_scale(0x3800u, 0.5f);
    test_q40_scale(0xbc00u, -1.0f);
    test_q40_scale(0x0001u, ml_fp16_to_fp32(0x0001u));
    test_q40_scale(0x7bffu, ml_fp16_to_fp32(0x7bffu));

    test_q80_scale(0x3c00u, 1.0f);
    test_q80_scale(0x3800u, 0.5f);
    test_q80_scale(0xbc00u, -1.0f);
    test_q80_scale(0x0001u, ml_fp16_to_fp32(0x0001u));
    test_q80_scale(0x7bffu, ml_fp16_to_fp32(0x7bffu));

    std::array<float, 32> out = {};
    std::array<unsigned char, 15> short_q4 = {};
    std::array<unsigned char, 31> short_q8 = {};
    assert(!minllama::decode_q4_0_block_f32(0x3c00u, short_q4.data(), short_q4.size(), out.data(), out.size()));
    assert(!minllama::decode_q8_0_block_f32(0x3c00u, short_q8.data(), short_q8.size(), out.data(), out.size()));
    return 0;
}
