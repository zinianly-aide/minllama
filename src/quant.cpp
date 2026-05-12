#include "minllama_internal.h"

#include <cmath>
#include <cstring>
#include <cstdint>

float ml_fp16_to_fp32(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fu;
    std::uint32_t mantissa = value & 0x03ffu;

    if (exponent == 0) {
        if (mantissa == 0) {
            std::uint32_t bits = sign;
            float out;
            std::memcpy(&out, &bits, sizeof(out));
            return out;
        }

        // Normalize subnormal half precision values.
        exponent = 1;
        while ((mantissa & 0x0400u) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        mantissa &= 0x03ffu;
    } else if (exponent == 0x1fu) {
        const std::uint32_t bits = sign | 0x7f800000u | (mantissa << 13);
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    const std::uint32_t bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
