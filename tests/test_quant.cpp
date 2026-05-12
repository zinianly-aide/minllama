#include "../src/minllama_internal.h"

#include <cassert>
#include <cmath>

int main() {
    assert(ml_fp16_to_fp32(0x0000u) == 0.0f);
    assert(ml_fp16_to_fp32(0x3c00u) == 1.0f);
    assert(ml_fp16_to_fp32(0xc000u) == -2.0f);
    assert(std::isinf(ml_fp16_to_fp32(0x7c00u)));
    return 0;
}
