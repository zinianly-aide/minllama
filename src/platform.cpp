#include "minllama_internal.h"

#include <cstdio>
#include <cstring>

// Platform-specific includes for CPU feature detection.
#if defined(__GNUC__) || defined(__clang__)
  // __builtin_cpu_supports is available on GCC ≥ 5 and Clang ≥ 3.8
  #if defined(__has_builtin) && __has_builtin(__builtin_cpu_supports)
    #define MINLLAMA_HAS_BUILTIN_CPU_SUPPORTS 1
  #elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ >= 1))
    #define MINLLAMA_HAS_BUILTIN_CPU_SUPPORTS 1
  #endif
#endif

#if defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  // Windows: use IsProcessorFeaturePresent (limited) or __cpuid intrinsic
  #include <intrin.h>
#endif

#if defined(__APPLE__)
  #include <sys/sysctl.h>
#endif

namespace minllama {

// =======================================================================
// CPU feature detection
// =======================================================================

namespace {

struct CpuCaps {
    bool sse2  = false;
    bool avx   = false;
    bool avx2  = false;
    bool neon  = false;
    bool armv8 = false;

    const char *arch_name = "unknown";
    const char *simd_name = "scalar";

    bool detected = false;
};

CpuCaps g_caps;

// -------------------------------------------------------------------
// Compile-time known capabilities (set unconditionally)
// -------------------------------------------------------------------
void detect_compile_time(CpuCaps &caps) {
#if defined(__x86_64__) || defined(_M_X64)
    caps.arch_name = "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    caps.arch_name = "x86";
#elif defined(__aarch64__) || defined(__arm64__)
    caps.arch_name = "aarch64";
#elif defined(__arm__)
    caps.arch_name = "arm";
#else
    caps.arch_name = "unknown";
#endif

#ifdef __ARM_NEON
    caps.neon = true;
#endif
#ifdef __SSE2__
    caps.sse2 = true;
#endif
#ifdef __AVX__
    caps.avx = true;
#endif
#ifdef __AVX2__
    caps.avx2 = true;
#endif

    // Determine SIMD name from compile-time defines
#ifdef __ARM_NEON
    caps.simd_name = "NEON";
#elif defined(__AVX2__)
    caps.simd_name = "AVX2";
#elif defined(__SSE2__)
    caps.simd_name = "SSE2";
#else
    caps.simd_name = "scalar";
#endif
}

// -------------------------------------------------------------------
// Runtime CPU feature detection via OS / builtin
// -------------------------------------------------------------------
void detect_runtime_x86(CpuCaps &caps) {
#if MINLLAMA_HAS_BUILTIN_CPU_SUPPORTS
    caps.sse2 = __builtin_cpu_supports("sse2");
    caps.avx  = __builtin_cpu_supports("avx");
    caps.avx2 = __builtin_cpu_supports("avx2");
#elif defined(_WIN32)
    // Windows fallback: use IsProcessorFeaturePresent for SSE2
    caps.sse2 = IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE) != 0;

    // AVX/AVX2 detection via cpuid on Windows
    int cpu_info[4] = {0};
    __cpuid(cpu_info, 1);
    caps.avx  = (cpu_info[2] & (1 << 28)) != 0;  // ECX bit 28

    __cpuidex(cpu_info, 7, 0);
    caps.avx2 = (cpu_info[1] & (1 << 5)) != 0;   // EBX bit 5
#else
    // Fallback: trust compile-time flags
    // (Linux without builtin support — rare with modern GCC)
#endif
}

void detect_runtime_arm(CpuCaps &caps) {
#if defined(__aarch64__) || defined(__arm64__) || defined(__arm__)
    caps.armv8 = true;

#if defined(__APPLE__)
    // macOS: NEON is always available on arm64; detect via sysctl
    // (hw.optional.neon is deprecated on Apple Silicon but kept for compat)
    int has_neon = 0;
    std::size_t sz = sizeof(has_neon);
    if (sysctlbyname("hw.optional.neon", &has_neon, &sz, nullptr, 0) == 0) {
        caps.neon = (has_neon == 1);
    } else {
        caps.neon = true;  // arm64 macOS always has NEON
    }
#elif defined(__linux__)
    // Linux: parse /proc/cpuinfo or use getauxval(AT_HWCAP) & HWCAP_ASIMD
    // For now, trust compile-time detection (NEON is mandatory on armv8+)
    caps.neon = true;
#else
    caps.neon = true;  // assume available on modern ARM
#endif

#else
    (void)caps;  // unreachable for non-ARM
#endif
}

void detect_runtime(CpuCaps &caps) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    detect_runtime_x86(caps);
#elif defined(__aarch64__) || defined(__arm64__) || defined(__arm__)
    detect_runtime_arm(caps);
#endif
    caps.detected = true;
}

}  // anonymous namespace

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

const char *platform_arch_name() {
    if (!g_caps.detected) {
        detect_compile_time(g_caps);
        detect_runtime(g_caps);
    }
    return g_caps.arch_name;
}

const char *platform_simd_name() {
    if (!g_caps.detected) {
        detect_compile_time(g_caps);
        detect_runtime(g_caps);
    }
    return g_caps.simd_name;
}

bool platform_has_neon() {
    if (!g_caps.detected) {
        detect_compile_time(g_caps);
        detect_runtime(g_caps);
    }
    return g_caps.neon;
}

bool platform_has_sse2() {
    if (!g_caps.detected) {
        detect_compile_time(g_caps);
        detect_runtime(g_caps);
    }
    return g_caps.sse2;
}

bool platform_has_avx() {
    if (!g_caps.detected) {
        detect_compile_time(g_caps);
        detect_runtime(g_caps);
    }
    return g_caps.avx;
}

bool platform_has_avx2() {
    if (!g_caps.detected) {
        detect_compile_time(g_caps);
        detect_runtime(g_caps);
    }
    return g_caps.avx2;
}

void platform_dump_caps() {
    if (!g_caps.detected) {
        detect_compile_time(g_caps);
        detect_runtime(g_caps);
    }

    std::fprintf(stderr, "[platform] Architecture:   %s\n", g_caps.arch_name);
    std::fprintf(stderr, "[platform] Active SIMD:    %s\n", g_caps.simd_name);
    std::fprintf(stderr, "[platform] Runtime caps:\n");
    std::fprintf(stderr, "           SSE2:   %s\n", g_caps.sse2  ? "yes" : "no");
    std::fprintf(stderr, "           AVX:    %s\n", g_caps.avx   ? "yes" : "no");
    std::fprintf(stderr, "           AVX2:   %s\n", g_caps.avx2  ? "yes" : "no");
    std::fprintf(stderr, "           NEON:   %s\n", g_caps.neon  ? "yes" : "no");
    std::fprintf(stderr, "           ARMV8:  %s\n", g_caps.armv8 ? "yes" : "no");
}

void platform_skeleton_anchor() {}  // kept for backward compat

}  // namespace minllama
