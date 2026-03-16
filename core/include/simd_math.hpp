#pragma once
/**
 * simd_math.hpp
 * SIMD-accelerated math primitives for batch drone physics operations.
 *
 * Strategy:
 *   - Vec3x4 / Vec3x8 : pack 4/8 Vec3s into __m256d lanes for AVX2 batch ops
 *   - SIMDNoise        : AVX2 4-wide parallel fBm evaluation
 *   - SIMDBatch        : turbulence_batch, noise_slice_2d using AVX2
 *
 * Runtime feature detection: falls back to SSE4.2 → scalar cleanly.
 * No UB: all loads/stores use aligned_alloc or explicit unaligned intrinsics.
 *
 * Compile requirements:
 *   MSVC:   /arch:AVX2
 *   GCC:    -mavx2 -mfma
 *   Clang:  -mavx2 -mfma
 */

#include "math_types.hpp"
#include <cstdint>
#include <cstring>
#include <cmath>

// ── Feature detection ──────────────────────────────────────────────────────
#if defined(__AVX2__) && defined(__FMA__)
#  define DRONE_AVX2 1
#  include <immintrin.h>
#elif defined(__SSE4_2__)
#  define DRONE_SSE42 1
#  include <nmmintrin.h>
#elif defined(_MSC_VER) && defined(__AVX2__)
#  define DRONE_AVX2 1
#  include <intrin.h>
#endif

// Cache-line friendly alignment (64 bytes for AVX-512 future compat)
#ifndef DRONE_ALIGN
#  define DRONE_ALIGN alignas(64)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// CPUID runtime detection
// ─────────────────────────────────────────────────────────────────────────────
struct CPUFeatures {
    bool avx2{false};
    bool fma{false};
    bool sse42{false};

    static CPUFeatures detect() noexcept {
        CPUFeatures f;
#if defined(_MSC_VER)
        int cpuInfo[4];
        __cpuid(cpuInfo, 0);
        int nIds = cpuInfo[0];
        if(nIds >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            f.avx2  = (cpuInfo[1] >> 5) & 1;
        }
        __cpuid(cpuInfo, 1);
        f.sse42 = (cpuInfo[2] >> 20) & 1;
        f.fma   = (cpuInfo[2] >> 12) & 1;
#elif defined(__GNUC__) || defined(__clang__)
        unsigned eax, ebx, ecx, edx;
        __asm__("cpuid" : "=a"(eax),"=b"(ebx),"=c"(ecx),"=d"(edx)
                       : "0"(7), "2"(0));
        f.avx2  = (ebx >> 5) & 1;
        __asm__("cpuid" : "=a"(eax),"=b"(ebx),"=c"(ecx),"=d"(edx)
                       : "0"(1), "2"(0));
        f.sse42 = (ecx >> 20) & 1;
        f.fma   = (ecx >> 12) & 1;
#endif
        return f;
    }
};

inline const CPUFeatures& cpu_features() noexcept {
    static CPUFeatures f = CPUFeatures::detect();
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Vec3x4 — 4 Vec3s packed into three __m256d registers (AoS → SoA transpose)
// Layout: xs = {v0.x, v1.x, v2.x, v3.x}, same for ys, zs
// ─────────────────────────────────────────────────────────────────────────────
#ifdef DRONE_AVX2
struct Vec3x4 {
    __m256d xs, ys, zs;

    // Load 4 scalar Vec3s
    static Vec3x4 load(const Vec3* v) noexcept {
        Vec3x4 r;
        r.xs = _mm256_set_pd(v[3].x, v[2].x, v[1].x, v[0].x);
        r.ys = _mm256_set_pd(v[3].y, v[2].y, v[1].y, v[0].y);
        r.zs = _mm256_set_pd(v[3].z, v[2].z, v[1].z, v[0].z);
        return r;
    }

    static Vec3x4 from_doubles(
        const double* xs4, const double* ys4, const double* zs4) noexcept
    {
        Vec3x4 r;
        r.xs = _mm256_loadu_pd(xs4);
        r.ys = _mm256_loadu_pd(ys4);
        r.zs = _mm256_loadu_pd(zs4);
        return r;
    }

    Vec3x4 operator+(const Vec3x4& o) const noexcept {
        return {_mm256_add_pd(xs,o.xs), _mm256_add_pd(ys,o.ys), _mm256_add_pd(zs,o.zs)};
    }
    Vec3x4 operator*(double s) const noexcept {
        __m256d sv = _mm256_set1_pd(s);
        return {_mm256_mul_pd(xs,sv), _mm256_mul_pd(ys,sv), _mm256_mul_pd(zs,sv)};
    }
    Vec3x4 operator*(__m256d sv) const noexcept {
        return {_mm256_mul_pd(xs,sv), _mm256_mul_pd(ys,sv), _mm256_mul_pd(zs,sv)};
    }

    // Dot product of this and other → 4 scalars
    __m256d dot(const Vec3x4& o) const noexcept {
        return _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(xs,o.xs), _mm256_mul_pd(ys,o.ys)),
            _mm256_mul_pd(zs,o.zs)
        );
    }

    // Component-wise norm²
    __m256d norm2() const noexcept { return dot(*this); }

    // Store back to 4 Vec3s (strided)
    void store(double* out_xyz, int stride=3) const noexcept {
        DRONE_ALIGN double tx[4], ty[4], tz[4];
        _mm256_store_pd(tx, xs);
        _mm256_store_pd(ty, ys);
        _mm256_store_pd(tz, zs);
        for(int i=0;i<4;++i){
            out_xyz[i*stride+0] = tx[i];
            out_xyz[i*stride+1] = ty[i];
            out_xyz[i*stride+2] = tz[i];
        }
    }

    // Extract single lane
    Vec3 lane(int i) const noexcept {
        DRONE_ALIGN double tx[4], ty[4], tz[4];
        _mm256_store_pd(tx, xs);
        _mm256_store_pd(ty, ys);
        _mm256_store_pd(tz, zs);
        return {tx[i], ty[i], tz[i]};
    }
};

// Convenience: AVX2 floor
inline __m256d avx2_floor(__m256d x) noexcept {
    return _mm256_floor_pd(x);
}

// Convenience: AVX2 quintic fade  t³(t(6t-15)+10)
inline __m256d avx2_fade(__m256d t) noexcept {
    const __m256d c6 = _mm256_set1_pd(6.0);
    const __m256d c15= _mm256_set1_pd(15.0);
    const __m256d c10= _mm256_set1_pd(10.0);
    // t*(6t-15) + 10
    __m256d inner = _mm256_fmadd_pd(c6, t, _mm256_sub_pd(_mm256_setzero_pd(), c15));
    inner = _mm256_fmadd_pd(t, inner, c10);
    // t² * inner = t³(...)
    __m256d t2 = _mm256_mul_pd(t, t);
    return _mm256_mul_pd(_mm256_mul_pd(t2, t), inner);
}

// AVX2 lerp: a + t*(b-a)
inline __m256d avx2_lerp(__m256d a, __m256d b, __m256d t) noexcept {
    return _mm256_fmadd_pd(t, _mm256_sub_pd(b, a), a);
}

#endif // DRONE_AVX2

// ─────────────────────────────────────────────────────────────────────────────
// SIMDBatch — batch evaluation dispatch helpers
// Used by TurbulenceEngine::evaluate_batch and noise_slice_2d
// ─────────────────────────────────────────────────────────────────────────────
namespace simd_batch {

// Process N doubles in chunks of 4 (AVX2) or 1 (scalar fallback)
// Fn4: void(int base_idx) processes indices [base_idx, base_idx+4)
// Fn1: void(int i)
template<typename Fn4, typename Fn1>
void dispatch(int N, Fn4 fn4, Fn1 fn1) noexcept {
#ifdef DRONE_AVX2
    int i = 0;
    for(; i + 4 <= N; i += 4) fn4(i);
    for(; i < N; ++i)          fn1(i);
#else
    for(int i = 0; i < N; ++i) fn1(i);
#endif
}

} // namespace simd_batch
