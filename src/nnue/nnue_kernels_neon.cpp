// ARM NEON implementation of Bagatur NNUE kernels.
#include "nnue_kernels.h"
#include "nnue.h"
#include <arm_neon.h>

namespace nnue::kernels {
namespace {
constexpr int HS = HIDDEN_SIZE;

void v_add(std::int16_t* v, const std::int16_t* w) {
    for (int i = 0; i < HS; i += 8) {
        int16x8_t a = vld1q_s16(v + i);
        int16x8_t b = vld1q_s16(w + i);
        vst1q_s16(v + i, vaddq_s16(a, b));
    }
}
void v_sub(std::int16_t* v, const std::int16_t* w) {
    for (int i = 0; i < HS; i += 8) {
        int16x8_t a = vld1q_s16(v + i);
        int16x8_t b = vld1q_s16(w + i);
        vst1q_s16(v + i, vsubq_s16(a, b));
    }
}
void v_add_sub(std::int16_t* v, const std::int16_t* a, const std::int16_t* s) {
    for (int i = 0; i < HS; i += 8) {
        int16x8_t x = vld1q_s16(v + i);
        x = vaddq_s16(x, vld1q_s16(a + i));
        x = vsubq_s16(x, vld1q_s16(s + i));
        vst1q_s16(v + i, x);
    }
}
std::int64_t v_screlu_dot(const std::int16_t* acc, const std::int32_t* weights) {
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t hi = vdupq_n_s16(static_cast<std::int16_t>(QA));
    std::int64_t total = 0;
    for (int i = 0; i < HS; i += 8) {
        int16x8_t x = vmaxq_s16(vld1q_s16(acc + i), zero);
        x = vminq_s16(x, hi);
        int32x4_t lo = vmovl_s16(vget_low_s16(x));
        int32x4_t up = vmovl_s16(vget_high_s16(x));
        int32x4_t sq0 = vmulq_s32(lo, lo);
        int32x4_t sq1 = vmulq_s32(up, up);
        int32x4_t w0 = vld1q_s32(weights + i);
        int32x4_t w1 = vld1q_s32(weights + i + 4);
        // Widen each signed 32x32 product to 64 bits before accumulation.
        // vmulq_s32 would overflow for large NNUE weights before the sum is widened.
        const int64x2_t p00 = vmull_s32(vget_low_s32(sq0),  vget_low_s32(w0));
        const int64x2_t p01 = vmull_s32(vget_high_s32(sq0), vget_high_s32(w0));
        const int64x2_t p10 = vmull_s32(vget_low_s32(sq1),  vget_low_s32(w1));
        const int64x2_t p11 = vmull_s32(vget_high_s32(sq1), vget_high_s32(w1));
        alignas(16) std::int64_t tmp[8];
        vst1q_s64(tmp,     p00);
        vst1q_s64(tmp + 2, p01);
        vst1q_s64(tmp + 4, p10);
        vst1q_s64(tmp + 6, p11);
        for (int j = 0; j < 8; ++j) total += tmp[j];
    }
    return total;
}
}
const Kernels& neon() {
    static const Kernels k{&v_add, &v_sub, &v_add_sub, &v_screlu_dot, "ARM NEON"};
    return k;
}
}
