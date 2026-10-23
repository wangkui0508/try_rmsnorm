// FP16 RMSNorm for Apple Silicon, tuned for row length 4096.
//
// y[r, i] = x[r, i] * rsqrt(mean(x[r, :]^2) + eps) * w[i]
//
// The sum of squares is accumulated in FP32.  On M5 the FP16->FP32 widening
// FMA (FMLAL) lets us consume 8 halves per pair of instructions, which is
// noticeably cheaper than explicit conversion + FMA.
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static inline float hsum_f32x4(float32x4_t v) {
    return vaddvq_f32(v);
}

// One row.  n must be positive.  `w` may be NULL.
static inline void rmsnorm_row_f16(const __fp16 *__restrict x,
                                   const __fp16 *__restrict w,
                                   __fp16 *__restrict y,
                                   int n, float eps) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    float32x4_t a4 = vdupq_n_f32(0.0f);
    float32x4_t a5 = vdupq_n_f32(0.0f);
    float32x4_t a6 = vdupq_n_f32(0.0f);
    float32x4_t a7 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 64 <= n; i += 64) {
        float16x8_t x0 = vld1q_f16(x + i + 0);
        float16x8_t x1 = vld1q_f16(x + i + 8);
        float16x8_t x2 = vld1q_f16(x + i + 16);
        float16x8_t x3 = vld1q_f16(x + i + 24);
        float16x8_t x4 = vld1q_f16(x + i + 32);
        float16x8_t x5 = vld1q_f16(x + i + 40);
        float16x8_t x6 = vld1q_f16(x + i + 48);
        float16x8_t x7 = vld1q_f16(x + i + 56);

        a0 = vfmlalq_low_f16(a0, x0, x0);
        a0 = vfmlalq_high_f16(a0, x0, x0);
        a1 = vfmlalq_low_f16(a1, x1, x1);
        a1 = vfmlalq_high_f16(a1, x1, x1);
        a2 = vfmlalq_low_f16(a2, x2, x2);
        a2 = vfmlalq_high_f16(a2, x2, x2);
        a3 = vfmlalq_low_f16(a3, x3, x3);
        a3 = vfmlalq_high_f16(a3, x3, x3);
        a4 = vfmlalq_low_f16(a4, x4, x4);
        a4 = vfmlalq_high_f16(a4, x4, x4);
        a5 = vfmlalq_low_f16(a5, x5, x5);
        a5 = vfmlalq_high_f16(a5, x5, x5);
        a6 = vfmlalq_low_f16(a6, x6, x6);
        a6 = vfmlalq_high_f16(a6, x6, x6);
        a7 = vfmlalq_low_f16(a7, x7, x7);
        a7 = vfmlalq_high_f16(a7, x7, x7);
    }
    for (; i + 8 <= n; i += 8) {
        float16x8_t xv = vld1q_f16(x + i);
        a0 = vfmlalq_low_f16(a0, xv, xv);
        a0 = vfmlalq_high_f16(a0, xv, xv);
    }

    float sum = hsum_f32x4(vaddq_f32(
        vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)),
        vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7))));
    for (; i < n; ++i) {
        float v = (float)x[i];
        sum = fmaf(v, v, sum);
    }

    const float scale = 1.0f / sqrtf(sum / (float)n + eps);
    const float32x4_t vs = vdupq_n_f32(scale);

    i = 0;
    if (w == NULL) {
        for (; i + 32 <= n; i += 32) {
            float16x8_t x0 = vld1q_f16(x + i + 0);
            float16x8_t x1 = vld1q_f16(x + i + 8);
            float16x8_t x2 = vld1q_f16(x + i + 16);
            float16x8_t x3 = vld1q_f16(x + i + 24);

            float32x4_t o0 = vmulq_f32(vcvt_f32_f16(vget_low_f16(x0)), vs);
            float32x4_t o1 = vmulq_f32(vcvt_f32_f16(vget_high_f16(x0)), vs);
            float32x4_t o2 = vmulq_f32(vcvt_f32_f16(vget_low_f16(x1)), vs);
            float32x4_t o3 = vmulq_f32(vcvt_f32_f16(vget_high_f16(x1)), vs);
            float32x4_t o4 = vmulq_f32(vcvt_f32_f16(vget_low_f16(x2)), vs);
            float32x4_t o5 = vmulq_f32(vcvt_f32_f16(vget_high_f16(x2)), vs);
            float32x4_t o6 = vmulq_f32(vcvt_f32_f16(vget_low_f16(x3)), vs);
            float32x4_t o7 = vmulq_f32(vcvt_f32_f16(vget_high_f16(x3)), vs);

            vst1q_f16(y + i + 0, vcombine_f16(vcvt_f16_f32(o0), vcvt_f16_f32(o1)));
            vst1q_f16(y + i + 8, vcombine_f16(vcvt_f16_f32(o2), vcvt_f16_f32(o3)));
            vst1q_f16(y + i + 16, vcombine_f16(vcvt_f16_f32(o4), vcvt_f16_f32(o5)));
            vst1q_f16(y + i + 24, vcombine_f16(vcvt_f16_f32(o6), vcvt_f16_f32(o7)));
        }
    } else {
        for (; i + 32 <= n; i += 32) {
            float16x8_t x0 = vld1q_f16(x + i + 0);
            float16x8_t x1 = vld1q_f16(x + i + 8);
            float16x8_t x2 = vld1q_f16(x + i + 16);
            float16x8_t x3 = vld1q_f16(x + i + 24);
            float16x8_t w0 = vld1q_f16(w + i + 0);
            float16x8_t w1 = vld1q_f16(w + i + 8);
            float16x8_t w2 = vld1q_f16(w + i + 16);
            float16x8_t w3 = vld1q_f16(w + i + 24);

            float32x4_t o0 = vmulq_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(x0)), vs),
                                       vcvt_f32_f16(vget_low_f16(w0)));
            float32x4_t o1 = vmulq_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(x0)), vs),
                                       vcvt_f32_f16(vget_high_f16(w0)));
            float32x4_t o2 = vmulq_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(x1)), vs),
                                       vcvt_f32_f16(vget_low_f16(w1)));
            float32x4_t o3 = vmulq_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(x1)), vs),
                                       vcvt_f32_f16(vget_high_f16(w1)));
            float32x4_t o4 = vmulq_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(x2)), vs),
                                       vcvt_f32_f16(vget_low_f16(w2)));
            float32x4_t o5 = vmulq_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(x2)), vs),
                                       vcvt_f32_f16(vget_high_f16(w2)));
            float32x4_t o6 = vmulq_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(x3)), vs),
                                       vcvt_f32_f16(vget_low_f16(w3)));
            float32x4_t o7 = vmulq_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(x3)), vs),
                                       vcvt_f32_f16(vget_high_f16(w3)));

            vst1q_f16(y + i + 0, vcombine_f16(vcvt_f16_f32(o0), vcvt_f16_f32(o1)));
            vst1q_f16(y + i + 8, vcombine_f16(vcvt_f16_f32(o2), vcvt_f16_f32(o3)));
            vst1q_f16(y + i + 16, vcombine_f16(vcvt_f16_f32(o4), vcvt_f16_f32(o5)));
            vst1q_f16(y + i + 24, vcombine_f16(vcvt_f16_f32(o6), vcvt_f16_f32(o7)));
        }
    }
    for (; i + 8 <= n; i += 8) {
        float16x8_t xv = vld1q_f16(x + i);
        float32x4_t olo = vmulq_f32(vcvt_f32_f16(vget_low_f16(xv)), vs);
        float32x4_t ohi = vmulq_f32(vcvt_f32_f16(vget_high_f16(xv)), vs);
        if (w != NULL) {
            float16x8_t wv = vld1q_f16(w + i);
            olo = vmulq_f32(olo, vcvt_f32_f16(vget_low_f16(wv)));
            ohi = vmulq_f32(ohi, vcvt_f32_f16(vget_high_f16(wv)));
        }
        vst1q_f16(y + i, vcombine_f16(vcvt_f16_f32(olo), vcvt_f16_f32(ohi)));
    }
    for (; i < n; ++i) {
        float v = (float)x[i] * scale;
        if (w != NULL) v *= (float)w[i];
        y[i] = (__fp16)v;
    }
}

void rmsnorm_f16_row(const __fp16 *x, const __fp16 *w, __fp16 *y,
                     int n, float eps) {
    rmsnorm_row_f16(x, w, y, n, eps);
}

typedef struct {
    const __fp16 *x;
    const __fp16 *w;
    __fp16 *y;
    int rows;
    int n;
    float eps;
    int nthreads;
} batch_ctx;

static void batch_worker(void *vctx, size_t tid) {
    const batch_ctx *c = (const batch_ctx *)vctx;
    int t = (int)tid;
    int begin = (int)((int64_t)c->rows * t / c->nthreads);
    int end = (int)((int64_t)c->rows * (t + 1) / c->nthreads);
    for (int r = begin; r < end; ++r) {
        rmsnorm_row_f16(c->x + (size_t)r * c->n, c->w, c->y + (size_t)r * c->n,
                        c->n, c->eps);
    }
}

void rmsnorm_f16_batch(const __fp16 *x, const __fp16 *w, __fp16 *y,
                       int rows, int n, float eps, int nthreads) {
    if (rows <= 0 || n <= 0) return;
    if (nthreads <= 1 || rows == 1) {
        for (int r = 0; r < rows; ++r) {
            rmsnorm_row_f16(x + (size_t)r * n, w, y + (size_t)r * n, n, eps);
        }
        return;
    }
    if (nthreads > rows) nthreads = rows;
    batch_ctx ctx = {x, w, y, rows, n, eps, nthreads};
    dispatch_apply_f((size_t)nthreads, DISPATCH_APPLY_AUTO, &ctx, batch_worker);
}
