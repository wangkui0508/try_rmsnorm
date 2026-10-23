#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef __fp16 half;
void rmsnorm_f16_row(const half *x, const half *w, half *y, int n, float eps);
void rmsnorm_f16_batch(const half *x, const half *w, half *y, int rows, int n,
                       float eps, int nthreads);

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 1;
    int iters = argc > 2 ? atoi(argv[2]) : 1000000;
    int threads = argc > 3 ? atoi(argv[3]) : 1;
    int weight = argc > 4 ? atoi(argv[4]) : 0;
    size_t n = 4096;
    half *x = aligned_alloc(64, rows * n * sizeof(half));
    half *y = aligned_alloc(64, rows * n * sizeof(half));
    half *w = aligned_alloc(64, n * sizeof(half));
    for (size_t i = 0; i < (size_t)rows * n; ++i) x[i] = (half)((float)rand() * (1.0f / 2147483648.0f) - 0.5f);
    for (size_t i = 0; i < n; ++i) w[i] = (half)(0.5f + (float)rand() * (1.0f / 2147483648.0f));

    for (int i = 0; i < 2000; ++i)
        rmsnorm_f16_batch(x, weight ? w : 0, y, rows, (int)n, 1e-5f, threads);

    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        double t0 = now_ns();
        for (int i = 0; i < iters; ++i)
            rmsnorm_f16_batch(x, weight ? w : 0, y, rows, (int)n, 1e-5f, threads);
        double dt = (now_ns() - t0) / iters;
        if (dt < best) best = dt;
    }
    double bytes = (double)rows * n * sizeof(half) * (weight ? 3.0 : 2.0);
    printf("rows=%d threads=%d weight=%d  %.1f ns/call  %.1f GB/s  checksum=%.6f\n",
           rows, threads, weight, best, bytes / best, (double)y[0]);
    free(x); free(y); free(w);
    return 0;
}
