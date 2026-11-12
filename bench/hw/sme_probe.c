// Peak int8 outer-product rate of the CPU's SME units (SMOPA into 32-bit ZA tiles), per thread count.
// build: clang -O2 -march=armv9.2-a+sme2 -o build/sme_probe bench/hw/sme_probe.c
#include <arm_sme.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

__arm_new("za") __arm_locally_streaming static uint64_t run(long iters, const int8_t *a, const int8_t *b) {
  svbool_t pg = svptrue_b8();
  svint8_t za0 = svld1_s8(pg, a), za1 = svld1_s8(pg, a + 64), zb0 = svld1_s8(pg, b), zb1 = svld1_s8(pg, b + 64);
  svzero_za();
  for (long i = 0; i < iters; i++) {
    svmopa_za32_s8_m(0, pg, pg, za0, zb0);
    svmopa_za32_s8_m(1, pg, pg, za0, zb1);
    svmopa_za32_s8_m(2, pg, pg, za1, zb0);
    svmopa_za32_s8_m(3, pg, pg, za1, zb1);
  }
  return svcntsb();
}

static long ITERS = 20000000;
static int8_t A[256], B[256];
static void *worker(void *arg) { (void)arg; run(ITERS, A, B); return NULL; }

int main(void) {
  for (int i = 0; i < 256; i++) { A[i] = (int8_t)(i * 7); B[i] = (int8_t)(i * 3); }
  uint64_t svl = run(1, A, B);
  printf("streaming vector length: %llu bytes\n", (unsigned long long)svl);
  double macs_per_mopa = (double)(svl / 4) * (svl / 4) * 4;  // tile rows x cols x 4-way dot
  for (int nt = 1; nt <= 8; nt *= 2) {
    pthread_t th[8];
    double t = now();
    for (int i = 0; i < nt; i++) pthread_create(&th[i], NULL, worker, NULL);
    for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
    t = now() - t;
    double tops = 2.0 * macs_per_mopa * 4 * ITERS * nt / t / 1e12;
    printf("%d thread(s): %.2f TOPS int8 (%.3f s)\n", nt, tops, t);
  }
  return 0;
}
