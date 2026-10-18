// CPU second INT8 platform: peak SME outer-product throughput of the M5 Max CPU, INT8 vs FP16 / BF16 / FP32.
//
// Each thread runs a streaming-mode loop of independent outer products into all ZA tiles of the type (no data
// dependence between consecutive instructions on the same tile within a group), for a fixed number of iterations.
// MACs per instruction at SVL = 512 bits (sme_max_svl_b = 64):
//   SMOPA  za32 <- s8  x s8   (4-way)       16 x 16 x 4 = 1024     4 tiles
//   FMOPA  za16 <- f16 x f16  (non-widening) 32 x 32     = 1024     2 tiles   (FEAT_SME_F16F16, FP16 accumulation)
//   FMOPA  za32 <- f16 x f16  (2-way widening) 16 x 16 x 2 = 512    4 tiles   (FP32 accumulation, what GEMMs use)
//   BFMOPA za32 <- bf16       (2-way widening) 16 x 16 x 2 = 512    4 tiles
//   FMOPA  za32 <- f32                          16 x 16     = 256    4 tiles
// Threads: 1 .. 18 (QoS user-interactive); the SME units are shared per CPU cluster, so throughput scales with the
// number of clusters in use, not cores. Also reports Accelerate's cblas_sgemm (FP32, n = 4096) for context.
//
// Build (no SVE outside streaming mode: Apple CPUs have SME but no non-streaming SVE, so armv9 targets SIGILL):
// clang -O3 -march=armv8.7-a+sme2+sme-f16f16 -framework Accelerate kernels/cpu_sme_bench.c -o kernels/cpu_sme_bench
#include <arm_sme.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#define ITERS 200000000ull
#define U 2   // groups per loop iteration

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

__arm_new("za") __arm_locally_streaming static uint64_t k_s8(uint64_t n) {
  svbool_t p = svptrue_b8();
  svint8_t a0 = svdup_s8(1), a1 = svdup_s8(-2), b0 = svdup_s8(3), b1 = svdup_s8(1);
  for (uint64_t i = 0; i < n; ++i)
    for (int u = 0; u < U; ++u) {
      svmopa_za32_s8_m(0, p, p, a0, b0); svmopa_za32_s8_m(1, p, p, a0, b1);
      svmopa_za32_s8_m(2, p, p, a1, b0); svmopa_za32_s8_m(3, p, p, a1, b1);
    }
  return n * U * 4 * (svcntsw() * svcntsw() * 4);
}

__arm_new("za") __arm_locally_streaming static uint64_t k_f16(uint64_t n) {
  svbool_t p = svptrue_b16();
  svfloat16_t a0 = svdup_f16(1), a1 = svdup_f16(-2), b0 = svdup_f16(0.5), b1 = svdup_f16(1);
  for (uint64_t i = 0; i < n; ++i)
    for (int u = 0; u < U; ++u) {
      svmopa_za16_f16_m(0, p, p, a0, b0); svmopa_za16_f16_m(1, p, p, a0, b1);
      svmopa_za16_f16_m(0, p, p, a1, b0); svmopa_za16_f16_m(1, p, p, a1, b1);
    }
  return n * U * 4 * (svcntsh() * svcntsh());
}

__arm_new("za") __arm_locally_streaming static uint64_t k_f16w(uint64_t n) {
  svbool_t p = svptrue_b16();
  svfloat16_t a0 = svdup_f16(1), a1 = svdup_f16(-2), b0 = svdup_f16(0.5), b1 = svdup_f16(1);
  for (uint64_t i = 0; i < n; ++i)
    for (int u = 0; u < U; ++u) {
      svmopa_za32_f16_m(0, p, p, a0, b0); svmopa_za32_f16_m(1, p, p, a0, b1);
      svmopa_za32_f16_m(2, p, p, a1, b0); svmopa_za32_f16_m(3, p, p, a1, b1);
    }
  return n * U * 4 * (svcntsw() * svcntsw() * 2);
}

__arm_new("za") __arm_locally_streaming static uint64_t k_bf16w(uint64_t n) {
  svbool_t p = svptrue_b16();
  svbfloat16_t a0 = svreinterpret_bf16_s16(svdup_s16(0x3f80)), a1 = svreinterpret_bf16_s16(svdup_s16(0x4000));
  svbfloat16_t b0 = a1, b1 = a0;
  for (uint64_t i = 0; i < n; ++i)
    for (int u = 0; u < U; ++u) {
      svmopa_za32_bf16_m(0, p, p, a0, b0); svmopa_za32_bf16_m(1, p, p, a0, b1);
      svmopa_za32_bf16_m(2, p, p, a1, b0); svmopa_za32_bf16_m(3, p, p, a1, b1);
    }
  return n * U * 4 * (svcntsw() * svcntsw() * 2);
}

__arm_new("za") __arm_locally_streaming static uint64_t k_f32(uint64_t n) {
  svbool_t p = svptrue_b32();
  svfloat32_t a0 = svdup_f32(1), a1 = svdup_f32(-2), b0 = svdup_f32(0.5), b1 = svdup_f32(1);
  for (uint64_t i = 0; i < n; ++i)
    for (int u = 0; u < U; ++u) {
      svmopa_za32_f32_m(0, p, p, a0, b0); svmopa_za32_f32_m(1, p, p, a0, b1);
      svmopa_za32_f32_m(2, p, p, a1, b0); svmopa_za32_f32_m(3, p, p, a1, b1);
    }
  return n * U * 4 * (svcntsw() * svcntsw());
}

typedef uint64_t (*kern_t)(uint64_t);
typedef struct { kern_t k; uint64_t n, macs; } job_t;

static void *worker(void *arg) {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  job_t *j = arg;
  j->macs = j->k(j->n);
  return NULL;
}

static double run(kern_t k, int threads, uint64_t n) {
  pthread_t t[32];
  job_t j[32];
  double t0 = now();
  for (int i = 0; i < threads; ++i) { j[i] = (job_t){k, n, 0}; pthread_create(&t[i], NULL, worker, &j[i]); }
  uint64_t macs = 0;
  for (int i = 0; i < threads; ++i) { pthread_join(t[i], NULL); macs += j[i].macs; }
  return 2.0 * macs / (now() - t0) / 1e12;  // TOPS (2 ops per MAC)
}

int main(int argc, char **argv) {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  struct { const char *name; kern_t k; } ks[] = {
      {"int8 smopa -> int32", k_s8}, {"fp16 fmopa -> fp16 (non-widening)", k_f16},
      {"fp16 fmopa -> fp32 (widening)", k_f16w}, {"bf16 bfmopa -> fp32 (widening)", k_bf16w},
      {"fp32 fmopa -> fp32", k_f32}};
  int nth[] = {1, 2, 3, 4, 6, 8, 12, 18};
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc > 1) {  // debug: one kernel, one thread
    int i = atoi(argv[1]);
    printf("%s: %.3f TOPS\n", ks[i].name, run(ks[i].k, 1, 1000000));
    return 0;
  }
  printf("SVL = %llu bytes\n", (unsigned long long)svcntsb());
  printf("| instruction | %s |\n|---|", "TOPS at 1 / 2 / 3 / 4 / 6 / 8 / 12 / 18 threads");
  printf("\n");
  run(k_s8, 18, ITERS / 4);  // warm-up: clocks up
  for (int i = 0; i < 5; ++i) {
    printf("| %s |", ks[i].name);
    for (int t = 0; t < 8; ++t) {
      double best = 0;
      for (int r = 0; r < 3; ++r) { double v = run(ks[i].k, nth[t], ITERS / 8); if (v > best) best = v; }
      printf(" %.2f", best);
      fflush(stdout);
    }
    printf(" |\n");
  }
  // Accelerate FP32 GEMM for context
  int n = 4096;
  float *A = malloc(sizeof(float) * n * n), *B = malloc(sizeof(float) * n * n), *C = malloc(sizeof(float) * n * n);
  for (int i = 0; i < n * n; ++i) { A[i] = (float)(i % 7) * 0.1f; B[i] = (float)(i % 5) * 0.1f; }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, n, 1, A, n, B, n, 0, C, n);
  double best = 1e9;
  for (int r = 0; r < 5; ++r) {
    double t0 = now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, n, 1, A, n, B, n, 0, C, n);
    double dt = now() - t0;
    if (dt < best) best = dt;
  }
  printf("\nAccelerate cblas_sgemm %d^3: %.2f TFLOPS\n", n, 2.0 * n * (double)n * n / best / 1e12);
  return 0;
}
