// Probe the GPU neural accelerators' matmul rates per operand type, at the
// shapes of the MiniMax-H3-Image transformer (derived from raceline's nax_probe).
//
// Metal 4's MetalPerformancePrimitives `matmul2d` runs on the per-core
// accelerators of M5/M6-class GPUs and accepts *quantized* right operands
// natively (int4b, int2b, FP4, FP8, int8), optionally with a UE8M0 scale
// plane (one power-of-two scale per 32 elements). This measures how fast each
// operand combination really is, so the weight format can be chosen to match
// the hardware instead of being dequantized in software first.
//
// build: clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation \
//          -o build/hw/nax_probe bench/hw/nax_probe.mm
// run:   build/hw/nax_probe [--check] [--quick]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static NSString *const kSource = @R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Params { int M; int N; int K; int a_stride; int w_stride; int c_stride; };

// C[M][N] = A[M][K] * W[N][K]^T.  A, W, C are row-major; tensor extents list
// the innermost (contiguous) dimension first.
kernel void mm(device A_STORE *a [[buffer(0)]],
               device W_STORE *w [[buffer(1)]],
               device C_T *c [[buffer(2)]],
               constant Params &p [[buffer(3)]],
#if HAS_SCALES
               device uchar *s [[buffer(4)]],
#endif
               uint3 tgid [[threadgroup_position_in_grid]])
{
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
                                            false, BLOCKED ? false : true, RELAXED);
  matmul2d<desc, execution_simdgroups<NSG>> op;

  array<int, 2> a_str = {1, p.a_stride};
  array<int, 2> w_str = {1, p.w_stride};
  array<int, 2> c_str = {1, p.c_stride};

  tensor<device A_T, dextents<int, 2>, tensor_inline> A(a, dextents<int, 2>(p.K, p.M), a_str);
#if BLOCKED
  // Blocked layout: W is stored per block of BLOCK output rows as [K][BLOCK],
  // so one threadgroup streams a single contiguous region (BLOCK*bits/8 bytes
  // per k) instead of TN separate row streams.
  const int blk = (int(tgid.x) * TN) / BLOCK;
  const int col = (int(tgid.x) * TN) % BLOCK;
  array<int, 2> wb_str = {1, BLOCK};
  tensor<device W_T, dextents<int, 2>, tensor_inline> Wb(
      w + size_t(blk) * size_t(p.K) * (BLOCK * W_BITS / 8), dextents<int, 2>(BLOCK, p.K), wb_str);
  tensor<device C_T, dextents<int, 2>, tensor_inline> C(c, dextents<int, 2>(p.N, p.M), c_str);
  auto tA = A.slice(0, int(tgid.y) * TM);
  auto tW = Wb.slice(col, 0);
  auto tC = C.slice(int(tgid.x) * TN, int(tgid.y) * TM);
  op.run(tA, tW, tC);
#else
#if HAS_SCALES
  using scales_t = tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
  scales_t sc(s);
  tensor<device W_T, dextents<int, 2>, tensor_inline, scales_t> W(w, dextents<int, 2>(p.K, p.N), w_str, sc);
#else
  tensor<device W_T, dextents<int, 2>, tensor_inline> W(w, dextents<int, 2>(p.K, p.N), w_str);
#endif
  tensor<device C_T, dextents<int, 2>, tensor_inline> C(c, dextents<int, 2>(p.N, p.M), c_str);

  auto tA = A.slice(0, int(tgid.y) * TM);
  auto tW = W.slice(0, int(tgid.x) * TN);
  auto tC = C.slice(int(tgid.x) * TN, int(tgid.y) * TM);
  op.run(tA, tW, tC);
#endif
}
)MSL";

struct Combo {
  const char *name;
  const char *a_t, *a_store;   // left element type and its storage pointee
  const char *w_t, *w_store;   // right element type and its storage pointee
  const char *c_t;
  int a_bits, w_bits, c_bytes;
  bool scales;
};

static const Combo kCombos[] = {
    {"half  x half        -> half", "half", "half", "half", "half", "half", 16, 16, 2, false},
    {"half  x half        -> float", "half", "half", "half", "half", "float", 16, 16, 4, false},
    {"float x float       -> float", "float", "float", "float", "float", "float", 32, 32, 4, false},
    {"half  x int8        -> half", "half", "half", "int8_t", "int8_t", "half", 16, 8, 2, false},
    {"half  x fp8_e4m3    -> half", "half", "half", "metal_fp8_e4m3_format", "uchar", "half", 16, 8, 2, false},
    {"half  x int4b       -> half", "half", "half", "int4b_format", "uchar", "half", 16, 4, 2, false},
    {"half  x int4b       -> float", "half", "half", "int4b_format", "uchar", "float", 16, 4, 4, false},
    {"half  x int4b+ue8m0 -> half", "half", "half", "int4b_format", "uchar", "half", 16, 4, 2, true},
    {"half  x fp4_e2m1    -> half", "half", "half", "metal_fp4_e2m1_format", "uchar", "half", 16, 4, 2, false},
    {"half  x fp4+ue8m0   -> half", "half", "half", "metal_fp4_e2m1_format", "uchar", "half", 16, 4, 2, true},
    {"half  x int2b       -> half", "half", "half", "int2b_format", "uchar", "half", 16, 2, 2, false},
    {"half  x int2b+ue8m0 -> half", "half", "half", "int2b_format", "uchar", "half", 16, 2, 2, true},
    {"int8  x int8        -> int32", "int8_t", "int8_t", "int8_t", "int8_t", "int32_t", 8, 8, 4, false},
    {"int8  x int4b       -> int32", "int8_t", "int8_t", "int4b_format", "uchar", "int32_t", 8, 4, 4, false},
    {"int8  x int2b       -> int32", "int8_t", "int8_t", "int2b_format", "uchar", "int32_t", 8, 2, 4, false},
    {"fp8   x fp8_e4m3    -> half", "metal_fp8_e4m3_format", "uchar", "metal_fp8_e4m3_format", "uchar", "half", 8, 8, 2, false},
    {"fp4   x fp4_e2m1    -> half", "metal_fp4_e2m1_format", "uchar", "metal_fp4_e2m1_format", "uchar", "half", 4, 4, 2, false},
};

struct Tile { int tm, tn, nsg; };

static id<MTLDevice> gDev;
static id<MTLCommandQueue> gQueue;

static int gBlock = 0;
static std::vector<int> gMs = {1280, 4352, 16640};
static std::vector<int> gCombos;  // empty = all  // 0 = row-major W[N][K]; else block size of the blocked layout

static id<MTLComputePipelineState> build(const Combo &c, Tile t, bool relaxed, std::string *err) {
  MTLCompileOptions *opt = [MTLCompileOptions new];
  opt.languageVersion = MTLLanguageVersion4_1;
  opt.preprocessorMacros = @{
    @"A_T" : @(c.a_t), @"A_STORE" : @(c.a_store), @"W_T" : @(c.w_t), @"W_STORE" : @(c.w_store),
    @"C_T" : @(c.c_t), @"HAS_SCALES" : @(c.scales ? 1 : 0),
    @"TM" : @(t.tm), @"TN" : @(t.tn), @"NSG" : @(t.nsg), @"RELAXED" : relaxed ? @"true" : @"false",
    @"BLOCKED" : @(gBlock ? 1 : 0), @"BLOCK" : @(gBlock ? gBlock : 256), @"W_BITS" : @(c.w_bits),
  };
  NSError *e = nil;
  id<MTLLibrary> lib = [gDev newLibraryWithSource:kSource options:opt error:&e];
  if (!lib) { *err = e.localizedDescription.UTF8String; return nil; }
  id<MTLFunction> fn = [lib newFunctionWithName:@"mm"];
  id<MTLComputePipelineState> pso = [gDev newComputePipelineStateWithFunction:fn error:&e];
  if (!pso) { *err = e.localizedDescription.UTF8String; return nil; }
  return pso;
}

struct Params { int M, N, K, a_stride, w_stride, c_stride; };

static size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

struct Problem {
  int M, N, K;
  id<MTLBuffer> a, w, c, s, p;
  size_t w_row_bytes;
};

static Problem make_problem(const Combo &c, int M, int N, int K) {
  Problem pr{M, N, K};
  size_t a_bytes = (size_t)M * K * c.a_bits / 8;
  pr.w_row_bytes = round_up((size_t)K * c.w_bits / 8, 128);
  size_t w_bytes = pr.w_row_bytes * N;
  size_t c_bytes = (size_t)M * N * c.c_bytes;
  pr.a = [gDev newBufferWithLength:round_up(a_bytes, 16384) options:MTLResourceStorageModeShared];
  pr.w = [gDev newBufferWithLength:round_up(w_bytes, 16384) options:MTLResourceStorageModeShared];
  pr.c = [gDev newBufferWithLength:round_up(c_bytes, 16384) options:MTLResourceStorageModeShared];
  pr.p = [gDev newBufferWithLength:sizeof(Params) options:MTLResourceStorageModeShared];
  Params p{M, N, K, K, (int)(pr.w_row_bytes * 8 / c.w_bits), N};
  memcpy(pr.p.contents, &p, sizeof(p));
  if (c.scales) {
    pr.s = [gDev newBufferWithLength:round_up((size_t)N * (K / 32), 16384) options:MTLResourceStorageModeShared];
    memset(pr.s.contents, 127, pr.s.length);  // UE8M0 127 == 2^0
  }
  // Left operand: small values so nothing overflows half.
  uint64_t x = 0x243f6a8885a308d3ULL;
  auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
  if (c.a_bits == 16) {
    _Float16 *a = (_Float16 *)pr.a.contents;
    for (size_t i = 0; i < (size_t)M * K; i++) a[i] = (_Float16)(((int)(rnd() % 2001) - 1000) / 8000.0f);
  } else if (c.a_bits == 32) {
    float *a = (float *)pr.a.contents;
    for (size_t i = 0; i < (size_t)M * K; i++) a[i] = ((int)(rnd() % 2001) - 1000) / 8000.0f;
  } else {
    uint8_t *a = (uint8_t *)pr.a.contents;
    for (size_t i = 0; i < a_bytes; i++) a[i] = (uint8_t)(rnd() & 0x33);  // small codes
  }
  if (c.w_bits == 16) {
    _Float16 *w = (_Float16 *)pr.w.contents;
    for (size_t i = 0; i < w_bytes / 2; i++) w[i] = (_Float16)(((int)(rnd() % 2001) - 1000) / 8000.0f);
  } else if (c.w_bits == 32) {
    float *w = (float *)pr.w.contents;
    for (size_t i = 0; i < w_bytes / 4; i++) w[i] = ((int)(rnd() % 2001) - 1000) / 8000.0f;
  } else {
    uint8_t *w = (uint8_t *)pr.w.contents;
    uint8_t mask = strstr(c.w_t, "fp8") ? 0x3f : 0xff;  // keep FP8 codes finite and small
    for (size_t i = 0; i < w_bytes; i++) w[i] = (uint8_t)rnd() & mask;
  }
  return pr;
}

// Returns best GPU time in seconds for one dispatch. Each command buffer holds
// `inner` back-to-back dispatches: a lone sub-millisecond kernel is timed while
// the GPU is still ramping its clock, which is not what a decode loop sees.
static double run(id<MTLComputePipelineState> pso, const Problem &pr, Tile t, int reps, int inner = 1) {
  double best = 1e30;
  for (int r = 0; r < reps + 1; r++) {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:pr.a offset:0 atIndex:0];
    [enc setBuffer:pr.w offset:0 atIndex:1];
    [enc setBuffer:pr.c offset:0 atIndex:2];
    [enc setBuffer:pr.p offset:0 atIndex:3];
    if (pr.s) [enc setBuffer:pr.s offset:0 atIndex:4];
    MTLSize groups = MTLSizeMake((pr.N + t.tn - 1) / t.tn, (pr.M + t.tm - 1) / t.tm, 1);
    for (int i = 0; i < inner; i++)
      [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(32 * t.nsg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return -1;
    double dt = (cb.GPUEndTime - cb.GPUStartTime) / inner;
    if (r > 0 && dt < best) best = dt;
  }
  return best;
}

static void throughput(bool quick) {
  struct Shape { int M, K, N; const char *what; };
  std::vector<Shape> shapes;
  for (int M : gMs) {
    shapes.push_back({M, 5376, 21504, "DiT qkv"});
    shapes.push_back({M, 7168, 5376, "DiT out"});
    shapes.push_back({M, 5376, 28672, "DiT fc1"});
    shapes.push_back({M, 14336, 5376, "DiT fc2"});
  }
  std::vector<Tile> tiles = {{64, 64, 4}, {128, 64, 4}, {64, 128, 4}, {128, 128, 4}, {128, 64, 8},
                             {64, 128, 8}, {128, 128, 8}, {256, 128, 8}, {128, 256, 8}, {256, 128, 16}};
  if (quick) { tiles = {{128, 64, 4}, {128, 128, 8}}; }

  for (const Shape &sh : shapes) {
    printf("\n== M=%d K=%d N=%d  (%s) ==\n", sh.M, sh.K, sh.N, sh.what);
    printf("%-30s %-14s %9s %8s\n", "operands", "tile(m,n,sg)", "ms", "TFLOPS");
    for (int ci = 0; ci < (int)(sizeof kCombos / sizeof kCombos[0]); ci++) {
      if (!gCombos.empty() && std::find(gCombos.begin(), gCombos.end(), ci) == gCombos.end()) continue;
      const Combo &c = kCombos[ci];
      double best_tf = 0, best_ms = 0; Tile best_t{};
      std::string first_err;
      for (Tile t : tiles) {
        std::string err;
        id<MTLComputePipelineState> pso = build(c, t, true, &err);
        if (!pso) { if (first_err.empty()) first_err = err; continue; }
        if (32 * t.nsg > (int)pso.maxTotalThreadsPerThreadgroup) continue;
        Problem pr = make_problem(c, sh.M, sh.N, sh.K);
        double dt = run(pso, pr, t, 4);
        if (dt <= 0) continue;
        double tf = 2.0 * sh.M * sh.K * sh.N / dt / 1e12;
        if (tf > best_tf) { best_tf = tf; best_ms = dt * 1e3; best_t = t; }
      }
      if (best_tf > 0) {
        char tb[32]; snprintf(tb, sizeof tb, "%d,%d,%d", best_t.tm, best_t.tn, best_t.nsg);
        printf("%-30s %-14s %9.2f %8.2f\n", c.name, tb, best_ms, best_tf);
      } else {
        std::string e = first_err.substr(0, 150);
        for (char &ch : e) if (ch == '\n') ch = ' ';
        printf("%-30s FAILED: %s\n", c.name, e.c_str());
      }
      fflush(stdout);
    }
  }
}

// Small-M regime: single-token decode and speculative verification. The weights
// are streamed once whatever M is, so the yardstick is bytes of W per second
// against the ~150 GB/s the memory system can deliver.
static void small_m() {
  struct Shape { int K, N; const char *what; };
  const Shape shapes[] = {{5120, 17408, "27B gate/up"}, {17408, 5120, "27B down"}, {2560, 9216, "4B gate/up"}};
  const int combo_idx[] = {6, 13, 3, 1};  // half x int4b -> float, int8 x int4b, half x int8, half x half -> float
  const int Ms[] = {1, 4, 8, 16, 32, 64, 128};
  const Tile tiles[] = {{1, 32, 1}, {4, 32, 1}, {8, 32, 1}, {16, 32, 1}, {16, 64, 2}, {16, 128, 4}, {16, 256, 8},
                        {32, 32, 1}, {32, 64, 2}, {32, 128, 4}, {64, 32, 2}, {64, 64, 4}, {64, 128, 8}, {128, 64, 8}};
  for (const Shape &sh : shapes) {
    printf("\n== small M: K=%d N=%d (%s) ==\n", sh.K, sh.N, sh.what);
    printf("%-30s %5s %-14s %8s %8s %8s\n", "operands", "M", "tile(m,n,sg)", "ms", "W GB/s", "TFLOPS");
    for (int ci : combo_idx) {
      const Combo &c = kCombos[ci];
      for (int M : Ms) {
        double best = 1e30; Tile bt{};
        for (Tile t : tiles) {
          if (t.tm < M && t.tm < 16) continue;        // tiny tiles only for tiny M
          if (t.tm > 4 * M && t.tm > 16) continue;    // don't pad absurdly
          std::string err;
          id<MTLComputePipelineState> pso = build(c, t, true, &err);
          if (!pso || 32 * t.nsg > (int)pso.maxTotalThreadsPerThreadgroup) continue;
          int m_alloc = (M + t.tm - 1) / t.tm * t.tm;
          Problem pr = make_problem(c, m_alloc, sh.N, sh.K);
          double dt = run(pso, pr, t, 3, 40);
          if (dt > 0 && dt < best) { best = dt; bt = t; }
        }
        if (best < 1e29) {
          char tb[32]; snprintf(tb, sizeof tb, "%d,%d,%d", bt.tm, bt.tn, bt.nsg);
          double w_bytes = (double)sh.N * sh.K * c.w_bits / 8;
          printf("%-30s %5d %-14s %8.3f %8.1f %8.2f\n", c.name, M, tb, best * 1e3,
                 w_bytes / best / 1e9, 2.0 * M * sh.K * sh.N / best / 1e12);
        } else {
          printf("%-30s %5d (no working tile)\n", c.name, M);
        }
        fflush(stdout);
      }
    }
  }
}

int main(int argc, char **argv) {
  bool quick = false, small = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--quick")) quick = true;
    if (!strcmp(argv[i], "--small")) small = true;
    if (!strcmp(argv[i], "--block") && i + 1 < argc) gBlock = atoi(argv[++i]);
    if (!strcmp(argv[i], "--m") && i + 1 < argc) { gMs.clear(); for (char *t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) gMs.push_back(atoi(t)); }
    if (!strcmp(argv[i], "--combos") && i + 1 < argc) { for (char *t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) gCombos.push_back(atoi(t)); }
  }
  @autoreleasepool {
    gDev = MTLCreateSystemDefaultDevice();
    gQueue = [gDev newCommandQueue];
    printf("device: %s   W layout: %s\n", gDev.name.UTF8String,
           gBlock ? "blocked [N/B][K][B]" : "row-major [N][K]");
    if (small) small_m(); else throughput(quick);
  }
  return 0;
}
