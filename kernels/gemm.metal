// Matrix products on the GPU neural accelerators (Metal 4 tensor ops).
//
//   v[m][n] = sum_k A[m][k] * W[n][k]          A [M][K] row-major, W [N][K] row-major
//
// Weight formats (WFMT):
//   1  Q8   A int8, W int8,  W[n][k] = sw[n] * q[n][k]               v = sa[m] * sw[n] * (a . q)
//   2  Q4   A int8, W int4,  W[n][k] = sw[n] * q[n][k]               (two codes per byte, even k low)
//   3  Q4G  A int8, W int4,  W[n][k] = sw[k / G][n] * q[n][k]        integer partial products per group
//   4  F16  A half, W half
//   5  BF16 A bfloat, W bfloat
//   6  H8   A half, W int8, W[n][k] = sw[n] * q[n][k]               (activations kept in half precision)
//   7  H4   A half, W int4, W[n][k] = sw[n] * q[n][k]
//   8  H4G  A half, W int4, W[n][k] = sw[k / G][n] * q[n][k]        float partial products per group
// The accelerator multiplies the integer codes itself and accumulates in
// int32; nothing is dequantized in software. Q4G scales each group's int32
// partial product while it is still in registers.
//
// Epilogues (EPI):
//   0  C[m][n] = OUT_T(v + bias[n])
//   1  X[m][n] += gate[type[m]][n] * (v + bias[n])     X float: the residual stream; one gate row per row type
//
// A, C and X are padded to a whole number of TM-row tiles; N is a multiple of TN.
// Specialization: TM, TN, NSG (SIMD-groups per threadgroup), G, OUT_T, HAS_BIAS.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#include "params.h"

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef WFMT
#define WFMT 1
#endif
#ifndef TM
#define TM 64
#endif
#ifndef TN
#define TN 128
#endif
#ifndef NSG
#define NSG 4
#endif
#ifndef EPI
#define EPI 0
#endif
#ifndef OUT_T
#define OUT_T half
#endif
#ifndef HAS_BIAS
#define HAS_BIAS 0
#endif
#ifndef G
#define G 128
#endif
#ifndef SWZ
#define SWZ 1
#endif
#ifndef FAST_LAYOUT
#define FAST_LAYOUT 0
#endif
#ifndef RELAXED
#define RELAXED true
#endif

#if WFMT == 1
#define A_T int8_t
#define W_T int8_t
#define W_PTR device int8_t
#define ACC_T int32_t
#elif WFMT == 2 || WFMT == 3
#define A_T int8_t
#define W_T int4b_format
#define W_PTR device uchar
#define ACC_T int32_t
#elif WFMT == 4
#define A_T half
#define W_T half
#define W_PTR device half
#define ACC_T float
#elif WFMT == 5
#define A_T bfloat
#define W_T bfloat
#define W_PTR device bfloat
#define ACC_T float
#elif WFMT == 6
#define A_T half
#define W_T int8_t
#define W_PTR device int8_t
#define ACC_T float
#else
#define A_T half
#define W_T int4b_format
#define W_PTR device uchar
#define ACC_T float
#endif
#if WFMT == 3 || WFMT == 8
#define GROUPED 1
#define PART_T ACC_T
#else
#define GROUPED 0
#endif

kernel void gemm(device A_T *a [[buffer(0)]],
                 W_PTR *w [[buffer(1)]],
                 device const float *sw [[buffer(2)]],
                 device const float *sa [[buffer(3)]],
                 device void *out [[buffer(4)]],
                 device const float *bias [[buffer(5)]],
                 device const uchar *rtype [[buffer(6)]],
                 device const float *gate [[buffer(7)]],
                 constant GemmParams &p [[buffer(8)]],
                 uint3 tgid [[threadgroup_position_in_grid]],
                 uint3 tgs [[threadgroups_per_grid]])
{
#if SWZ > 1
  // Threadgroups run roughly in dispatch order; remap them so that consecutive
  // ones sweep SWZ row tiles for each column tile, and the weight tile they
  // share is still in the cache when it is reused.
  const int nt = int(tgs.x), mt = int(tgs.y);
  const int t = int(tgid.y) * nt + int(tgid.x);
  const int band = t / (SWZ * nt);
  const int rows = min(SWZ, mt - band * SWZ);
  const int in_band = t - band * SWZ * nt;
  const int m_tile = band * SWZ + in_band % rows;
  const int n_tile = in_band / rows;
  const int n0 = n_tile * TN;
  const int m0 = m_tile * TM;
#else
  const int n0 = int(tgid.x) * TN;
  const int m0 = int(tgid.y) * TM;
#endif

  array<int, 2> a_str = {1, p.a_stride};
  array<int, 2> w_str = {1, p.w_stride};
  tensor<device A_T, dextents<int, 2>, tensor_inline> A(a, dextents<int, 2>(p.K, p.M), a_str);
  tensor<device W_T, dextents<int, 2>, tensor_inline> W(w, dextents<int, 2>(p.K, p.N), w_str);

#if GROUPED
  constexpr auto desc = matmul2d_descriptor(TM, TN, G, false, true, RELAXED, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto tA = A.slice<G, TM>(0, m0);
  auto tW = W.slice<G, TN>(0, n0);
  auto part = op.get_destination_cooperative_tensor<decltype(tA), decltype(tW), PART_T>();
  auto acc = op.get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  }
#if FAST_LAYOUT
  // The accelerator's destination layout on this GPU generation (checked by
  // tests/test_gemm.py and at start-up): element i of a thread sits in
  // column base + i % 4 + 32 * ((i / 8) % 4), so a thread needs the scales of
  // four runs of four adjacent columns per group: four 16-byte loads.
  // Scales are stored group-major, [K/G][N].
  int cb[4];
#pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) cb[j] = n0 + int(acc.get_multidimensional_index(8 * j)[0]);
  for (int g = 0; g < p.groups; ++g) {
    tA = A.slice<G, TM>(g * G, m0);
    tW = W.slice<G, TN>(g * G, n0);
    op.run(tA, tW, part);
    const device float *sg = sw + size_t(g) * p.N;
    float4 s4[4];
#pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) s4[j] = *(const device float4 *)(sg + cb[j]);
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] += float(part[i]) * s4[(i / 8) % 4][i % 4];
  }
#else
  for (int g = 0; g < p.groups; ++g) {
    tA = A.slice<G, TM>(g * G, m0);
    tW = W.slice<G, TN>(g * G, n0);
    op.run(tA, tW, part);
    const device float *sg = sw + size_t(g) * p.N + n0;
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (acc.is_valid_element(i)) {
        auto idx = acc.get_multidimensional_index(i);  // {column, row} within the tile
        acc[i] += float(part[i]) * sg[idx[0]];
      }
    }
  }
#endif
#else
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, RELAXED,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto tA = A.slice(0, m0);
  auto tW = W.slice(0, n0);
  auto acc = op.get_destination_cooperative_tensor<decltype(tA), decltype(tW), ACC_T>();
  op.run(tA, tW, acc);
#endif

#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto idx = acc.get_multidimensional_index(i);
    const int n = n0 + int(idx[0]);
    const int m = m0 + int(idx[1]);
#if WFMT == 1 || WFMT == 2
    float v = float(acc[i]) * sa[m] * sw[n];
#elif WFMT == 3
    float v = acc[i] * sa[m];
#elif WFMT == 8
    float v = acc[i];
#elif WFMT == 6 || WFMT == 7
    float v = acc[i] * sw[n];
#else
    float v = float(acc[i]);
#endif
#if HAS_BIAS
    v += bias[n];
#endif
#if EPI == 0
    ((device OUT_T *)out)[size_t(m) * p.c_stride + n] = OUT_T(v);
#else
    device float *x = (device float *)out + size_t(m) * p.c_stride + n;
    *x += gate[size_t(rtype[m]) * p.g_stride + n] * v;
#endif
  }
}
