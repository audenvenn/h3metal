// Softmax attention with an int8 score product, QR fragments of 16 query rows per SIMD-group.
//
// Same algorithm and layouts as kernels/attention.metal with QK8 (see there), restructured to spend fewer
// instructions per score:
//   - QR = 2: every key or value fragment a SIMD-group loads from threadgroup memory feeds the products of
//     32 query rows instead of 16;
//   - the query row's scale is applied inside the exponent: with t = (q . k) * k_scale and c = q_scale * scale
//     (positive), max over a row of c * t is c * max t, so exp2(c * t - m) is one fused multiply-add.
//
//   Q, K [H|KVH][rows][HD] int8 with row scales qs, ks; V [KVH][k_rows][HD] half; O [rows][...] (O_T)
//
// Specialization: HD (a multiple of 32), WM, BK (a multiple of 32), QR (1 or 2), O_T, KPAD. No causal mask.

#include <metal_stdlib>
#include "nax.h"
#include "params.h"

using namespace metal;

#ifndef O_T
#define O_T half
#endif
#ifndef KPAD
#define KPAD 8
#endif
#ifndef QR
#define QR 2
#endif
#ifndef PVH
#define PVH 0  // 1: each block's P V accumulates in half (32 terms, P <= 1), then adds into the float output
#endif

#define BQ (16 * QR * WM)
#define DF (HD / 16)
#define KF (BK / 16)
#define LDK (HD + KPAD)
#define LDQK (HD + 16)

kernel void attention_q8(device const int8_t *Q [[buffer(0)]],
                         device const int8_t *K [[buffer(1)]],
                         device const half *V [[buffer(2)]],
                         device O_T *O [[buffer(3)]],
                         constant AttnParams &p [[buffer(4)]],
                         device const float *qs [[buffer(5)]],
                         device const float *ksc [[buffer(6)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort tid [[thread_index_in_threadgroup]])
{
  threadgroup int8_t ks[BK * LDQK];
  threadgroup half vs[BK * LDK];
  threadgroup float kscale[BK];
  const int h = int(tg.y);
  const int kvh = h / (p.H / p.KVH);
  const int q0 = int(tg.x) * BQ + int(sg) * 16 * QR;
  const bool active = q0 < p.Lq;
  const short2 sc = nax::coord();
  K += size_t(kvh) * p.k_rows * HD;
  V += size_t(kvh) * p.k_rows * HD;
  const device float *ksh = ksc + size_t(kvh) * p.k_rows;

  nax::frag<int8_t> qf[QR][DF];
  float c0[QR], c1[QR];  // per-row score scales (query scale times the softmax scale, in log2 units)
#pragma clang loop unroll(full)
  for (short r = 0; r < QR; r++) {
    const int qr0 = q0 + r * 16;
    const int lim = max(0, p.Lq - qr0);
    const device int8_t *Qh = Q + (size_t(h) * p.q_rows + min(qr0, p.Lq - 1)) * HD;
#pragma clang loop unroll(full)
    for (short d = 0; d < DF; d++) nax::load_rows(qf[r][d], Qh + d * 16, HD, sc, lim);
    // Rows past Lq get scale 1 (any positive value): their outputs are never stored, and 0 * -inf would be NaN.
    c0[r] = sc.y < lim ? qs[size_t(h) * p.q_rows + qr0 + sc.y] * p.scale : 1.0f;
    c1[r] = sc.y + 8 < lim ? qs[size_t(h) * p.q_rows + qr0 + sc.y + 8] * p.scale : 1.0f;
  }

  nax::frag<float> o[QR][DF];
  float m0[QR], m1[QR], l0[QR], l1[QR];
#pragma clang loop unroll(full)
  for (short r = 0; r < QR; r++) {
#pragma clang loop unroll(full)
    for (short d = 0; d < DF; d++) o[r][d] = 0.0f;
    m0[r] = m1[r] = -INFINITY;
    l0[r] = l1[r] = 0.0f;
  }
  constexpr int NT = WM * 32;
  constexpr int CHUNKS = BK * HD / 8;

  for (int kb = 0; kb < p.L; kb += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (int c = tid; c < CHUNKS; c += NT) {
      const int rr = c / (HD / 8), col = (c % (HD / 8)) * 8;
      *(threadgroup packed_char4 *)(ks + rr * LDQK + col) = *(const device packed_char4 *)(K + size_t(kb + rr) * HD + col);
      *(threadgroup packed_char4 *)(ks + rr * LDQK + col + 4) =
          *(const device packed_char4 *)(K + size_t(kb + rr) * HD + col + 4);
      *(threadgroup half4 *)(vs + rr * LDK + col) = *(const device half4 *)(V + size_t(kb + rr) * HD + col);
      *(threadgroup half4 *)(vs + rr * LDK + col + 4) = *(const device half4 *)(V + size_t(kb + rr) * HD + col + 4);
    }
    if (tid < BK) kscale[tid] = ksh[kb + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) continue;

    // Scores: t = (q . k) * k_scale; the products accumulate in int32.
    nax::frag<float> s[QR][KF];
    {
      nax::frag<int> si[QR][KF];
#pragma clang loop unroll(full)
      for (short r = 0; r < QR; r++) {
#pragma clang loop unroll(full)
        for (short j = 0; j < KF; j++) si[r][j] = 0;
      }
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j += 2) {
#pragma clang loop unroll(full)
        for (short d = 0; d < DF; d++) {
          nax::frag<int8_t> k0, k1;
          nax::load_tg(k0, ks + (j * 16) * LDQK + d * 16, LDQK, sc);
          nax::load_tg(k1, ks + (j * 16 + 16) * LDQK + d * 16, LDQK, sc);
#pragma clang loop unroll(full)
          for (short r = 0; r < QR; r++) nax::mma<int, int8_t, int8_t, false, true>(si[r][j], si[r][j + 1], qf[r][d], k0, k1);
        }
      }
#pragma clang loop unroll(full)
      for (short r = 0; r < QR; r++) {
#pragma clang loop unroll(full)
        for (short j = 0; j < KF; j++) s[r][j] = nax::frag<float>(si[r][j]);
      }
    }
    const bool edge = kb + BK > p.L;
#pragma clang loop unroll(full)
    for (short j = 0; j < KF; j++) {
      const float4 kq = *(const threadgroup float4 *)(kscale + j * 16 + sc.x);
#pragma clang loop unroll(full)
      for (short r = 0; r < QR; r++) {
#pragma clang loop unroll(full)
        for (short e = 0; e < 8; e++) {
          float v = s[r][j][e] * kq[e & 3];
          if (edge && kb + j * 16 + sc.x + (e & 3) >= p.L) v = -INFINITY;
          s[r][j][e] = v;
        }
      }
    }
#pragma clang loop unroll(full)
    for (short r = 0; r < QR; r++) {
      float t0 = -INFINITY, t1 = -INFINITY;
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) nax::row_max(s[r][j], t0, t1);
      const float n0 = max(m0[r], c0[r] * t0), n1 = max(m1[r], c1[r] * t1);
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) {
#pragma clang loop unroll(full)
        for (short e = 0; e < 4; e++) s[r][j][e] = fast::exp2(fma(c0[r], s[r][j][e], -n0));
#pragma clang loop unroll(full)
        for (short e = 4; e < 8; e++) s[r][j][e] = fast::exp2(fma(c1[r], s[r][j][e], -n1));
      }
      const float f0 = fast::exp2(m0[r] - n0), f1 = fast::exp2(m1[r] - n1);
      m0[r] = n0;
      m1[r] = n1;
      l0[r] *= f0;
      l1[r] *= f1;
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) nax::row_sum(s[r][j], l0[r], l1[r]);
      if (simd_any(f0 != 1.0f || f1 != 1.0f)) {
#pragma clang loop unroll(full)
        for (short d = 0; d < DF; d++) {
#pragma clang loop unroll(full)
          for (short e = 0; e < 4; e++) o[r][d][e] *= f0;
#pragma clang loop unroll(full)
          for (short e = 4; e < 8; e++) o[r][d][e] *= f1;
        }
      }
    }
    // P V: each value fragment pair feeds every query fragment.
#if PVH
    nax::frag<half> ph[QR][KF];
#pragma clang loop unroll(full)
    for (short r = 0; r < QR; r++) {
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) ph[r][j] = nax::frag<half>(s[r][j]);
    }
#pragma clang loop unroll(full)
    for (short d = 0; d < DF; d += 2) {
      nax::frag<half> acc0[QR], acc1[QR];
#pragma clang loop unroll(full)
      for (short r = 0; r < QR; r++) {
        acc0[r] = half(0.0f);
        acc1[r] = half(0.0f);
      }
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) {
        nax::frag<half> v0, v1;
        nax::load_tg(v0, vs + (j * 16) * LDK + d * 16, LDK, sc);
        nax::load_tg(v1, vs + (j * 16) * LDK + d * 16 + 16, LDK, sc);
#pragma clang loop unroll(full)
        for (short r = 0; r < QR; r++) nax::mma<half, half, half, false, false>(acc0[r], acc1[r], ph[r][j], v0, v1);
      }
#pragma clang loop unroll(full)
      for (short r = 0; r < QR; r++) {
        o[r][d] += nax::frag<float>(acc0[r]);
        o[r][d + 1] += nax::frag<float>(acc1[r]);
      }
    }
#else
#pragma clang loop unroll(full)
    for (short j = 0; j < KF; j++) {
      nax::frag<half> ph[QR];
#pragma clang loop unroll(full)
      for (short r = 0; r < QR; r++) ph[r] = nax::frag<half>(s[r][j]);
#pragma clang loop unroll(full)
      for (short d = 0; d < DF; d += 2) {
        nax::frag<half> v0, v1;
        nax::load_tg(v0, vs + (j * 16) * LDK + d * 16, LDK, sc);
        nax::load_tg(v1, vs + (j * 16) * LDK + d * 16 + 16, LDK, sc);
#pragma clang loop unroll(full)
        for (short r = 0; r < QR; r++) nax::mma<float, half, half, false, false>(o[r][d], o[r][d + 1], ph[r], v0, v1);
      }
    }
#endif
  }
  if (!active) return;
  const int seq = h / p.seq_heads;
#pragma clang loop unroll(full)
  for (short r = 0; r < QR; r++) {
    const int qr0 = q0 + r * 16;
    if (qr0 >= p.Lq) break;
    const float r0 = 1.0f / l0[r], r1 = 1.0f / l1[r];
    device O_T *out = O + (size_t(seq) * p.seq_rows + qr0) * p.o_stride + (h - seq * p.seq_heads) * HD;
#pragma clang loop unroll(full)
    for (short d = 0; d < DF; d++) {
#pragma clang loop unroll(full)
      for (short e = 0; e < 4; e++) o[r][d][e] *= r0;
#pragma clang loop unroll(full)
      for (short e = 4; e < 8; e++) o[r][d][e] *= r1;
      nax::store_rows(o[r][d], out + d * 16, p.o_stride, sc, p.Lq - qr0);
    }
  }
}
