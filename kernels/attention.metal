// Softmax attention on the GPU neural accelerators, flash style: the scores
// of a block of keys live in registers only, with a running row maximum and
// sum (online softmax), so nothing of size queries x keys is ever written.
//
//   Q [H][q_rows][HD], K and V [KVH][k_rows][HD] (half), O [rows][...] (O_T)
//
// One SIMD-group owns 16 query rows of one head; a threadgroup holds WM of
// them (BQ = 16 * WM rows) and walks the keys in blocks of BK, which it stages
// in threadgroup memory once for all its SIMD-groups (rows padded by KPAD to
// spread them over the memory banks). Queries stay in registers. Both products
// (Q K^T and P V) run on the accelerator with operands held in registers
// (kernels/nax.h). Keys at or beyond L are masked; with CAUSAL, query row i
// sees keys 0 .. i + causal_off. Head h writes its output at row
// (h / seq_heads) * seq_rows + i, columns (h % seq_heads) * HD: a batch of
// short sequences (the VAE's tiles) runs as one dispatch.
//
// Specialization: HD (head dim, a multiple of 32), WM, BK (a multiple of 32), CAUSAL, O_T, KPAD.

#include <metal_stdlib>
#include "nax.h"
#include "params.h"

using namespace metal;

#ifndef CAUSAL
#define CAUSAL 0
#endif
#ifndef O_T
#define O_T half
#endif
#ifndef KPAD
#define KPAD 8
#endif

#define BQ (16 * WM)
#define DF (HD / 16)
#define KF (BK / 16)
#define LDK (HD + KPAD)

#ifndef QK8
#define QK8 0
#endif
#ifndef DB
#define DB 0  // QK8 only: double-buffered keys and values, the next block's loads in flight during this block
#endif
#ifndef PVORDER
#define PVORDER 0  // 1: P V with the output dimensions in the outer loop
#endif
#if DB && !QK8
#error "DB needs QK8 (two buffers of half keys do not fit in threadgroup memory)"
#endif
#if QK8
#define QK_T int8_t
#define LDQK (HD + 16)
#else
#define QK_T half
#define LDQK LDK
#endif

kernel void attention(device const QK_T *Q [[buffer(0)]],
                         device const QK_T *K [[buffer(1)]],
                         device const half *V [[buffer(2)]],
                         device O_T *O [[buffer(3)]],
                         constant AttnParams &p [[buffer(4)]],
                         device const float *qs [[buffer(5)]],
                         device const float *ksc [[buffer(6)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort tid [[thread_index_in_threadgroup]])
{
  threadgroup QK_T ks_[DB + 1][BK * LDQK];
  threadgroup half vs_[DB + 1][BK * LDK];
#if QK8
  threadgroup float kscale_[DB + 1][BK];
#endif
  const int h = int(tg.y);
  const int kvh = h / (p.H / p.KVH);
  const int q0 = int(tg.x) * BQ + int(sg) * 16;
  const bool active = q0 < p.Lq;
  const short2 sc = nax::coord();
  const device QK_T *Qh = Q + (size_t(h) * p.q_rows + min(q0, p.Lq - 1)) * HD;
  K += size_t(kvh) * p.k_rows * HD;
  V += size_t(kvh) * p.k_rows * HD;

  nax::frag<QK_T> qf[DF];
#pragma clang loop unroll(full)
  for (short d = 0; d < DF; d++) nax::load_rows(qf[d], Qh + d * 16, HD, sc, active ? p.Lq - q0 : 0);
#if QK8
  const int lim = active ? p.Lq - q0 : 0;
  const float qsc0 = sc.y < lim ? qs[size_t(h) * p.q_rows + q0 + sc.y] * p.scale : 0.0f;
  const float qsc1 = sc.y + 8 < lim ? qs[size_t(h) * p.q_rows + q0 + sc.y + 8] * p.scale : 0.0f;
  const device float *ksh = ksc + size_t(kvh) * p.k_rows;
#endif

  nax::frag<float> o[DF];
#pragma clang loop unroll(full)
  for (short d = 0; d < DF; d++) o[d] = 0.0f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  int kend = p.L;
#if CAUSAL
  kend = min(p.L, int(tg.x) * BQ + BQ + p.causal_off);
#endif
  const int row0 = q0 + sc.y, row1 = row0 + 8;
  constexpr int NT = WM * 32;              // threads
  constexpr int CHUNKS = BK * HD / 8;      // 16-byte chunks per block

#if DB
  constexpr int PER = (CHUNKS + NT - 1) / NT;
#pragma clang loop unroll(full)
  for (int i = 0; i < PER; i++) {  // block 0 into buffer 0
    const int c = tid + i * NT;
    if (c < CHUNKS) {
      const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
      *(threadgroup packed_char4 *)(ks_[0] + r * LDQK + col) = *(const device packed_char4 *)(K + size_t(r) * HD + col);
      *(threadgroup packed_char4 *)(ks_[0] + r * LDQK + col + 4) = *(const device packed_char4 *)(K + size_t(r) * HD + col + 4);
      *(threadgroup half4 *)(vs_[0] + r * LDK + col) = *(const device half4 *)(V + size_t(r) * HD + col);
      *(threadgroup half4 *)(vs_[0] + r * LDK + col + 4) = *(const device half4 *)(V + size_t(r) * HD + col + 4);
    }
  }
  if (tid < BK) kscale_[0][tid] = ksh[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int cur = 0;
#endif
  for (int kb = 0; kb < kend; kb += BK) {
#if DB
    const bool more = kb + BK < kend;
    packed_char4 rk0[PER], rk1[PER];
    half4 rv0[PER], rv1[PER];
    float rks = 0.0f;
    if (more) {
#pragma clang loop unroll(full)
      for (int i = 0; i < PER; i++) {
        const int c = tid + i * NT;
        if (c < CHUNKS) {
          const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
          const size_t at = size_t(kb + BK + r) * HD + col;
          rk0[i] = *(const device packed_char4 *)(K + at);
          rk1[i] = *(const device packed_char4 *)(K + at + 4);
          rv0[i] = *(const device half4 *)(V + at);
          rv1[i] = *(const device half4 *)(V + at + 4);
        }
      }
      if (tid < BK) rks = ksh[kb + BK + tid];
    }
    const threadgroup QK_T *ks = ks_[cur];
    const threadgroup half *vs = vs_[cur];
    const threadgroup float *kscale = kscale_[cur];
    if (active) {
#else
    threadgroup QK_T *ks = ks_[0];
    threadgroup half *vs = vs_[0];
#if QK8
    threadgroup float *kscale = kscale_[0];
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (int c = tid; c < CHUNKS; c += NT) {
      const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
#if QK8
      *(threadgroup packed_char4 *)(ks + r * LDQK + col) = *(const device packed_char4 *)(K + size_t(kb + r) * HD + col);
      *(threadgroup packed_char4 *)(ks + r * LDQK + col + 4) = *(const device packed_char4 *)(K + size_t(kb + r) * HD + col + 4);
#else
      *(threadgroup half4 *)(ks + r * LDK + col) = *(const device half4 *)(K + size_t(kb + r) * HD + col);
      *(threadgroup half4 *)(ks + r * LDK + col + 4) = *(const device half4 *)(K + size_t(kb + r) * HD + col + 4);
#endif
      *(threadgroup half4 *)(vs + r * LDK + col) = *(const device half4 *)(V + size_t(kb + r) * HD + col);
      *(threadgroup half4 *)(vs + r * LDK + col + 4) = *(const device half4 *)(V + size_t(kb + r) * HD + col + 4);
    }
#if QK8
    if (tid < BK) kscale[tid] = ksh[kb + tid];
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) continue;
    {
#endif

    nax::frag<float> s[KF];
#if QK8
    {
      nax::frag<int> si[KF];
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) si[j] = 0;
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j += 2) {
#pragma clang loop unroll(full)
        for (short d = 0; d < DF; d++) {
          nax::frag<int8_t> k0, k1;
          nax::load_tg(k0, ks + (j * 16) * LDQK + d * 16, LDQK, sc);
          nax::load_tg(k1, ks + (j * 16 + 16) * LDQK + d * 16, LDQK, sc);
          nax::mma<int, int8_t, int8_t, false, true>(si[j], si[j + 1], qf[d], k0, k1);
        }
      }
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) {
        const float4 kq = *(const threadgroup float4 *)(kscale + j * 16 + sc.x);
#pragma clang loop unroll(full)
        for (short e = 0; e < 8; e++) s[j][e] = float(si[j][e]) * ((e < 4 ? qsc0 : qsc1) * kq[e & 3]);
      }
    }
#else
#pragma clang loop unroll(full)
    for (short j = 0; j < KF; j++) s[j] = 0.0f;
#pragma clang loop unroll(full)
    for (short j = 0; j < KF; j += 2) {
#pragma clang loop unroll(full)
      for (short d = 0; d < DF; d++) {
        nax::frag<half> k0, k1;
        nax::load_tg(k0, ks + (j * 16) * LDK + d * 16, LDK, sc);
        nax::load_tg(k1, ks + (j * 16 + 16) * LDK + d * 16, LDK, sc);
        nax::mma<float, half, half, false, true>(s[j], s[j + 1], qf[d], k0, k1);
      }
    }
#endif
    const bool edge = kb + BK > p.L || CAUSAL;
#pragma clang loop unroll(full)
    for (short j = 0; j < KF; j++) {
#pragma clang loop unroll(full)
      for (short e = 0; e < 8; e++) {
#if QK8
        float v = s[j][e];
#else
        float v = s[j][e] * p.scale;
#endif
        if (edge) {
          const int key = kb + j * 16 + sc.x + (e & 3);
          bool masked = key >= p.L;
#if CAUSAL
          masked = masked || key > (e < 4 ? row0 : row1) + p.causal_off;
#endif
          if (masked) v = -INFINITY;
        }
        s[j][e] = v;
      }
    }
    float n0 = m0, n1 = m1;
#pragma clang loop unroll(full)
    for (short j = 0; j < KF; j++) nax::row_max(s[j], n0, n1);
#pragma clang loop unroll(full)
    for (short j = 0; j < KF; j++) {
#pragma clang loop unroll(full)
      for (short e = 0; e < 4; e++) s[j][e] = fast::exp2(s[j][e] - n0);
#pragma clang loop unroll(full)
      for (short e = 4; e < 8; e++) s[j][e] = fast::exp2(s[j][e] - n1);
    }
    const float f0 = fast::exp2(m0 - n0), f1 = fast::exp2(m1 - n1);
    m0 = n0;
    m1 = n1;
    l0 *= f0;
    l1 *= f1;
#pragma clang loop unroll(full)
    for (short j = 0; j < KF; j++) nax::row_sum(s[j], l0, l1);
    // The output only needs rescaling when a row's running maximum moved; after the first blocks it
    // rarely does, and skipping the multiply by exactly 1 changes nothing.
    if (simd_any(f0 != 1.0f || f1 != 1.0f)) {
#pragma clang loop unroll(full)
    for (short d = 0; d < DF; d++) {
#pragma clang loop unroll(full)
      for (short e = 0; e < 4; e++) o[d][e] *= f0;
#pragma clang loop unroll(full)
      for (short e = 4; e < 8; e++) o[d][e] *= f1;
    }
    }
#if PVORDER
    nax::frag<half> ph[KF];
#pragma clang loop unroll(full)
    for (short j = 0; j < KF; j++) ph[j] = nax::frag<half>(s[j]);
#pragma clang loop unroll(full)
    for (short d = 0; d < DF; d += 2) {
#pragma clang loop unroll(full)
      for (short j = 0; j < KF; j++) {
        nax::frag<half> v0, v1;
        nax::load_tg(v0, vs + (j * 16) * LDK + d * 16, LDK, sc);
        nax::load_tg(v1, vs + (j * 16) * LDK + d * 16 + 16, LDK, sc);
        nax::mma<float, half, half, false, false>(o[d], o[d + 1], ph[j], v0, v1);
      }
    }
#else
#pragma clang loop unroll(full)
    for (short j = 0; j < KF; j++) {
      nax::frag<half> ph = nax::frag<half>(s[j]);
#pragma clang loop unroll(full)
      for (short d = 0; d < DF; d += 2) {
        nax::frag<half> v0, v1;
        nax::load_tg(v0, vs + (j * 16) * LDK + d * 16, LDK, sc);
        nax::load_tg(v1, vs + (j * 16) * LDK + d * 16 + 16, LDK, sc);
        nax::mma<float, half, half, false, false>(o[d], o[d + 1], ph, v0, v1);
      }
    }
#endif
    }  // compute
#if DB
    if (more) {  // the next block into the other buffer
#pragma clang loop unroll(full)
      for (int i = 0; i < PER; i++) {
        const int c = tid + i * NT;
        if (c < CHUNKS) {
          const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
          *(threadgroup packed_char4 *)(ks_[cur ^ 1] + r * LDQK + col) = rk0[i];
          *(threadgroup packed_char4 *)(ks_[cur ^ 1] + r * LDQK + col + 4) = rk1[i];
          *(threadgroup half4 *)(vs_[cur ^ 1] + r * LDK + col) = rv0[i];
          *(threadgroup half4 *)(vs_[cur ^ 1] + r * LDK + col + 4) = rv1[i];
        }
      }
      if (tid < BK) kscale_[cur ^ 1][tid] = rks;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur ^= 1;
#endif
  }
  if (!active) return;
  const float r0 = 1.0f / l0, r1 = 1.0f / l1;
  const int seq = h / p.seq_heads;
  device O_T *out = O + (size_t(seq) * p.seq_rows + q0) * p.o_stride + (h - seq * p.seq_heads) * HD;
#pragma clang loop unroll(full)
  for (short d = 0; d < DF; d++) {
#pragma clang loop unroll(full)
    for (short e = 0; e < 4; e++) o[d][e] *= r0;
#pragma clang loop unroll(full)
    for (short e = 4; e < 8; e++) o[d][e] *= r1;
    nax::store_rows(o[d], out + d * 16, p.o_stride, sc, p.Lq - q0);
  }
}
