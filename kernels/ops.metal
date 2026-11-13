// Element-wise and row-wise kernels around the matrix products.
//
// Rotation: the int8 inputs of the quantized products are rotated by a
// block-diagonal Walsh-Hadamard matrix (blocks of 256 channels, scaled by
// 1/16 so it is orthonormal). The weights were rotated the same way when the
// model was converted, so the product is unchanged while outlier channels are
// spread over their block before the per-row int8 quantization. Inside a
// block, lane l of a SIMD-group holds channels 8l .. 8l+7: three butterfly
// stages run in registers, five across lanes.

#include <metal_stdlib>

#include "params.h"

using namespace metal;

#define RB 256  // rotation block

static inline void fwht8(thread float *v) {
#pragma clang loop unroll(full)
  for (int s = 1; s < 8; s <<= 1) {
#pragma clang loop unroll(full)
    for (int i = 0; i < 8; i++) {
      if ((i & s) == 0) {
        const float a = v[i], b = v[i + s];
        v[i] = a + b;
        v[i + s] = a - b;
      }
    }
  }
}

// Completes the 256-point transform of a block spread 8 per lane over a SIMD-group.
static inline void fwht_lanes(thread float *v, ushort lane) {
  fwht8(v);
#pragma clang loop unroll(full)
  for (ushort s = 1; s < 32; s <<= 1) {
    const bool hi = (lane & s) != 0;
#pragma clang loop unroll(full)
    for (int i = 0; i < 8; i++) {
      const float o = simd_shuffle_xor(v[i], s);
      v[i] = hi ? o - v[i] : v[i] + o;
    }
  }
#pragma clang loop unroll(full)
  for (int i = 0; i < 8; i++) v[i] *= 0.0625f;
}

// Threadgroup-wide reductions over NSG SIMD-groups.
template <int NG>
static inline float tg_sum(float v, threadgroup float *tmp, ushort sg, ushort lane) {
  v = simd_sum(v);
  if (lane == 0) tmp[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = 0;
#pragma clang loop unroll(full)
  for (int i = 0; i < NG; i++) r += tmp[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

template <int NG>
static inline float tg_max(float v, threadgroup float *tmp, ushort sg, ushort lane) {
  v = simd_max(v);
  if (lane == 0) tmp[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = 0;
#pragma clang loop unroll(full)
  for (int i = 0; i < NG; i++) r = max(r, tmp[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

#ifndef NSG
#define NSG 8
#endif
#ifndef MAXB
#define MAXB 3  // rotation blocks per SIMD-group (D <= NSG * MAXB * 256)
#endif
#ifndef IN_T
#define IN_T bfloat
#endif
#ifndef OUT_F
#define OUT_F 0
#endif
#ifndef OUT_H
#define OUT_H 0
#endif
#ifndef OUT_HR
#define OUT_HR 0
#endif
#ifndef HDIM
#define HDIM 128
#endif

#ifndef POST
#define POST 0
#endif
#ifndef LN
#define LN 0  // LayerNorm without affine (FLUX.2) instead of RMSNorm: the row's mean is subtracted first
#endif
#if LN && POST
#error "LN excludes POST"
#endif

// a[m] = int8(rot(rmsnorm(x[m]) * scale'[type] + shift[type])), one threadgroup per row.
// mod: [types][2][D] floats per row type: scale' (the norm weight times 1 + scale) and shift.
// With OUT_F, writes the modulated row as floats instead (no rotation, no quantization).
// With POST, first x[m] += post * rmsnorm(y[m]) (written back): the previous product's normalized, gated output
// joins the residual stream in the same pass (post_norm_add below, fused).
kernel void norm_mod_q8(device float *x [[buffer(0)]],
                        device const float *mod [[buffer(1)]],
                        device const uchar *rtype [[buffer(2)]],
                        device void *out [[buffer(3)]],
                        device float *sa [[buffer(4)]],
                        constant NormQParams &p [[buffer(5)]],
                        device const float *y [[buffer(6)]],
                        device const float *post [[buffer(7)]],
                        uint m [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]])
{
  threadgroup float tmp[NSG];
  device float *row = x + size_t(m) * p.x_stride;
  const int nb = p.D / RB;

  float v[MAXB][8];
  float ss = 0;
#if LN
  float sm = 0;
#endif
#pragma clang loop unroll(full)
  for (int b = 0; b < MAXB; b++) {
    const int blk = sg + b * NSG;
    if (blk < nb) {
      const device float4 *src = (const device float4 *)(row + blk * RB + lane * 8);
      const float4 a = src[0], c = src[1];
      v[b][0] = a.x; v[b][1] = a.y; v[b][2] = a.z; v[b][3] = a.w;
      v[b][4] = c.x; v[b][5] = c.y; v[b][6] = c.z; v[b][7] = c.w;
#if LN
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i++) sm += v[b][i];
#elif !POST
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i++) ss += v[b][i] * v[b][i];
#endif
    }
  }
#if LN
  {  // two passes over the registers: the mean first, then the variance about it (no cancellation)
    const float mean = tg_sum<NSG>(sm, tmp, sg, lane) / float(p.D);
#pragma clang loop unroll(full)
    for (int b = 0; b < MAXB; b++) {
      if (sg + b * NSG < nb) {
#pragma clang loop unroll(full)
        for (int i = 0; i < 8; i++) {
          v[b][i] -= mean;
          ss += v[b][i] * v[b][i];
        }
      }
    }
  }
#endif
#if POST
  {
    float yv[MAXB][8];
    float ssy = 0;
#pragma clang loop unroll(full)
    for (int b = 0; b < MAXB; b++) {
      const int blk = sg + b * NSG;
      if (blk < nb) {
        const device float4 *src = (const device float4 *)(y + size_t(m) * p.D + blk * RB + lane * 8);
        const float4 a = src[0], c = src[1];
        yv[b][0] = a.x; yv[b][1] = a.y; yv[b][2] = a.z; yv[b][3] = a.w;
        yv[b][4] = c.x; yv[b][5] = c.y; yv[b][6] = c.z; yv[b][7] = c.w;
#pragma clang loop unroll(full)
        for (int i = 0; i < 8; i++) ssy += yv[b][i] * yv[b][i];
      }
    }
    const float iy = rsqrt(tg_sum<NSG>(ssy, tmp, sg, lane) / float(p.D) + p.eps);
#pragma clang loop unroll(full)
    for (int b = 0; b < MAXB; b++) {
      const int blk = sg + b * NSG;
      if (blk < nb) {
        const device float4 *w = (const device float4 *)(post + blk * RB + lane * 8);
        const float4 w0 = w[0], w1 = w[1];
        const float pw[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma clang loop unroll(full)
        for (int i = 0; i < 8; i++) {
          v[b][i] += pw[i] * yv[b][i] * iy;
          ss += v[b][i] * v[b][i];
        }
        device float4 *dst = (device float4 *)(row + blk * RB + lane * 8);
        dst[0] = float4(v[b][0], v[b][1], v[b][2], v[b][3]);
        dst[1] = float4(v[b][4], v[b][5], v[b][6], v[b][7]);
      }
    }
  }
#endif
  const float inv = rsqrt(tg_sum<NSG>(ss, tmp, sg, lane) / float(p.D) + p.eps);
  const device float *md = mod + size_t(rtype[m]) * p.mod_stride;

#if OUT_F || OUT_H
#if OUT_F
  device float *o = (device float *)out + size_t(m) * p.D;
#else
  device half *o = (device half *)out + size_t(m) * p.D;
#endif
#pragma clang loop unroll(full)
  for (int b = 0; b < MAXB; b++) {
    const int blk = sg + b * NSG;
    if (blk < nb) {
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i++) {
        const int c = blk * RB + lane * 8 + i;
        o[c] = v[b][i] * inv * md[c] + md[p.D + c];
      }
    }
  }
#else
  float amax = 0;
#pragma clang loop unroll(full)
  for (int b = 0; b < MAXB; b++) {
    const int blk = sg + b * NSG;
    if (blk < nb) {
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i++) {
        const int c = blk * RB + lane * 8 + i;
        v[b][i] = v[b][i] * inv * md[c] + md[p.D + c];
      }
      fwht_lanes(v[b], lane);
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i++) amax = max(amax, fabs(v[b][i]));
    }
  }
#if OUT_HR
  device half *oh = (device half *)out + size_t(m) * p.D;
#pragma clang loop unroll(full)
  for (int b = 0; b < MAXB; b++) {
    const int blk = sg + b * NSG;
    if (blk < nb) {
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i++) oh[blk * RB + lane * 8 + i] = half(v[b][i]);
    }
  }
#else
  amax = tg_max<NSG>(amax, tmp, sg, lane);
  const float scale = amax > 0 ? amax / 127.0f : 1.0f;
  const float rs = 1.0f / scale;
  device char *o = (device char *)out + size_t(m) * p.D;
#pragma clang loop unroll(full)
  for (int b = 0; b < MAXB; b++) {
    const int blk = sg + b * NSG;
    if (blk < nb) {
      char4 lo, hi;
      lo.x = char(rint(v[b][0] * rs)); lo.y = char(rint(v[b][1] * rs));
      lo.z = char(rint(v[b][2] * rs)); lo.w = char(rint(v[b][3] * rs));
      hi.x = char(rint(v[b][4] * rs)); hi.y = char(rint(v[b][5] * rs));
      hi.z = char(rint(v[b][6] * rs)); hi.w = char(rint(v[b][7] * rs));
      device char4 *dst = (device char4 *)(o + blk * RB + lane * 8);
      dst[0] = lo;
      dst[1] = hi;
    }
  }
  if (sg == 0 && lane == 0) sa[m] = scale;
#endif
#endif
}

static inline float silu(float x) { return x / (1.0f + exp(-x)); }

// a[m] = int8(rot(in[m])), or with MODE 1 int8(rot(value * silu(gate))) where
// in[m] holds the values in its first D columns and the gates in the next D.
kernel void rot_q8(device const IN_T *in [[buffer(0)]],
                   device char *out [[buffer(1)]],
                   device float *sa [[buffer(2)]],
                   constant RotQParams &p [[buffer(3)]],
                   uint m [[threadgroup_position_in_grid]],
                   ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]])
{
  threadgroup float tmp[NSG];
  const device IN_T *row = in + size_t(m) * p.x_stride;
  const int nb = p.D / RB;
  float v[MAXB][8];
  float amax = 0;
#pragma clang loop unroll(full)
  for (int b = 0; b < MAXB; b++) {
    const int blk = sg + b * NSG;
    if (blk < nb) {
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i++) {
        const int c = blk * RB + lane * 8 + i;
        float val = float(row[c]);
        if (p.mode == 1) val *= silu(float(row[p.D + c]));
        v[b][i] = val;
      }
      fwht_lanes(v[b], lane);
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i++) amax = max(amax, fabs(v[b][i]));
    }
  }
#if OUT_HR
  device half *oh = (device half *)out + size_t(m) * p.D;
#pragma clang loop unroll(full)
  for (int b = 0; b < MAXB; b++) {
    const int blk = sg + b * NSG;
    if (blk < nb) {
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i++) oh[blk * RB + lane * 8 + i] = half(v[b][i]);
    }
  }
  return;
#endif
  amax = tg_max<NSG>(amax, tmp, sg, lane);
  const float scale = amax > 0 ? amax / 127.0f : 1.0f;
  const float rs = 1.0f / scale;
  device char *o = out + size_t(m) * p.D;
#pragma clang loop unroll(full)
  for (int b = 0; b < MAXB; b++) {
    const int blk = sg + b * NSG;
    if (blk < nb) {
      char4 lo, hi;
      lo.x = char(rint(v[b][0] * rs)); lo.y = char(rint(v[b][1] * rs));
      lo.z = char(rint(v[b][2] * rs)); lo.w = char(rint(v[b][3] * rs));
      hi.x = char(rint(v[b][4] * rs)); hi.y = char(rint(v[b][5] * rs));
      hi.z = char(rint(v[b][6] * rs)); hi.w = char(rint(v[b][7] * rs));
      device char4 *dst = (device char4 *)(o + blk * RB + lane * 8);
      dst[0] = lo;
      dst[1] = hi;
    }
  }
  if (sg == 0 && lane == 0) sa[m] = scale;
}

// Per-head RMS norm of queries and keys, rotary embedding, and the
// head-major layout attention reads: one SIMD-group per (row, head slot).
// Slots 0..Hq-1 are query heads, then Hk key heads, then Hv value heads; the
// input row holds them in that order, HDIM channels each. The first ROT
// channels of a query or key head are rotated in pairs (i, i + ROT/2) by the
// angles cs[row][i] (cos) and sn[row][i] (sin), i < ROT/2. Lane l holds
// channels 4l .. 4l+3.
// Q, K, V outputs: [heads][out_rows][HDIM] half.
struct QkRopeParams {
  int M, Hq, Hk, Hv, in_stride, out_rows, rot, norm;
  float eps;
  int seq_len;  // rows per sequence: row m is row m % seq_len of sequence m / seq_len (heads of sequence j
                // are output heads j * H .. j * H + H - 1); the rotary tables are per row of a sequence
};

// With QK8, queries and keys are written as int8 with one scale per row (qs, ks) for the
// integer score product of kernels/attention.metal.
#ifndef QKROT
#define QKROT 0  // with QK8: rotate queries and keys before their int8 rounding (see below)
#endif
#ifndef QK8
#define QK8 0
#endif
#if QK8
#define QK_T char
#else
#define QK_T half
#endif

kernel void qk_norm_rope(device const IN_T *in [[buffer(0)]],
                         device const float *wq [[buffer(1)]],
                         device const float *wk [[buffer(2)]],
                         device const float *cs [[buffer(3)]],
                         device const float *sn [[buffer(4)]],
                         device QK_T *Q [[buffer(5)]],
                         device QK_T *K [[buffer(6)]],
                         device half *V [[buffer(7)]],
                         constant QkRopeParams &p [[buffer(8)]],
                         device float *qsc [[buffer(9)]],
                         device float *ksc [[buffer(10)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]])
{
  const int m = int(tg.x);
  const int sq = m / p.seq_len, r = m - sq * p.seq_len;
  const int slot = int(tg.y) * NSG + sg;
  const int nslots = p.Hq + p.Hk + p.Hv;
  if (slot >= nslots) return;
  constexpr int PER = HDIM / 32;
  const device IN_T *src = in + size_t(m) * p.in_stride + size_t(slot) * HDIM + lane * PER;
  float v[PER];
#pragma clang loop unroll(full)
  for (int i = 0; i < PER; i++) v[i] = float(src[i]);

  if (slot < p.Hq + p.Hk) {
    const bool isq = slot < p.Hq;
    const int h = isq ? slot : slot - p.Hq;
    if (p.norm) {
      float ss = 0;
#pragma clang loop unroll(full)
      for (int i = 0; i < PER; i++) ss += v[i] * v[i];
      const float inv = rsqrt(simd_sum(ss) / float(HDIM) + p.eps);
      const device float *w = (isq ? wq : wk) + lane * PER;
#pragma clang loop unroll(full)
      for (int i = 0; i < PER; i++) v[i] = v[i] * inv * w[i];
    }
    if (p.rot) {
      const int half_rot = p.rot / 2;
      const int lanes = half_rot / PER;  // lanes holding the first half of the rotated channels
      // partner of channel c (< rot) is c + half_rot (first half) or c - half_rot (second half)
      const ushort src_lane = lane < lanes ? lane + lanes : (lane < 2 * lanes ? lane - lanes : lane);
      float pv[PER];
#pragma clang loop unroll(full)
      for (int i = 0; i < PER; i++) pv[i] = simd_shuffle(v[i], src_lane);
      if (lane < 2 * lanes) {
        const bool first = lane < lanes;
        const int base = (first ? lane : lane - lanes) * PER;  // pair index of v[0]
        const device float *c = cs + size_t(r) * half_rot + base;
        const device float *s = sn + size_t(r) * half_rot + base;
#pragma clang loop unroll(full)
        for (int i = 0; i < PER; i++) v[i] = first ? v[i] * c[i] - pv[i] * s[i] : v[i] * c[i] + pv[i] * s[i];
      }
    }
#if QK8 && QKROT
    // Queries and keys both rotated by R = diag(signs) H / sqrt(HDIM), H the Walsh-Hadamard matrix: R is
    // orthonormal, so every product q.k, and with it attention, is unchanged. The rotation spreads the few channels
    // that dominate a row (the rotary ones, mostly) over all of them before the row gets its one int8 scale.
#pragma clang loop unroll(full)
    for (int i = 0; i < PER; i++) {
      const uint c = uint(lane) * PER + uint(i);
      if (popcount(c * 0x2545F491u) & 1u) v[i] = -v[i];  // a fixed pseudo-random sign per channel
    }
#pragma clang loop unroll(full)
    for (int s = 1; s < PER; s <<= 1) {  // butterflies within a lane's channels
#pragma clang loop unroll(full)
      for (int i = 0; i < PER; i++) {
        if (!(i & s)) { const float a = v[i], b = v[i + s]; v[i] = a + b; v[i + s] = a - b; }
      }
    }
#pragma clang loop unroll(full)
    for (ushort ls = 1; ls < 32; ls <<= 1) {  // and across lanes: channel c pairs with c ^ (ls * PER)
#pragma clang loop unroll(full)
      for (int i = 0; i < PER; i++) {
        const float o = simd_shuffle_xor(v[i], ls);
        v[i] = (lane & ls) ? o - v[i] : v[i] + o;
      }
    }
    const float rn = rsqrt(float(HDIM));
#pragma clang loop unroll(full)
    for (int i = 0; i < PER; i++) v[i] *= rn;
#endif
    const size_t hrow = isq ? (size_t(sq) * p.Hq + h) * p.out_rows + r : (size_t(sq) * p.Hk + h) * p.out_rows + r;
    device QK_T *dst = (isq ? Q : K) + hrow * HDIM;
#if QK8
    float amax = 0;
#pragma clang loop unroll(full)
    for (int i = 0; i < PER; i++) amax = max(amax, fabs(v[i]));
    amax = simd_max(amax);
    const float scale = amax > 0 ? amax / 127.0f : 1.0f;
#pragma clang loop unroll(full)
    for (int i = 0; i < PER; i++) dst[lane * PER + i] = char(rint(v[i] / scale));
    if (lane == 0) (isq ? qsc : ksc)[hrow] = scale;
#else
#pragma clang loop unroll(full)
    for (int i = 0; i < PER; i++) dst[lane * PER + i] = half(v[i]);
#endif
  } else {
    device half *dst = V + ((size_t(sq) * p.Hv + slot - p.Hq - p.Hk) * p.out_rows + r) * HDIM;
#pragma clang loop unroll(full)
    for (int i = 0; i < PER; i++) dst[lane * PER + i] = half(v[i]);
  }
}

// out[m][n] = sum_k in[m][k] * w[n][k] + b[n], all float, for the small
// projections around the transformer (patch in, per-step output heads).
// One SIMD-group per (row, 32 outputs): lanes split k.
struct LinF32Params { int M, N, K, in_stride, out_stride, has_bias; };

kernel void linear_f32(device const float *in [[buffer(0)]],
                       device const float *w [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       device float *out [[buffer(3)]],
                       constant LinF32Params &p [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]])
{
  const int m = int(tg.y) * NSG + sg;
  if (m >= p.M) return;
  const int n0 = int(tg.x) * 32;
  const device float *x = in + size_t(m) * p.in_stride;
  float acc = 0;
  // Each lane accumulates one output column over all k when K is small, or
  // the lanes split k for each column when K is large.
  if (p.K <= 256) {
    const int n = n0 + lane;
    if (n < p.N) {
      const device float *wr = w + size_t(n) * p.K;
      for (int k = 0; k < p.K; k++) acc += x[k] * wr[k];
      out[size_t(m) * p.out_stride + n] = acc + (p.has_bias ? b[n] : 0.0f);
    }
  } else {
    for (int j = 0; j < 32 && n0 + j < p.N; j++) {
      const device float *wr = w + size_t(n0 + j) * p.K;
      float s = 0;
      for (int k = lane; k < p.K; k += 32) s += x[k] * wr[k];
      s = simd_sum(s);
      if (lane == j) acc = s;
    }
    const int n = n0 + lane;
    if (n < p.N) out[size_t(m) * p.out_stride + n] = acc + (p.has_bias ? b[n] : 0.0f);
  }
}

// dst[i] = src[i] for n float4 elements.
kernel void copy4(device const float4 *src [[buffer(0)]],
                  device float4 *dst [[buffer(1)]],
                  constant int &n [[buffer(2)]],
                  uint i [[thread_position_in_grid]])
{
  if (int(i) < n) dst[i] = src[i];
}

// LayerNorm with affine weight and bias over D channels -> half; one threadgroup per row.
struct LnParams { int M, D, x_stride; float eps; };

#ifndef LN_OUT_T
#define LN_OUT_T half
#endif

kernel void layer_norm(device const float *x [[buffer(0)]],
                       device const float *w [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       device LN_OUT_T *out [[buffer(3)]],
                       constant LnParams &p [[buffer(4)]],
                       uint m [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]],
                       ushort tid [[thread_index_in_threadgroup]])
{
  threadgroup float tmp[NSG];
  const device float *row = x + size_t(m) * p.x_stride;
  float s = 0;
  for (int c = tid; c < p.D; c += NSG * 32) s += row[c];
  const float mean = tg_sum<NSG>(s, tmp, sg, lane) / float(p.D);
  float v = 0;
  for (int c = tid; c < p.D; c += NSG * 32) { const float d = row[c] - mean; v += d * d; }
  const float inv = rsqrt(tg_sum<NSG>(v, tmp, sg, lane) / float(p.D) + p.eps);
  device LN_OUT_T *o = out + size_t(m) * p.D;
  for (int c = tid; c < p.D; c += NSG * 32) o[c] = LN_OUT_T((row[c] - mean) * inv * w[c] + b[c]);
}

// out[m][j] = value * silu(gate), in[m] = [values (F) | gates (F)], half -> half.
struct SwigluParams { int M, F; };

kernel void swiglu(device const half *in [[buffer(0)]],
                   device half *out [[buffer(1)]],
                   constant SwigluParams &p [[buffer(2)]],
                   uint2 gid [[thread_position_in_grid]])
{
  const int j = int(gid.x), m = int(gid.y);
  if (j >= p.F || m >= p.M) return;
  const device half *row = in + size_t(m) * 2 * p.F;
  const float g = float(row[p.F + j]);
  out[size_t(m) * p.F + j] = half(float(row[j]) * g / (1.0f + exp(-g)));
}

// ---- Convolutions as matrix products (first layers with few channels) ------
// Activations are NHWC. im2col writes, for every output pixel of a 3x3 (or
// 1x1) convolution, the KH*KW*C input values it reads, in (ky, kx, c) order,
// padded with zeros to K columns. Out-of-image taps read the reflection of
// the image (no edge repeat), or zeros with ZPAD; pad_t/pad_l give the
// padding before the first row/column, the bottom/right side is whatever the
// stride leaves. Output rows start at row0 (a stripe of Ho rows). Wider
// convolutions run without im2col (kernels/conv.metal).
#ifndef ZPAD
#define ZPAD 0
#endif
struct Im2colParams { int N, H, W, C, Ho, Wo, KH, KW, stride, pad_t, pad_l, K, row0; };

static inline int reflect(int i, int n) {
  if (i < 0) i = -i;
  if (i >= n) i = 2 * n - 2 - i;
  return i;
}

kernel void im2col(device const half *x [[buffer(0)]],
                   device half *col [[buffer(1)]],
                   constant Im2colParams &p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]])
{
  // one thread per (4 channels, tap, output pixel), flattened in that order
  const int C4 = (p.C + 3) / 4, taps = p.KH * p.KW;
  const int c4 = (int(gid) % C4) * 4;
  const int tap = (int(gid) / C4) % taps;
  const int pix = int(gid) / (C4 * taps);
  if (pix >= p.N * p.Ho * p.Wo) return;
  const int n = pix / (p.Ho * p.Wo), r = pix % (p.Ho * p.Wo);
  const int oy = r / p.Wo + p.row0, ox = r % p.Wo;
  const int ky = tap / p.KW, kx = tap % p.KW;
  int iy = oy * p.stride + ky - p.pad_t, ix = ox * p.stride + kx - p.pad_l;
  device half *dst = col + size_t(pix) * p.K + tap * p.C + c4;
#if ZPAD
  if (iy < 0 || iy >= p.H || ix < 0 || ix >= p.W) {
    for (int i = 0; i < 4 && c4 + i < p.C; i++) dst[i] = 0;
    return;
  }
#else
  iy = reflect(iy, p.H);
  ix = reflect(ix, p.W);
#endif
  const device half *src = x + ((size_t(n) * p.H + iy) * p.W + ix) * p.C + c4;
  if ((p.C & 3) == 0) {
    *(device half4 *)dst = *(const device half4 *)src;
  } else {
    for (int i = 0; i < 4 && c4 + i < p.C; i++) dst[i] = src[i];
  }
}

// GroupNorm statistics per image and group, in two passes: pass 0 sums the values, pass 1 the squared
// deviations from the mean pass 0 found; stats[n][g] = (sum, sum of squared deviations). No atomics, so equal
// inputs give equal statistics bit for bit (float addition is not associative: atomics that land in a varying
// order made the same image normalize slightly differently from run to run). Each pass is gn_stats, one
// threadgroup of 256 threads per run of GN_CHUNK pixels of one image writing that run's per-group sums to
// part[n][chunk][g], then gn_reduce, one thread per image and group adding the runs in order.
struct GnParams { int N, HW, C, G; float eps; int silu; int pass; int W; int chunks; };
#ifndef GN_IN_T
#define GN_IN_T float  // the input's precision (statistics and normalization always in float)
#endif
#ifndef PADW
#define PADW 0  // gn_apply: write the inside of images with a one-pixel border (the input of a conv3x3)
#endif
#define GN_CHUNK 2048

kernel void gn_stats(device const GN_IN_T *x [[buffer(0)]],
                     device float *part [[buffer(1)]],
                     device const float *stats [[buffer(2)]],
                     constant GnParams &p [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]])
{
  // Thread t reads channels 4 (t % C4) .. +3 of every (256 / C4)-th pixel of the run, so a SIMD-group reads
  // whole rows (coalesced); the four channels of a float4 always share a group (groups hold multiples of four).
  threadgroup float tmp[256];
  const int n = int(tg.y), chunk = int(tg.x), px0 = chunk * GN_CHUNK;
  const int C4 = p.C / 4, cg = p.C / p.G;
  const int q = tid % C4, g = q * 4 / cg;
  const float mean = p.pass == 1 ? stats[(n * p.G + g) * 2] / float(p.HW * cg) : 0.0f;
  const device vec<GN_IN_T, 4> *x4 = (const device vec<GN_IN_T, 4> *)x + size_t(n) * p.HW * C4;
  float s = 0;
  const int end = min(px0 + GN_CHUNK, p.HW);
  for (int px = px0 + tid / C4; px < end; px += 256 / C4) {
    const float4 v = float4(x4[size_t(px) * C4 + q]) - mean;
    s += p.pass == 1 ? dot(v, v) : (v.x + v.y) + (v.z + v.w);
  }
  tmp[tid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < p.G) {  // group tid: channel quads tid * cg/4 .. +cg/4 of every pixel lane, in a fixed order
    const int q0 = int(tid) * cg / 4, nq = cg / 4, lanes = 256 / C4;
    float t = 0;
    for (int r = 0; r < lanes; r++)
      for (int k = 0; k < nq; k++) t += tmp[r * C4 + q0 + k];
    part[(size_t(n) * p.chunks + chunk) * p.G + tid] = t;
  }
}

kernel void gn_reduce(device const float *part [[buffer(0)]],
                      device float *stats [[buffer(1)]],
                      constant GnParams &p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]])
{
  const int n = int(gid) / p.G, g = int(gid) % p.G;
  if (n >= p.N) return;
  float t = 0;
  for (int c = 0; c < p.chunks; c++) t += part[(size_t(n) * p.chunks + c) * p.G + g];
  stats[(n * p.G + g) * 2 + p.pass] = t;
}

// y = (x - mean) / std * w + b, optionally SiLU, float or half in -> half out.
kernel void gn_apply(device const GN_IN_T *x [[buffer(0)]],
                     device const float *sums [[buffer(1)]],
                     device const float *w [[buffer(2)]],
                     device const float *b [[buffer(3)]],
                     device half *y [[buffer(4)]],
                     constant GnParams &p [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]])
{
  const int c = int(gid.x), pix = int(gid.y);
  if (c >= p.C || pix >= p.N * p.HW) return;
  const int n = pix / p.HW, g = c / (p.C / p.G);
  const float cnt = float(p.HW * (p.C / p.G));
  const float mean = sums[(n * p.G + g) * 2] / cnt;
  const float var = sums[(n * p.G + g) * 2 + 1] / cnt;
  float v = (float(x[size_t(pix) * p.C + c]) - mean) * rsqrt(var + p.eps) * w[c] + b[c];
  if (p.silu) v = v / (1.0f + exp(-v));
#if PADW
  const int H = p.HW / p.W, r = pix % p.HW, yy = r / p.W, xx = r % p.W;
  y[((size_t(n) * (H + 2) + yy + 1) * (p.W + 2) + xx + 1) * p.C + c] = half(v);
#else
  y[size_t(pix) * p.C + c] = half(v);
#endif
}

// y = half(x), n elements.
kernel void to_half(device const float *x [[buffer(0)]],
                    device half *y [[buffer(1)]],
                    constant int &n [[buffer(2)]],
                    uint i [[thread_position_in_grid]])
{
  if (int(i) < n) y[i] = half(x[i]);
}

// ---- Vision tower (edits) ----------------------------------------------------------------------------------------
// qkv [N][3][H][HD] float -> Q, K, V [H][rows][HDP] half, rotary embedding (rotate-half pairs (c, c + HD/2), angle
// tables cos/sin [N][HD]) on queries and keys, head dim zero-padded from HD to HDP for the attention kernel.
struct VitRopeParams { int N, H, HD, HDP, rows; };

kernel void vit_qkv_rope(device const float *qkv [[buffer(0)]],
                         device const float *cs [[buffer(1)]],
                         device const float *sn [[buffer(2)]],
                         device half *Q [[buffer(3)]],
                         device half *K [[buffer(4)]],
                         device half *V [[buffer(5)]],
                         constant VitRopeParams &p [[buffer(6)]],
                         uint3 gid [[thread_position_in_grid]])
{
  const int c = int(gid.x), slot = int(gid.y), n = int(gid.z);  // channel, (component, head), row
  if (c >= p.HDP || slot >= 3 * p.H || n >= p.N) return;
  const int comp = slot / p.H, h = slot % p.H;
  device half *dst = (comp == 0 ? Q : comp == 1 ? K : V) + (size_t(h) * p.rows + n) * p.HDP + c;
  if (c >= p.HD) { *dst = half(0.0f); return; }
  const device float *src = qkv + size_t(n) * 3 * p.H * p.HD + size_t(slot) * p.HD;
  float v = src[c];
  if (comp < 2) {
    const int half_d = p.HD / 2;
    const float partner = c < half_d ? -src[c + half_d] : src[c - half_d];
    v = v * cs[size_t(n) * p.HD + c] + partner * sn[size_t(n) * p.HD + c];
  }
  *dst = half(v);
}

// y = half(gelu(x)), x float [M][N] row-major; mode 0: tanh approximation, 1: exact (erf).
struct GeluParams { int n; int exact; };

static inline float erf_approx(float x) {
  // Abramowitz and Stegun 7.1.26 (|error| < 1.5e-7).
  const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
  const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t +
                          0.254829592f) * t * exp(-x * x);
  return copysign(y, x);
}

kernel void gelu(device const float *x [[buffer(0)]],
                 device half *y [[buffer(1)]],
                 constant GeluParams &p [[buffer(2)]],
                 uint i [[thread_position_in_grid]])
{
  if (int(i) >= p.n) return;
  const float v = x[i];
  const float g = p.exact ? 0.5f * v * (1.0f + erf_approx(v * 0.70710678f))
                          : 0.5f * v * (1.0f + precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
  y[i] = half(g);
}

// x[m] += post * rmsnorm(y[m]): a product's output normalized (RMS over the row), scaled per channel and
// added to the residual stream (Z-Image normalizes every block's attention and MLP output this way, with
// post = the norm weight times the step's tanh gate). One threadgroup per row; post: [D] floats.
kernel void post_norm_add(device float *x [[buffer(0)]],
                          device const float *y [[buffer(1)]],
                          device const float *post [[buffer(2)]],
                          constant NormQParams &p [[buffer(3)]],
                          uint m [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]])
{
  threadgroup float tmp[NSG];
  const device float *row = y + size_t(m) * p.D;
  device float *xr = x + size_t(m) * p.x_stride;
  const int nb = p.D / RB;
  float4 v[MAXB][2];
  float ss = 0;
#pragma clang loop unroll(full)
  for (int b = 0; b < MAXB; b++) {
    const int blk = sg + b * NSG;
    if (blk < nb) {
      const device float4 *src = (const device float4 *)(row + blk * RB + lane * 8);
      v[b][0] = src[0];
      v[b][1] = src[1];
      ss += dot(v[b][0], v[b][0]) + dot(v[b][1], v[b][1]);
    }
  }
  const float inv = rsqrt(tg_sum<NSG>(ss, tmp, sg, lane) / float(p.D) + p.eps);
#pragma clang loop unroll(full)
  for (int b = 0; b < MAXB; b++) {
    const int blk = sg + b * NSG;
    if (blk < nb) {
      const int c = blk * RB + lane * 8;
      device float4 *dst = (device float4 *)(xr + c);
      const device float4 *w = (const device float4 *)(post + c);
      dst[0] += w[0] * v[b][0] * inv;
      dst[1] += w[1] * v[b][1] * inv;
    }
  }
}

// p[m] = half(softmax(scale * s[m][0 .. valid))), zeros from valid to L. One threadgroup per row (the FLUX
// decoder's single-head attention, its scores materialized a chunk of query rows at a time).
struct SoftmaxParams { int M, L, valid; float scale; };

kernel void softmax_rows(device const float *s [[buffer(0)]],
                         device half *out [[buffer(1)]],
                         constant SoftmaxParams &p [[buffer(2)]],
                         uint m [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]])
{
  threadgroup float tmp[NSG];
  const device float *row = s + size_t(m) * p.L;
  device half *o = out + size_t(m) * p.L;
  float mx = -INFINITY;
  for (int c = tid; c < p.valid; c += NSG * 32) mx = max(mx, row[c] * p.scale);
  mx = simd_max(mx);
  if (lane == 0) tmp[sg] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mx = tmp[0];
  for (int i = 1; i < NSG; i++) mx = max(mx, tmp[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0;
  for (int c = tid; c < p.valid; c += NSG * 32) sum += exp(row[c] * p.scale - mx);
  const float inv = 1.0f / tg_sum<NSG>(sum, tmp, sg, lane);
  for (int c = tid; c < p.L; c += NSG * 32) o[c] = c < p.valid ? half(exp(row[c] * p.scale - mx) * inv) : half(0);
}
