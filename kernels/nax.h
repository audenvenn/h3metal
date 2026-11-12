// Register fragments for the GPU neural accelerators.
//
// A 16x16 fragment is spread over the 32 lanes of a SIMD-group, eight
// elements per lane: rows fm and fm + 8, columns fn .. fn + 3. This is the
// operand layout the accelerator's 16x32x16 SIMD-group product uses on
// M5/M6-class GPUs (the same layout MLX's steel NAX kernels rely on), so
// fragments can be loaded from memory straight into the registers a product
// consumes, and a product's result can feed the next one without a round trip
// through threadgroup memory.
#pragma once

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace nax {

template <typename T>
using frag = vec<T, 8>;

// (column, row) of the first element this lane holds.
inline short2 coord() {
  const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  return short2{fn, fm};
}

// Loads a 16x16 fragment whose element (r, c) is at src[r * ld + c].
template <typename T, typename U>
inline void load(thread frag<T> &f, const device U *src, int ld, short2 sc) {
  src += sc.y * ld + sc.x;
#pragma clang loop unroll(full)
  for (short i = 0; i < 2; i++) {
#pragma clang loop unroll(full)
    for (short j = 0; j < 4; j++) f[i * 4 + j] = static_cast<T>(src[i * 8 * ld + j]);
  }
}

// Same, rows at or beyond `rows` read as zero.
template <typename T, typename U>
inline void load_rows(thread frag<T> &f, const device U *src, int ld, short2 sc, int rows) {
  src += sc.y * ld + sc.x;
#pragma clang loop unroll(full)
  for (short i = 0; i < 2; i++) {
    const bool ok = sc.y + i * 8 < rows;
#pragma clang loop unroll(full)
    for (short j = 0; j < 4; j++) f[i * 4 + j] = ok ? static_cast<T>(src[i * 8 * ld + j]) : T(0);
  }
}

// Same, from threadgroup memory; a lane's four adjacent elements of a row come in one vector load.
template <typename T, typename U>
inline void load_tg(thread frag<T> &f, const threadgroup U *src, int ld, short2 sc) {
  src += sc.y * ld + sc.x;
#pragma clang loop unroll(full)
  for (short i = 0; i < 2; i++) {
    const vec<U, 4> v = *(const threadgroup vec<U, 4> *)(src + i * 8 * ld);
#pragma clang loop unroll(full)
    for (short j = 0; j < 4; j++) f[i * 4 + j] = static_cast<T>(v[j]);
  }
}

template <typename T, typename U>
inline void store(const thread frag<T> &f, device U *dst, int ld, short2 sc) {
  dst += sc.y * ld + sc.x;
#pragma clang loop unroll(full)
  for (short i = 0; i < 2; i++) {
#pragma clang loop unroll(full)
    for (short j = 0; j < 4; j++) dst[i * 8 * ld + j] = static_cast<U>(f[i * 4 + j]);
  }
}

template <typename T, typename U>
inline void store_rows(const thread frag<T> &f, device U *dst, int ld, short2 sc, int rows) {
  dst += sc.y * ld + sc.x;
#pragma clang loop unroll(full)
  for (short i = 0; i < 2; i++) {
    if (sc.y + i * 8 < rows) {
#pragma clang loop unroll(full)
      for (short j = 0; j < 4; j++) dst[i * 8 * ld + j] = static_cast<U>(f[i * 4 + j]);
    }
  }
}

// [c0 c1] += a x [b0 b1]: a 16x16, b0/b1 16x16 (or their transposes), c 16x32.
template <typename CT, typename AT, typename BT, bool TA, bool TB>
inline void mma(thread frag<CT> &c0, thread frag<CT> &c1, const thread frag<AT> &a, const thread frag<BT> &b0,
                const thread frag<BT> &b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, TA, TB, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
  auto cb = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
  auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>,
                                                          metal::remove_addrspace_t<decltype(cb)>, CT>();
#pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    ca[i] = a[i];
    cb[i] = b0[i];
    cb[8 + i] = b1[i];
    cc[i] = c0[i];
    cc[8 + i] = c1[i];
  }
  op.run(ca, cb, cc);
#pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    c0[i] = cc[i];
    c1[i] = cc[8 + i];
  }
}

// Reductions over the 16 columns of each of a lane's two rows: the four lanes
// that share a row differ in lane bits 0 and 3.
inline void row_max(const thread frag<float> &f, thread float &r0, thread float &r1) {
  float a = max(max(f[0], f[1]), max(f[2], f[3]));
  float b = max(max(f[4], f[5]), max(f[6], f[7]));
  a = max(a, simd_shuffle_xor(a, ushort(1)));
  a = max(a, simd_shuffle_xor(a, ushort(8)));
  b = max(b, simd_shuffle_xor(b, ushort(1)));
  b = max(b, simd_shuffle_xor(b, ushort(8)));
  r0 = max(r0, a);
  r1 = max(r1, b);
}

inline void row_sum(const thread frag<float> &f, thread float &r0, thread float &r1) {
  float a = (f[0] + f[1]) + (f[2] + f[3]);
  float b = (f[4] + f[5]) + (f[6] + f[7]);
  a += simd_shuffle_xor(a, ushort(1));
  a += simd_shuffle_xor(a, ushort(8));
  b += simd_shuffle_xor(b, ushort(1));
  b += simd_shuffle_xor(b, ushort(8));
  r0 += a;
  r1 += b;
}

}  // namespace nax
