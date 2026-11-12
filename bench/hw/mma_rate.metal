// Peak rate of the 16x32x16 SIMD-group product on register fragments (kernels/nax.h), per operand types:
// every SIMD-group runs ITERS chained products on its own fragments; nothing touches memory in the loop.
#include <metal_stdlib>
#include "../../kernels/nax.h"
using namespace metal;

#ifndef ACC_TYPE
#define ACC_TYPE float
#endif
#ifndef IN_TYPE
#define IN_TYPE half
#endif
#ifndef ITERS
#define ITERS 4096
#endif
#ifndef CHAINS
#define CHAINS 8
#endif

kernel void mma_rate(device ACC_TYPE *out [[buffer(0)]],
                     uint gid [[thread_position_in_grid]])
{
  nax::frag<IN_TYPE> a, b0, b1;
  for (short i = 0; i < 8; i++) {
    a[i] = IN_TYPE(float((gid + i) & 3));
    b0[i] = IN_TYPE(float((gid * 3 + i) & 1));
    b1[i] = IN_TYPE(float((gid * 5 + i) & 1));
  }
  // CHAINS independent accumulator pairs, so the products' latency overlaps and only their rate counts.
  nax::frag<ACC_TYPE> c[2 * CHAINS];
#pragma clang loop unroll(full)
  for (short k = 0; k < 2 * CHAINS; k++) c[k] = ACC_TYPE(0);
  for (int it = 0; it < ITERS; it++) {
#pragma clang loop unroll(full)
    for (short k = 0; k < CHAINS; k++) nax::mma<ACC_TYPE, IN_TYPE, IN_TYPE, false, false>(c[2 * k], c[2 * k + 1], a, b0, b1);
  }
  float s = 0;
#pragma clang loop unroll(full)
  for (short k = 0; k < 2 * CHAINS; k++) {
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) s += float(c[k][i]);
  }
  out[gid] = ACC_TYPE(s);
}
