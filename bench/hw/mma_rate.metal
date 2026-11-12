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

kernel void mma_rate(device ACC_TYPE *out [[buffer(0)]],
                     uint gid [[thread_position_in_grid]])
{
  nax::frag<IN_TYPE> a, b0, b1;
  for (short i = 0; i < 8; i++) {
    a[i] = IN_TYPE(float((gid + i) & 3));
    b0[i] = IN_TYPE(float((gid * 3 + i) & 1));
    b1[i] = IN_TYPE(float((gid * 5 + i) & 1));
  }
  nax::frag<ACC_TYPE> c0 = ACC_TYPE(0), c1 = ACC_TYPE(0), c2 = ACC_TYPE(0), c3 = ACC_TYPE(0);
  for (int it = 0; it < ITERS; it++) {
    nax::mma<ACC_TYPE, IN_TYPE, IN_TYPE, false, false>(c0, c1, a, b0, b1);
    nax::mma<ACC_TYPE, IN_TYPE, IN_TYPE, false, false>(c2, c3, a, b1, b0);
  }
  float s = 0;
  for (short i = 0; i < 8; i++) s += float(c0[i]) + float(c1[i]) + float(c2[i]) + float(c3[i]);
  out[gid] = ACC_TYPE(s);
}
