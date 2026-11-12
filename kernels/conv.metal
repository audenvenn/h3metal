// 3x3 convolutions (stride 1, zero padding 1) on the GPU neural accelerators, without im2col.
//
//   out[y][x][n] = bias[n] + sum over taps (ky, kx) and channels c of xp[y + ky][x + kx][c] * w[n][(ky*3 + kx)*C + c]
//
// xp is the half-precision NHWC input with a one-pixel zero border, [(H + 2)][(W + 2)][C], followed by at least
// TM pixels of slack. For a fixed tap, the inputs of TM consecutive output pixels of a row are TM consecutive
// pixels of xp: an ordinary [TM][C] matrix with row stride C. So the convolution is nine accumulating matrix
// products whose left operands are views into xp; nothing is copied. A threadgroup computes TM pixels of one
// output row (the last tile of a row runs past it: those results are dropped) by TN output channels.
//
// Epilogues (EPI): 0 out[m][n] = OUT_T(v + bias[n]); 1 out[m][n] += v + bias[n] (out float, a residual).
// Specialization: TM, TN, NSG, EPI, OUT_T.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

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
#define OUT_T float
#endif

struct ConvParams {
  int H, W, C;  // output rows and columns (= input), input channels (a multiple of 32)
  int N;        // output channels (a multiple of TN)
  int K;        // elements per weight row (9 C, padded)
};

kernel void conv3x3(device half *xp [[buffer(0)]],
                    device half *w [[buffer(1)]],
                    device void *out [[buffer(2)]],
                    device const float *bias [[buffer(3)]],
                    constant ConvParams &p [[buffer(4)]],
                    uint3 tgid [[threadgroup_position_in_grid]])
{
  const int n0 = int(tgid.x) * TN;
  const int tiles_x = (p.W + TM - 1) / TM;
  const int y = int(tgid.y) / tiles_x, x0 = (int(tgid.y) % tiles_x) * TM;

  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, true,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  using TA = tensor<device half, dextents<int, 2>, tensor_inline>;
  const array<int, 2> a_str = {1, p.C};
  const array<int, 2> w_str = {1, p.K};
  TA A0(xp + (size_t(y) * (p.W + 2) + x0) * p.C, dextents<int, 2>(p.C, TM), a_str);
  TA W0(w + size_t(n0) * p.K, dextents<int, 2>(p.C, TN), w_str);
  auto acc = op.get_destination_cooperative_tensor<decltype(A0), decltype(W0), float>();
#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  }
  for (int t = 0; t < 9; t++) {
    const int ky = t / 3, kx = t % 3;
    TA A(xp + (size_t(y + ky) * (p.W + 2) + x0 + kx) * p.C, dextents<int, 2>(p.C, TM), a_str);
    TA Wt(w + size_t(n0) * p.K + t * p.C, dextents<int, 2>(p.C, TN), w_str);
    op.run(A, Wt, acc);
  }
#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto idx = acc.get_multidimensional_index(i);  // {column, row} within the tile
    const int n = n0 + int(idx[0]);
    const int x = x0 + int(idx[1]);
    if (x >= p.W) continue;
    const float v = acc[i] + bias[n];
    const size_t at = (size_t(y) * p.W + x) * p.N + n;
#if EPI == 0
    ((device OUT_T *)out)[at] = OUT_T(v);
#else
    ((device float *)out)[at] += v;
#endif
  }
}

// xp[(H + 2)][(W + 2)][C] <- the half-precision copy of x[H][W][C] (float), upsampled UP times by nearest
// neighbour, with a zero border: the input of a conv3x3. One thread per (4 channels, padded pixel).
struct PadParams { int H, W, C, UP; };  // output H, W (upsampled), channels, upsampling factor

kernel void pad_half(device const float *x [[buffer(0)]],
                     device half *xp [[buffer(1)]],
                     constant PadParams &p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
  const int C4 = p.C / 4, Wp = p.W + 2;
  const int c4 = int(gid) % C4, pix = int(gid) / C4;
  if (pix >= (p.H + 2) * Wp) return;
  const int yy = pix / Wp - 1, xx = pix % Wp - 1;
  half4 v = half4(0.0h);
  if (yy >= 0 && yy < p.H && xx >= 0 && xx < p.W) {
    const int sy = yy / p.UP, sx = xx / p.UP, sw = p.W / p.UP;
    v = half4(((const device float4 *)x)[(size_t(sy) * sw + sx) * C4 + c4]);
  }
  ((device half4 *)xp)[size_t(pix) * C4 + c4] = v;
}

// Zeroes the one-pixel border of a padded image xp[(H + 2)][(W + 2)][C] (its inside is written elsewhere).
kernel void zero_border(device half *xp [[buffer(0)]],
                        constant PadParams &p [[buffer(1)]],
                        uint gid [[thread_position_in_grid]])
{
  const int C4 = p.C / 4, Wp = p.W + 2, Hp = p.H + 2;
  const int c4 = int(gid) % C4, b = int(gid) / C4;
  if (b >= 2 * Wp + 2 * p.H) return;
  int yy, xx;
  if (b < Wp) {
    yy = 0; xx = b;
  } else if (b < 2 * Wp) {
    yy = Hp - 1; xx = b - Wp;
  } else {
    const int r = b - 2 * Wp;
    yy = 1 + r / 2; xx = (r & 1) ? Wp - 1 : 0;
  }
  ((device half4 *)xp)[(size_t(yy) * Wp + xx) * C4 + c4] = half4(0.0h);
}
