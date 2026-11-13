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
// Epilogues (EPI): 0 out[m][n] = OUT_T(v + bias[n]); 1 out[m][n] += v + bias[n] (out a residual, OUT_T
// float or half: the sum is taken in float either way). Only the first NO output channels are stored, NO
// apart (a convolution to 3 channels stores 3, though its weights are padded to a whole tile).
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
  int H, W, C;  // output rows and columns, input channels (a multiple of 32)
  int N;        // output channels (a multiple of TN)
  int K;        // elements per weight row (9 C, padded)
  int NB;       // images in the batch
  int S;        // stride
  int Hp, Wp;   // the padded input's sides
  int off;      // padded coordinates of tap (0, 0) of output (0, 0): 0 for padding 1, 1 for padding 0
  int NO;       // output channels stored per pixel (<= N)
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
  const int row = int(tgid.y) / tiles_x, x0 = (int(tgid.y) % tiles_x) * TM;
  const int img = row / p.H, y = row % p.H;
  // tap (ky, kx) of output pixel (y, x0 + i) reads padded pixel (S y + ky + off, S (x0 + i) + kx + off)
  device half *base = xp + (size_t(img) * p.Hp * p.Wp + size_t(p.S * y + p.off) * p.Wp + p.S * x0 + p.off) * p.C;

  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, true,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  using TA = tensor<device half, dextents<int, 2>, tensor_inline>;
  const array<int, 2> a_str = {1, p.S * p.C};
  const array<int, 2> w_str = {1, p.K};
  TA A0(base, dextents<int, 2>(p.C, TM), a_str);
  TA W0(w + size_t(n0) * p.K, dextents<int, 2>(p.C, TN), w_str);
  auto acc = op.get_destination_cooperative_tensor<decltype(A0), decltype(W0), float>();
#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  }
  for (int t = 0; t < 9; t++) {
    const int ky = t / 3, kx = t % 3;
    TA A(base + (size_t(ky) * p.Wp + kx) * p.C, dextents<int, 2>(p.C, TM), a_str);
    TA Wt(w + size_t(n0) * p.K + t * p.C, dextents<int, 2>(p.C, TN), w_str);
    op.run(A, Wt, acc);
  }
#pragma clang loop unroll(full)
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto idx = acc.get_multidimensional_index(i);  // {column, row} within the tile
    const int n = n0 + int(idx[0]);
    const int x = x0 + int(idx[1]);
    if (x >= p.W || n >= p.NO) continue;
    const float v = acc[i] + bias[n];
    device OUT_T *o = (device OUT_T *)out + (size_t(row) * p.W + x) * p.NO + n;
#if EPI == 0
    *o = OUT_T(v);
#else
    *o = OUT_T(float(*o) + v);
#endif
  }
}

// xp[NB][(H + 2)][(W + 2)][C] <- the half-precision copy of x[NB][H][W][C] (float or half), upsampled UP times by
// nearest neighbour, with a one-pixel border of zeros (or, with REFLECT, the reflection of the image): the
// input of a conv3x3. One thread per (4 channels, padded pixel).
struct PadParams { int H, W, C, UP, NB, REFLECT; };  // output H, W (upsampled), channels, factor, images, mode
#ifndef PAD_IN_T
#define PAD_IN_T float  // pad_half's input precision
#endif

static inline int reflect1(int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); }

kernel void pad_half(device const PAD_IN_T *x [[buffer(0)]],
                     device half *xp [[buffer(1)]],
                     constant PadParams &p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
  const int C4 = p.C / 4, Wp = p.W + 2, Hp = p.H + 2;
  const int c4 = int(gid) % C4, pix = int(gid) / C4;
  if (pix >= p.NB * Hp * Wp) return;
  const int img = pix / (Hp * Wp), r = pix % (Hp * Wp);
  int yy = r / Wp - 1, xx = r % Wp - 1;
  half4 v = half4(0.0h);
  const bool inside = yy >= 0 && yy < p.H && xx >= 0 && xx < p.W;
  if (inside || p.REFLECT) {
    yy = reflect1(yy, p.H);
    xx = reflect1(xx, p.W);
    const int sy = yy / p.UP, sx = xx / p.UP, sw = p.W / p.UP, sh = p.H / p.UP;
    v = half4(((const device vec<PAD_IN_T, 4> *)x)[((size_t(img) * sh + sy) * sw + sx) * C4 + c4]);
  }
  ((device half4 *)xp)[size_t(pix) * C4 + c4] = v;
}

// Fills the one-pixel border of padded images xp[NB][(H + 2)][(W + 2)][C] (their inside written elsewhere) with
// zeros, or with REFLECT the reflection of the inside (rows and columns reflected alike, corners included).
kernel void fill_border(device half *xp [[buffer(0)]],
                        constant PadParams &p [[buffer(1)]],
                        uint gid [[thread_position_in_grid]])
{
  const int C4 = p.C / 4, Wp = p.W + 2, Hp = p.H + 2;
  const int per = 2 * Wp + 2 * p.H;  // border pixels per image
  const int c4 = int(gid) % C4, b = (int(gid) / C4) % per, img = int(gid) / C4 / per;
  if (img >= p.NB) return;
  int yy, xx;
  if (b < Wp) {
    yy = 0; xx = b;
  } else if (b < 2 * Wp) {
    yy = Hp - 1; xx = b - Wp;
  } else {
    const int r = b - 2 * Wp;
    yy = 1 + r / 2; xx = (r & 1) ? Wp - 1 : 0;
  }
  device half4 *im = (device half4 *)xp + size_t(img) * Hp * Wp * C4;
  half4 v = half4(0.0h);
  if (p.REFLECT) {  // padded coordinate i <-> image i - 1; reflect within the image, then back
    const int sy = reflect1(yy - 1, p.H) + 1, sx = reflect1(xx - 1, p.W) + 1;
    v = im[(size_t(sy) * Wp + sx) * C4 + c4];
  }
  im[(size_t(yy) * Wp + xx) * C4 + c4] = v;
}
