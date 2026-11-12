// Kernel parameter blocks, shared by the Metal kernels and the Python host
// (h3metal/ops.py packs them with struct). Plain 32-bit ints and floats only,
// so the layout is the same on both sides.
#ifndef H3_PARAMS_H
#define H3_PARAMS_H

struct GemmParams {
  int M;         // rows of A and C (A is padded to a multiple of the tile height)
  int N;         // output features (a multiple of the tile width)
  int K;         // input features (a multiple of the group)
  int a_stride;  // elements between rows of A
  int c_stride;  // elements between rows of C (or of the residual X)
  int w_stride;  // elements (int8) or nibbles (int4) between rows of W
  int groups;    // K / G for group-wise scales
  int g_stride;  // elements between rows of the gate table
};

struct NormQParams {
  int M, D;          // rows, features (D a multiple of 256)
  int x_stride;      // elements between rows of X
  int mod_stride;    // elements between rows of the modulation table (a row: [2][D] = scale', shift')
  float eps;
};

struct RotQParams {
  int M, D;          // rows, features
  int x_stride;      // elements between rows of the input
  int mode;          // 0: input as is; 1: input holds [value | gate] pairs, output value * silu(gate)
};

struct QkParams {
  int M;             // rows
  int H;             // heads
  int D;             // head dim (128)
  int rot;           // rotated channels at the start of every head (96)
  int qkv_stride;    // elements between rows of the fused q|k|v input
  int out_rows;      // rows reserved per head in the outputs (M padded)
  float eps;
};

struct AttnParams {
  int Lq;            // query rows
  int L;             // keys
  int H;             // query heads
  int KVH;           // key/value heads (query head h reads key/value head h / (H / KVH))
  int q_rows;        // rows reserved per head in Q
  int k_rows;        // rows reserved per key/value head in K and V (a multiple of BK)
  int o_stride;      // elements between rows of the output [Lq][H * D]
  float scale;       // softmax scale times log2(e)
  int causal_off;    // with CAUSAL: query row i sees keys 0 .. i + causal_off
  int seq_heads;     // heads per sequence: head h belongs to sequence h / seq_heads (batched short sequences)
  int seq_rows;      // output rows between consecutive sequences
};

#endif
