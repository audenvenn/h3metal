#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
// Writes, for every thread and element slot of a TMxTN int32 destination, the (column,row) it holds.
kernel void layout(device int8_t *a [[buffer(0)]], device uchar *w [[buffer(1)]], device int *out [[buffer(2)]],
                   uint tid [[thread_index_in_threadgroup]]) {
  array<int, 2> a_str = {1, 128};
  array<int, 2> w_str = {1, 128};
  tensor<device int8_t, dextents<int, 2>, tensor_inline> A(a, dextents<int, 2>(128, TM), a_str);
  tensor<device int4b_format, dextents<int, 2>, tensor_inline> W(w, dextents<int, 2>(128, TN), w_str);
  constexpr auto desc = matmul2d_descriptor(TM, TN, 128, false, true, true, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto tA = A.slice<128, TM>(0, 0);
  auto tW = W.slice<128, TN>(0, 0);
  auto acc = op.get_destination_cooperative_tensor<decltype(tA), decltype(tW), int32_t>();
  const int cap = acc.get_capacity();
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    int c = -1, r = -1;
    if (acc.is_valid_element(i)) { auto idx = acc.get_multidimensional_index(i); c = idx[0]; r = idx[1]; }
    out[(tid * 256 + i) * 2] = c;
    out[(tid * 256 + i) * 2 + 1] = r;
  }
  if (tid == 0) out[(NSG * 32 * 256) * 2] = cap;
}
