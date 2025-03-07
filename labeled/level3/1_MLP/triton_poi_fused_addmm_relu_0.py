# From: 1_MLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_relu_0poi_fused_addmm_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_value = tl.load(in_out_ptr0 + (x0), xmask)
    weight_value = tl.load(in_ptr0 + (x0), xmask)
    sum_result = input_value + weight_value
    zero_value = tl.full([1], 0, tl.int32)
    relu_result = triton_helpers.maximum(zero_value, sum_result)
    tl.store(in_out_ptr0 + (x0), relu_result, xmask)