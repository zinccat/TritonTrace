# From: 4_LeNet5

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_relu_5poi_fused_addmm_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 84
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_value = tl.load(in_out_ptr0 + (x0), xmask)
    addend_value = tl.load(in_ptr0 + (x0), xmask)
    sum_result = input_value + addend_value
    zero_value = tl.full([1], 0, tl.int32)
    relu_result = triton_helpers.maximum(zero_value, sum_result)
    tl.store(in_out_ptr0 + (x0), relu_result, xmask)