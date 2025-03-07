# From: 4_LeNet5

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_relu_4poi_fused_addmm_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    output_data = tl.load(in_out_ptr0 + (x0), xmask)
    input_data = tl.load(in_ptr0 + (x0), xmask)
    sum_data = output_data + input_data
    zero_value = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_value, sum_data)
    tl.store(in_out_ptr0 + (x0), relu_output, xmask)