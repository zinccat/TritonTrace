# From: 11_VGG16

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_9poi_fused_convolution_relu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = x_index
    x0 = (x_index % 64)
    output_value = tl.load(in_out_ptr0 + (x2), None)
    input_value = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    sum_value = output_value + input_value
    zero_value = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_value, sum_value)
    tl.store(in_out_ptr0 + (x2), relu_output, None)