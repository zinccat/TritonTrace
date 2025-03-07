# From: 76_Gemm_Add_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_relu_threshold_backward_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    input_output_value = tl.load(in_out_ptr0 + (x2), None)
    input_value = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    sum_value = input_output_value + input_value
    zero_value = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_value, sum_value)
    threshold = 0.0
    is_below_threshold = relu_output <= threshold
    tl.store(in_out_ptr0 + (x2), relu_output, None)
    tl.store(out_ptr0 + (x2), is_below_threshold, None)