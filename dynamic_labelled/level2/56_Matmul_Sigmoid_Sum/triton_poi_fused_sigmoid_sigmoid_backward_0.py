# From: 56_Matmul_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_sigmoid_sigmoid_backward_0poi_fused_sigmoid_sigmoid_backward_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    input_block_index = xindex // 20
    output_index = xindex
    input_value = tl.load(in_ptr0 + (input_block_index), xmask, eviction_policy='evict_last')
    output_value = tl.load(in_out_ptr0 + (output_index), xmask)
    sigmoid_output = tl.sigmoid(output_value)
    one_minus_sigmoid = 1.0 - sigmoid_output
    gradient = sigmoid_output * one_minus_sigmoid
    updated_value = input_value * gradient
    tl.store(in_out_ptr0 + (output_index), updated_value, xmask)