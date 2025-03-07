# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_6poi_fused_convolution_relu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1140576
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel
    linear_index = block_indices
    channel_index = block_indices % 96
    output_value = tl.load(in_out_ptr0 + (linear_index), valid_mask)
    input_value = tl.load(in_ptr0 + (channel_index), valid_mask, eviction_policy='evict_last')
    fused_value = output_value + input_value
    zero_value = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_value, fused_value)
    tl.store(in_out_ptr0 + (linear_index), relu_output, valid_mask)