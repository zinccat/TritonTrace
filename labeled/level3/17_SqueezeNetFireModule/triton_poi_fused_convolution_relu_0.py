# From: 17_SqueezeNetFireModule

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_0poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    linear_index = block_indices
    channel_index = ((block_indices // 50176) % 6)
    
    output_value = tl.load(in_out_ptr0 + (linear_index), None)
    input_value = tl.load(in_ptr0 + (channel_index), None, eviction_policy='evict_last')
    
    fused_value = output_value + input_value
    zero_value = tl.full([1], 0, tl.int32)
    
    relu_output = triton_helpers.maximum(zero_value, fused_value)
    tl.store(in_out_ptr0 + (linear_index), relu_output, None)