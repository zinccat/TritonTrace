# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_in_output = block_indices
    index_in_input = ((block_indices // 3136) % 96)
    
    value_from_output = tl.load(in_out_ptr0 + (index_in_output), None)
    value_from_input = tl.load(in_ptr0 + (index_in_input), None, eviction_policy='evict_last')
    
    result_value = value_from_output + value_from_input
    tl.store(in_out_ptr0 + (index_in_output), result_value, None)