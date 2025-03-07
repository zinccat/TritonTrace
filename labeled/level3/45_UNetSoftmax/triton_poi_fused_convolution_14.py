# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_14poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    global_index = block_indices
    local_index = (block_indices // 512) % 512
    
    output_value = tl.load(in_out_ptr0 + (global_index), None)
    input_value = tl.load(in_ptr0 + (local_index), None, eviction_policy='evict_last')
    
    result_value = output_value + input_value
    tl.store(in_out_ptr0 + (global_index), result_value, None)