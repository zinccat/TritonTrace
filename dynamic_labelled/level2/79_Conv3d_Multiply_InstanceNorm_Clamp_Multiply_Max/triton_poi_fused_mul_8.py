# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_8poi_fused_mul_8(in_out_ptr0, in_ptr0, kernel_size, num_elements, XBLOCK : tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    channel_index = ((block_indices // kernel_size) % 16)
    
    output_data = tl.load(in_out_ptr0 + (linear_index), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(in_ptr0 + (channel_index), valid_mask, eviction_policy='evict_last')
    
    result_data = output_data * input_data
    tl.store(in_out_ptr0 + (linear_index), result_data, valid_mask)