# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_mul_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    global_index = block_indices
    batch_index = (block_indices // 123039) % 16
    spatial_index = block_indices % 3969
    depth_index = (block_indices // 3969)
    
    input_output_value = tl.load(in_out_ptr0 + (global_index), None)
    input_value_0 = tl.load(in_ptr0 + (batch_index), None, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr1 + (0))
    
    broadcast_value = tl.broadcast_to(input_value_1, [XBLOCK])
    
    accumulated_value = input_output_value + input_value_0
    scaled_value = accumulated_value * broadcast_value
    
    tl.store(in_out_ptr0 + (global_index), accumulated_value, None)
    tl.store(out_ptr0 + (spatial_index + (4000 * depth_index)), scaled_value, None)