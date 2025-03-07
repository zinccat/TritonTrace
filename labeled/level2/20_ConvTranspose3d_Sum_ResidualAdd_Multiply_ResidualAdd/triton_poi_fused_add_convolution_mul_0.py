# From: 20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_convolution_mul_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Descriptive variable names
    global_index = xindex
    channel_index = (xindex // 131072) % 64
    
    # Load data
    in_out_value = tl.load(in_out_ptr0 + (global_index), None)
    input_value_0 = tl.load(in_ptr0 + (channel_index), None, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr1 + (channel_index), None, eviction_policy='evict_last')
    
    # Perform operations
    intermediate_sum_1 = in_out_value + input_value_0
    intermediate_sum_2 = intermediate_sum_1 + input_value_1
    intermediate_sum_3 = intermediate_sum_2 + intermediate_sum_1
    intermediate_product = intermediate_sum_3 * intermediate_sum_1
    final_result = intermediate_product + intermediate_sum_1
    
    # Store results
    tl.store(in_out_ptr0 + (global_index), intermediate_sum_1, None)
    tl.store(out_ptr0 + (global_index), final_result, None)