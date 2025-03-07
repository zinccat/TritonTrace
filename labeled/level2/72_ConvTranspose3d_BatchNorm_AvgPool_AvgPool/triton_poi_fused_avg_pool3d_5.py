# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_avg_pool3d_5(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    # Calculate indices for 3D grid
    z_index = block_indices % 15
    y_index = (block_indices // 15) % 15
    x_index = (block_indices // 225) % 15
    batch_index = block_indices // 3375
    linear_index = block_indices
    
    # Load values from input tensor
    value_0 = tl.load(input_ptr + ((2 * z_index) + (62 * y_index) + (1922 * x_index) + (29792 * batch_index)), None, eviction_policy='evict_last')
    value_1 = tl.load(input_ptr + (1 + (2 * z_index) + (62 * y_index) + (1922 * x_index) + (29792 * batch_index)), None, eviction_policy='evict_last')
    value_3 = tl.load(input_ptr + (31 + (2 * z_index) + (62 * y_index) + (1922 * x_index) + (29792 * batch_index)), None, eviction_policy='evict_last')
    value_5 = tl.load(input_ptr + (32 + (2 * z_index) + (62 * y_index) + (1922 * x_index) + (29792 * batch_index)), None, eviction_policy='evict_last')
    value_7 = tl.load(input_ptr + (961 + (2 * z_index) + (62 * y_index) + (1922 * x_index) + (29792 * batch_index)), None, eviction_policy='evict_last')
    value_9 = tl.load(input_ptr + (962 + (2 * z_index) + (62 * y_index) + (1922 * x_index) + (29792 * batch_index)), None, eviction_policy='evict_last')
    value_11 = tl.load(input_ptr + (992 + (2 * z_index) + (62 * y_index) + (1922 * x_index) + (29792 * batch_index)), None, eviction_policy='evict_last')
    value_13 = tl.load(input_ptr + (993 + (2 * z_index) + (62 * y_index) + (1922 * x_index) + (29792 * batch_index)), None, eviction_policy='evict_last')
    
    # Accumulate values
    sum_2 = value_1 + value_0
    sum_4 = value_3 + sum_2
    sum_6 = value_5 + sum_4
    sum_8 = value_7 + sum_6
    sum_10 = value_9 + sum_8
    sum_12 = value_11 + sum_10
    sum_14 = value_13 + sum_12
    
    # Average the accumulated sum
    average_factor = 0.125
    average_value = sum_14 * average_factor
    
    # Store the result in the output tensor
    tl.store(output_ptr + (linear_index), average_value, None)