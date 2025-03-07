# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_avg_pool3d_mul_1(input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    x_dim = block_indices % 31
    y_dim = (block_indices // 31) % 31
    z_dim = (block_indices // 961) % 15
    batch_dim = (block_indices // 14415)
    linear_index = block_indices % 14415
    depth_dim = (block_indices // 14415) % 16
    full_index = block_indices
    
    # Load data from input_ptr0 with different offsets
    input_val0 = tl.load(input_ptr0 + ((2 * x_dim) + (126 * y_dim) + (8000 * z_dim) + (124000 * batch_dim)), None, eviction_policy='evict_last')
    input_val1 = tl.load(input_ptr0 + (1 + (2 * x_dim) + (126 * y_dim) + (8000 * z_dim) + (124000 * batch_dim)), None, eviction_policy='evict_last')
    input_val2 = tl.load(input_ptr0 + (63 + (2 * x_dim) + (126 * y_dim) + (8000 * z_dim) + (124000 * batch_dim)), None, eviction_policy='evict_last')
    input_val3 = tl.load(input_ptr0 + (64 + (2 * x_dim) + (126 * y_dim) + (8000 * z_dim) + (124000 * batch_dim)), None, eviction_policy='evict_last')
    input_val4 = tl.load(input_ptr0 + (4000 + (2 * x_dim) + (126 * y_dim) + (8000 * z_dim) + (124000 * batch_dim)), None, eviction_policy='evict_last')
    input_val5 = tl.load(input_ptr0 + (4001 + (2 * x_dim) + (126 * y_dim) + (8000 * z_dim) + (124000 * batch_dim)), None, eviction_policy='evict_last')
    input_val6 = tl.load(input_ptr0 + (4063 + (2 * x_dim) + (126 * y_dim) + (8000 * z_dim) + (124000 * batch_dim)), None, eviction_policy='evict_last')
    input_val7 = tl.load(input_ptr0 + (4064 + (2 * x_dim) + (126 * y_dim) + (8000 * z_dim) + (124000 * batch_dim)), None, eviction_policy='evict_last')
    
    # Load data from input_ptr1 and input_ptr2
    input_val8 = tl.load(input_ptr1 + (depth_dim), None, eviction_policy='evict_last')
    input_val9 = tl.load(input_ptr2 + (0))
    
    # Broadcast input_val9 to the block size
    broadcast_val = tl.broadcast_to(input_val9, [BLOCK_SIZE])
    
    # Compute the average pooling
    sum_val1 = input_val1 + input_val0
    sum_val2 = input_val2 + sum_val1
    sum_val3 = input_val3 + sum_val2
    sum_val4 = input_val4 + sum_val3
    sum_val5 = input_val5 + sum_val4
    sum_val6 = input_val6 + sum_val5
    sum_val7 = input_val7 + sum_val6
    
    # Scale the result
    scale_factor = 0.125
    scaled_sum = sum_val7 * scale_factor
    
    # Add the value from input_ptr1
    final_sum = scaled_sum + input_val8
    
    # Multiply by the broadcasted value
    result = final_sum * broadcast_val
    
    # Store the results
    tl.store(output_ptr0 + (linear_index + (14432 * batch_dim)), scaled_sum, None)
    tl.store(output_ptr1 + (full_index), result, None)