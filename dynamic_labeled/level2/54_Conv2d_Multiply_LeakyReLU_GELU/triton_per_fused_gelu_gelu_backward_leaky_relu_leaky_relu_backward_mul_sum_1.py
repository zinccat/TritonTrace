# From: 54_Conv2d_Multiply_LeakyReLU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_mul_sum_1(input_ptr, output_ptr, num_elements_x, num_elements_r, BLOCK_SIZE_X : tl.constexpr):
    num_elements_x = 16
    num_elements_r = 15
    BLOCK_SIZE_R: tl.constexpr = 16
    
    # Calculate the starting index for the current block
    start_index_x = tl.program_id(0) * BLOCK_SIZE_X
    indices_x = start_index_x + tl.arange(0, BLOCK_SIZE_X)[:, None]
    
    # Create masks to ensure indices are within bounds
    mask_x = indices_x < num_elements_x
    indices_r = tl.arange(0, BLOCK_SIZE_R)[None, :]
    mask_r = indices_r < num_elements_r
    
    # Load data from input pointer with masking
    loaded_data = tl.load(input_ptr + (indices_x + 16 * indices_r), mask_r & mask_x, other=0.0)
    
    # Broadcast loaded data to the shape [BLOCK_SIZE_X, BLOCK_SIZE_R]
    broadcasted_data = tl.broadcast_to(loaded_data, [BLOCK_SIZE_X, BLOCK_SIZE_R])
    
    # Apply mask and zero out elements outside the valid range
    masked_data = tl.where(mask_r & mask_x, broadcasted_data, 0)
    
    # Sum along the second dimension and reshape
    summed_data = tl.sum(masked_data, 1)[:, None]
    
    # Store the result in the output pointer with masking
    tl.store(output_ptr + (indices_x), summed_data, mask_x)