# From: 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mul_sigmoid_backward_sum_tanh_2per_fused_mul_sigmoid_backward_sum_tanh_2(
    input_ptr, output_ptr, num_elements_x, num_elements_r, BLOCK_SIZE_X: tl.constexpr
):
    num_elements_x = 16
    num_elements_r = 21
    BLOCK_SIZE_R: tl.constexpr = 32

    # Calculate the starting index for the current block
    start_index_x = tl.program_id(0) * BLOCK_SIZE_X
    index_x = start_index_x + tl.arange(0, BLOCK_SIZE_X)[:, None]

    # Create masks to ensure indices are within bounds
    mask_x = index_x < num_elements_x
    index_r = tl.arange(0, BLOCK_SIZE_R)[None, :]
    mask_r = index_r < num_elements_r

    # Load data from input pointer with masking
    load_index = index_x + 16 * index_r
    loaded_data = tl.load(input_ptr + load_index, mask_r & mask_x, other=0.0)

    # Broadcast loaded data to the block size
    broadcasted_data = tl.broadcast_to(loaded_data, [BLOCK_SIZE_X, BLOCK_SIZE_R])

    # Apply mask and zero out out-of-bound elements
    masked_data = tl.where(mask_r & mask_x, broadcasted_data, 0)

    # Sum along the r dimension
    summed_data = tl.sum(masked_data, 1)[:, None]

    # Store the result in the output pointer
    tl.store(output_ptr + index_x, summed_data, mask_x)