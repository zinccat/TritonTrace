# From: 56_Matmul_Sigmoid_Sum

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_sigmoid_sum_0(input_ptr, output_ptr, num_elements_x, num_elements_r, XBLOCK: tl.constexpr):
    num_elements_x = 128
    num_elements_r = 20
    RBLOCK: tl.constexpr = 32

    # Calculate the offset for the current program ID
    x_offset = tl.program_id(0) * XBLOCK

    # Generate indices for x and r dimensions
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_elements_r

    # Load data from input pointer with masking
    loaded_data = tl.load(input_ptr + (r_indices + (20 * x_indices)), r_mask & x_mask, other=0.0)

    # Apply sigmoid function
    sigmoid_result = tl.sigmoid(loaded_data)

    # Broadcast the sigmoid result to match the shape [XBLOCK, RBLOCK]
    broadcasted_result = tl.broadcast_to(sigmoid_result, [XBLOCK, RBLOCK])

    # Apply mask and zero out elements outside the mask
    masked_result = tl.where(r_mask & x_mask, broadcasted_result, 0)

    # Sum along the r dimension
    summed_result = tl.sum(masked_result, 1)[:, None]

    # Store the result in the output pointer
    tl.store(output_ptr + (x_indices), summed_result, x_mask)