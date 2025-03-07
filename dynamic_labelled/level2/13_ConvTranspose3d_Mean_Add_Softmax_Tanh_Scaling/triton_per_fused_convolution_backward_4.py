# From: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_4(input_ptr, output_ptr, num_elements_x, num_elements_r, XBLOCK: tl.constexpr):
    num_elements_x = 16
    num_elements_r = 21
    RBLOCK: tl.constexpr = 32

    # Calculate the offset for the current program ID
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]

    # Create masks for valid indices
    x_mask = x_indices < num_elements_x
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_elements_r

    # Load data from input pointer with masking
    r_indices_adjusted = r_indices
    x_indices_adjusted = x_indices
    loaded_data = tl.load(input_ptr + (r_indices_adjusted + 21 * x_indices_adjusted), r_mask & x_mask, other=0.0)

    # Broadcast loaded data to the required shape
    broadcasted_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])

    # Apply mask and zero out invalid entries
    masked_data = tl.where(r_mask & x_mask, broadcasted_data, 0)

    # Sum along the first dimension and reshape
    summed_data = tl.sum(masked_data, 1)[:, None]

    # Store the result in the output pointer with masking
    tl.store(output_ptr + (x_indices_adjusted), summed_data, x_mask)