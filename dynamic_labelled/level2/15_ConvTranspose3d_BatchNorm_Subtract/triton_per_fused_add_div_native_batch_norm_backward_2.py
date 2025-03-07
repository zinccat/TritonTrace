# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_div_native_batch_norm_backward_2(
    input_ptr, output_ptr, num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    num_elements_x = 32
    num_elements_r = 11
    RBLOCK: tl.constexpr = 16

    # Calculate offsets and indices
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x

    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_elements_r

    # Load and broadcast
    loaded_values = tl.load(input_ptr + (x_indices + 32 * r_indices), r_mask & x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])

    # Apply mask and sum
    masked_values = tl.where(r_mask & x_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]

    # Store the result
    tl.store(output_ptr + (x_indices), summed_values, x_mask)