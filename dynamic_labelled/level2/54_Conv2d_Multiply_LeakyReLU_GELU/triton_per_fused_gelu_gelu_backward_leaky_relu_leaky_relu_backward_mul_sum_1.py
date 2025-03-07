# From: 54_Conv2d_Multiply_LeakyReLU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_mul_sum_1per_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_mul_sum_1(
    input_ptr, output_ptr, num_elements_x, num_elements_r, BLOCK_SIZE_X : tl.constexpr
):
    num_elements_x = 16
    num_elements_r = 15
    BLOCK_SIZE_R: tl.constexpr = 16

    # Calculate offsets and indices
    x_offset = tl.program_id(0) * BLOCK_SIZE_X
    x_indices = x_offset + tl.arange(0, BLOCK_SIZE_X)[:, None]
    x_mask = x_indices < num_elements_x

    r_indices = tl.arange(0, BLOCK_SIZE_R)[None, :]
    r_mask = r_indices < num_elements_r

    # Load and process data
    loaded_data = tl.load(input_ptr + (x_indices + 16 * r_indices), r_mask & x_mask, other=0.0)
    broadcasted_data = tl.broadcast_to(loaded_data, [BLOCK_SIZE_X, BLOCK_SIZE_R])
    masked_data = tl.where(r_mask & x_mask, broadcasted_data, 0)

    # Sum and store results
    summed_data = tl.sum(masked_data, 1)[:, None]
    tl.store(output_ptr + (x_indices), summed_data, x_mask)