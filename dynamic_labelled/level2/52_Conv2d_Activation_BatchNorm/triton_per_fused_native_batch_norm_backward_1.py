# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_batch_norm_backward_1per_fused_native_batch_norm_backward_1(
    input_ptr, output_ptr, num_elements_x, num_elements_r, BLOCK_SIZE_X: tl.constexpr
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

    # Load and broadcast
    loaded_values = tl.load(input_ptr + (x_indices + 16 * r_indices), r_mask & x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [BLOCK_SIZE_X, BLOCK_SIZE_R])
    masked_values = tl.where(r_mask & x_mask, broadcasted_values, 0)

    # Sum and store
    summed_values = tl.sum(masked_values, 1)[:, None]
    tl.store(output_ptr + (x_indices), summed_values, x_mask)