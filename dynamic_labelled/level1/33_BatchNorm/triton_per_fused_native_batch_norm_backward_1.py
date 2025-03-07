# From: 33_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_batch_norm_backward_1per_fused_native_batch_norm_backward_1(
    input_ptr, output_ptr, num_elements_x, num_elements_r, BLOCK_SIZE_X: tl.constexpr
):
    num_elements_x = 64
    num_elements_r = 6
    BLOCK_SIZE_R: tl.constexpr = 8

    x_offset = tl.program_id(0) * BLOCK_SIZE_X
    x_indices = x_offset + tl.arange(0, BLOCK_SIZE_X)[:, None]
    x_mask = x_indices < num_elements_x

    r_indices = tl.arange(0, BLOCK_SIZE_R)[None, :]
    r_mask = r_indices < num_elements_r

    r1 = r_indices
    x0 = x_indices

    loaded_values = tl.load(input_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [BLOCK_SIZE_X, BLOCK_SIZE_R])
    masked_values = tl.where(r_mask & x_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]

    tl.store(output_ptr + (x0), summed_values, x_mask)