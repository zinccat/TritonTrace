# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_batch_norm_backward_2per_fused_native_batch_norm_backward_2(input_ptr, output_ptr, num_elements_x, num_elements_r, XBLOCK: tl.constexpr):
    num_elements_x = 16
    num_elements_r = 249
    RBLOCK: tl.constexpr = 256

    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x

    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_elements_r

    r_index = r_indices
    x_index = x_indices

    loaded_values = tl.load(input_ptr + (r_index + 249 * x_index), r_mask & x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(r_mask & x_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]

    tl.store(output_ptr + (x_index), summed_values, x_mask)