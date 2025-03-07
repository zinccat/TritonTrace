# From: 56_Matmul_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_sigmoid_sum_0(input_ptr, output_ptr, num_elements_x, num_elements_r, XBLOCK: tl.constexpr):
    num_elements_r = 20
    RBLOCK: tl.constexpr = 32
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_elements_r
    r_indices_adjusted = r_indices
    x_indices_adjusted = x_indices
    loaded_values = tl.load(input_ptr + (r_indices_adjusted + 20 * x_indices_adjusted), r_mask & x_mask, other=0.0)
    sigmoid_values = tl.sigmoid(loaded_values)
    broadcasted_values = tl.broadcast_to(sigmoid_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(r_mask & x_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    tl.store(output_ptr + (x_indices_adjusted), summed_values, x_mask)