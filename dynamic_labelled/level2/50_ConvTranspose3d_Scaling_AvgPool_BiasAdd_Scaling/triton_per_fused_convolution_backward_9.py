# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_9(
    input_ptr, output_ptr, input_num_elements, output_num_elements, XBLOCK: tl.constexpr
):
    input_num_elements = 16
    output_num_elements = 123
    RBLOCK: tl.constexpr = 128

    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements

    output_indices = tl.arange(0, RBLOCK)[None, :]
    output_mask = output_indices < output_num_elements

    output_row_indices = output_indices
    input_col_indices = input_indices

    loaded_values = tl.load(input_ptr + (input_col_indices + 16 * output_row_indices), output_mask & input_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(output_mask & input_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]

    tl.store(output_ptr + (input_col_indices), summed_values, input_mask)