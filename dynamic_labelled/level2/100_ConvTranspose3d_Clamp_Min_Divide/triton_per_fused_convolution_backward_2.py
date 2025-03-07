# From: 100_ConvTranspose3d_Clamp_Min_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_2(input_ptr, output_ptr, input_elements, output_elements, XBLOCK: tl.constexpr):
    input_elements = 16
    output_elements = 21
    RBLOCK: tl.constexpr = 32
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_elements
    output_index = tl.arange(0, RBLOCK)[None, :]
    output_mask = output_index < output_elements
    output_row_index = output_index
    input_col_index = input_index
    temp_input = tl.load(input_ptr + (output_row_index + 21 * input_col_index), output_mask & input_mask, other=0.0)
    temp_broadcast = tl.broadcast_to(temp_input, [XBLOCK, RBLOCK])
    temp_filtered = tl.where(output_mask & input_mask, temp_broadcast, 0)
    temp_summed = tl.sum(temp_filtered, 1)[:, None]
    tl.store(output_ptr + (input_col_index), temp_summed, input_mask)