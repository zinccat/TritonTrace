# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_7(
    input_ptr, output_ptr, input_num_elements, output_num_elements, XBLOCK: tl.constexpr
):
    input_num_elements = 16
    RBLOCK: tl.constexpr = 128
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    output_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    output_row = output_index
    input_col = input_index
    temp_data = tl.load(input_ptr + (input_col + 16 * output_row), input_mask, other=0.0)
    temp_broadcast = tl.broadcast_to(temp_data, [XBLOCK, RBLOCK])
    temp_masked = tl.where(input_mask, temp_broadcast, 0)
    temp_sum = tl.sum(temp_masked, 1)[:, None]
    tl.store(output_ptr + (input_col), temp_sum, input_mask)