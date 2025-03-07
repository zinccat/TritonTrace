# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_4per_fused_convolution_backward_4(input_ptr, output_ptr, input_elements, result_elements, INPUT_BLOCK : tl.constexpr):
    input_elements = 16
    result_elements = 21
    RESULT_BLOCK: tl.constexpr = 32
    input_offset = tl.program_id(0) * INPUT_BLOCK
    input_index = input_offset + tl.arange(0, INPUT_BLOCK)[:, None]
    input_mask = input_index < input_elements
    result_index = tl.arange(0, RESULT_BLOCK)[None, :]
    result_mask = result_index < result_elements
    result_row = result_index
    input_col = input_index
    temp_data = tl.load(input_ptr + (result_row + 21*input_col), result_mask & input_mask, other=0.0)
    temp_broadcast = tl.broadcast_to(temp_data, [INPUT_BLOCK, RESULT_BLOCK])
    temp_filtered = tl.where(result_mask & input_mask, temp_broadcast, 0)
    temp_sum = tl.sum(temp_filtered, 1)[:, None]
    tl.store(output_ptr + (input_col), temp_sum, input_mask)