# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_6per_fused_convolution_backward_6(input_ptr, output_ptr, input_num_elements, result_num_elements, INPUT_BLOCK : tl.constexpr):
    input_num_elements = 16
    result_num_elements = 123
    RESULT_BLOCK: tl.constexpr = 128
    input_offset = tl.program_id(0) * INPUT_BLOCK
    input_index = input_offset + tl.arange(0, INPUT_BLOCK)[:, None]
    input_mask = input_index < input_num_elements
    result_index = tl.arange(0, RESULT_BLOCK)[None, :]
    result_mask = result_index < result_num_elements
    result_row = result_index
    input_col = input_index
    temp_data = tl.load(input_ptr + (input_col + 16*result_row), result_mask & input_mask, other=0.0)
    temp_broadcast = tl.broadcast_to(temp_data, [INPUT_BLOCK, RESULT_BLOCK])
    temp_filtered = tl.where(result_mask & input_mask, temp_broadcast, 0)
    temp_summed = tl.sum(temp_filtered, 1)[:, None]
    tl.store(output_ptr + (input_col), temp_summed, input_mask)