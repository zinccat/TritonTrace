# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mul_sigmoid_backward_sum_1per_fused_mul_sigmoid_backward_sum_1(input_ptr, output_ptr, input_num_elements, result_num_elements, INPUT_BLOCK : tl.constexpr):
    input_num_elements = 64
    result_num_elements = 7
    RESULT_BLOCK: tl.constexpr = 8
    input_offset = tl.program_id(0) * INPUT_BLOCK
    input_index = input_offset + tl.arange(0, INPUT_BLOCK)[:, None]
    input_mask = input_index < input_num_elements
    result_index = tl.arange(0, RESULT_BLOCK)[None, :]
    result_mask = result_index < result_num_elements
    result_row = result_index
    input_col = input_index
    loaded_values = tl.load(input_ptr + (input_col + 64*result_row), result_mask & input_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [INPUT_BLOCK, RESULT_BLOCK])
    masked_values = tl.where(result_mask & input_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    tl.store(output_ptr + (input_col), summed_values, input_mask)