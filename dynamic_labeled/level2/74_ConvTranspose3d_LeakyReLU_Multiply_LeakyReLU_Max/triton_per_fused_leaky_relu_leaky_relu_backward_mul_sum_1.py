# From: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_leaky_relu_leaky_relu_backward_mul_sum_1(input_ptr, output_ptr, input_num_elements, result_num_elements, INPUT_BLOCK: tl.constexpr):
    input_num_elements = 32
    result_num_elements = 11
    RESULT_BLOCK: tl.constexpr = 16
    
    input_offset = tl.program_id(0) * INPUT_BLOCK
    input_indices = input_offset + tl.arange(0, INPUT_BLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    
    result_indices = tl.arange(0, RESULT_BLOCK)[None, :]
    result_mask = result_indices < result_num_elements
    
    result_index = result_indices
    input_index = input_indices
    
    loaded_values = tl.load(input_ptr + (input_index + 32 * result_index), result_mask & input_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [INPUT_BLOCK, RESULT_BLOCK])
    masked_values = tl.where(result_mask & input_mask, broadcasted_values, 0)
    
    summed_values = tl.sum(masked_values, 1)[:, None]
    tl.store(output_ptr + (input_index), summed_values, input_mask)