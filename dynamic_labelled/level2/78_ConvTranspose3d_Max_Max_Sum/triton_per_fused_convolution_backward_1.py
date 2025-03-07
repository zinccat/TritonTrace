# From: 78_ConvTranspose3d_Max_Max_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_1(input_ptr, output_ptr, input_num_elements, output_num_elements, INPUT_BLOCK : tl.constexpr):
    input_num_elements = 16
    output_num_elements = 21
    OUTPUT_BLOCK: tl.constexpr = 32
    
    input_offset = tl.program_id(0) * INPUT_BLOCK
    input_indices = input_offset + tl.arange(0, INPUT_BLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    
    output_indices = tl.arange(0, OUTPUT_BLOCK)[None, :]
    output_mask = output_indices < output_num_elements
    
    output_row_indices = output_indices
    input_col_indices = input_indices
    
    loaded_values = tl.load(input_ptr + (output_row_indices + 21 * input_col_indices), output_mask & input_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [INPUT_BLOCK, OUTPUT_BLOCK])
    masked_values = tl.where(output_mask & input_mask, broadcasted_values, 0)
    
    summed_values = tl.sum(masked_values, 1)[:, None]
    tl.store(output_ptr + (input_col_indices), summed_values, input_mask)