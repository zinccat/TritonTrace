# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_4(input_ptr, output_ptr, input_elements, result_elements, XBLOCK: tl.constexpr):
    input_elements = 16
    result_elements = 15
    RBLOCK: tl.constexpr = 16
    
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_elements
    
    result_indices = tl.arange(0, RBLOCK)[None, :]
    result_mask = result_indices < result_elements
    
    result_row_indices = result_indices
    input_col_indices = input_indices
    
    loaded_values = tl.load(input_ptr + (input_col_indices + 16 * result_row_indices), result_mask & input_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(result_mask & input_mask, broadcasted_values, 0)
    
    summed_values = tl.sum(masked_values, 1)[:, None]
    tl.store(output_ptr + (input_col_indices), summed_values, input_mask)