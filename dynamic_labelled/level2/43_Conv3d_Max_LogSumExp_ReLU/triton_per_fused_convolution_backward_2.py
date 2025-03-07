# From: 43_Conv3d_Max_LogSumExp_ReLU

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
    output_row = output_index
    input_col = input_index
    loaded_values = tl.load(input_ptr + (input_col + 16 * output_row), output_mask & input_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(output_mask & input_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    tl.store(output_ptr + (input_col), summed_values, input_mask)