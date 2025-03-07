# From: 69_Conv2d_HardSwish_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_2(input_ptr, output_ptr, input_elements, output_elements, XBLOCK: tl.constexpr):
    input_elements = 16
    output_elements = 15
    RBLOCK: tl.constexpr = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_elements
    output_index = tl.arange(0, RBLOCK)[None, :]
    output_mask = output_index < output_elements
    output_row = output_index
    input_col = input_index
    temp0 = tl.load(input_ptr + (input_col + 16 * output_row), output_mask & input_mask, other=0.0)
    temp1 = tl.broadcast_to(temp0, [XBLOCK, RBLOCK])
    temp3 = tl.where(output_mask & input_mask, temp1, 0)
    temp4 = tl.sum(temp3, 1)[:, None]
    tl.store(output_ptr + (input_col), temp4, input_mask)