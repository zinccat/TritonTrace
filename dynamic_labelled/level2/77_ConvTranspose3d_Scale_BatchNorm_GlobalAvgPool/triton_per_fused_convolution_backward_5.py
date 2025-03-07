# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_5(input_ptr, output_ptr, input_elements, output_elements, XBLOCK: tl.constexpr):
    input_elements = 32
    output_elements = 11
    RBLOCK: tl.constexpr = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_elements
    output_index = tl.arange(0, RBLOCK)[None, :]
    output_mask = output_index < output_elements
    output_row = output_index
    input_col = input_index
    temp_data = tl.load(input_ptr + (input_col + 32 * output_row), output_mask & input_mask, other=0.0)
    temp_broadcast = tl.broadcast_to(temp_data, [XBLOCK, RBLOCK])
    temp_filtered = tl.where(output_mask & input_mask, temp_broadcast, 0)
    temp_summed = tl.sum(temp_filtered, 1)[:, None]
    tl.store(output_ptr + (input_col), temp_summed, input_mask)