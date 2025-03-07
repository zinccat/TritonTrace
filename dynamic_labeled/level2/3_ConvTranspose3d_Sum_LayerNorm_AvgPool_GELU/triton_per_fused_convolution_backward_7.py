# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_7(input_ptr, output_ptr, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    input_num_elements = 64
    RBLOCK: tl.constexpr = 128
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_1 = reduction_index
    input_0 = input_index
    temp_0 = tl.load(input_ptr + (input_0 + 64 * reduction_1), input_mask, other=0.0)
    temp_1 = tl.broadcast_to(temp_0, [XBLOCK, RBLOCK])
    temp_3 = tl.where(input_mask, temp_1, 0)
    temp_4 = tl.sum(temp_3, 1)[:, None]
    tl.store(output_ptr + (input_0), temp_4, input_mask)