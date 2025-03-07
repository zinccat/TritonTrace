# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_6(input_ptr, output_ptr, input_num_elements, result_num_elements, XBLOCK: tl.constexpr):
    input_num_elements = 16
    result_num_elements = 123
    RBLOCK: tl.constexpr = 128

    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements

    result_indices = tl.arange(0, RBLOCK)[None, :]
    result_mask = result_indices < result_num_elements

    result_index = result_indices
    input_index = input_indices

    loaded_values = tl.load(input_ptr + (input_index + 16 * result_index), result_mask & input_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(result_mask & input_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]

    tl.store(output_ptr + (input_index), summed_values, input_mask)