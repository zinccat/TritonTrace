# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_mul_neg_sigmoid_sigmoid_backward_sub_sum_3(
    input_ptr, output_ptr, num_elements, reduced_num_elements, BLOCK_SIZE : tl.constexpr
):
    num_elements = 16
    reduced_num_elements = 21
    REDUCED_BLOCK_SIZE: tl.constexpr = 32
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:, None]
    block_mask = block_indices < num_elements
    reduced_indices = tl.arange(0, REDUCED_BLOCK_SIZE)[None, :]
    reduced_mask = reduced_indices < reduced_num_elements
    reduced_index = reduced_indices
    block_index = block_indices
    loaded_values = tl.load(input_ptr + (block_index + 16 * reduced_index), reduced_mask & block_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [BLOCK_SIZE, REDUCED_BLOCK_SIZE])
    masked_values = tl.where(reduced_mask & block_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    tl.store(output_ptr + (block_index), summed_values, block_mask)