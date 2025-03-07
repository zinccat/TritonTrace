# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_div_ge_le_logical_and_logsumexp_mul_neg_sigmoid_sub_sum_where_4(input_ptr, output_ptr, num_elements_x, num_elements_r, BLOCK_SIZE_X : tl.constexpr):
    num_elements_x = 16
    num_elements_r = 123
    BLOCK_SIZE_R: tl.constexpr = 128
    offset_x = tl.program_id(0) * BLOCK_SIZE_X
    index_x = offset_x + tl.arange(0, BLOCK_SIZE_X)[:, None]
    mask_x = index_x < num_elements_x
    index_r = tl.arange(0, BLOCK_SIZE_R)[None, :]
    mask_r = index_r < num_elements_r
    repeated_index_r = index_r
    original_index_x = index_x
    loaded_values = tl.load(input_ptr + (original_index_x + 16 * repeated_index_r), mask_r & mask_x, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [BLOCK_SIZE_X, BLOCK_SIZE_R])
    masked_values = tl.where(mask_r & mask_x, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    tl.store(output_ptr + (original_index_x), summed_values, mask_x)