# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_1per_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, kernel_size0, kernel_size1, kernel_size2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_index_2 = reduction_index
    kernel_index0 = (input_index % kernel_size0)
    kernel_index1 = input_index // kernel_size0
    linear_index = input_index
    temp_max = tl.load(in_ptr0 + (kernel_index0 + kernel_size1 * reduction_index_2 * kernel_size2 * kernel_size2 + 16 * kernel_size1 * kernel_index1 * kernel_size2 * kernel_size2), input_mask, eviction_policy='evict_last', other=0.0)
    temp_broadcast = tl.broadcast_to(temp_max, [XBLOCK, RBLOCK])
    temp_masked = tl.where(input_mask, temp_broadcast, float("-inf"))
    max_values = triton_helpers.max2(temp_masked, 1)[:, None]
    temp_subtracted = temp_max - max_values
    temp_exp = tl.math.exp(temp_subtracted)
    temp_exp_broadcast = tl.broadcast_to(temp_exp, [XBLOCK, RBLOCK])
    temp_exp_masked = tl.where(input_mask, temp_exp_broadcast, 0)
    sum_exp = tl.sum(temp_exp_masked, 1)[:, None]
    tl.store(out_ptr0 + (linear_index), max_values, input_mask)
    tl.store(out_ptr1 + (linear_index), sum_exp, input_mask)