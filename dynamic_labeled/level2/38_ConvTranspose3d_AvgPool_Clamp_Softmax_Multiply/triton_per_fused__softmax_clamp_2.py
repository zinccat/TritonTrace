# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_clamp_2(in_ptr0, out_ptr0, out_ptr1, kernel_size0, kernel_size1, kernel_size2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
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
    loaded_value = tl.load(in_ptr0 + (kernel_index0 + kernel_size1 * kernel_size2 * reduction_index_2 + 16 * kernel_size1 * kernel_size2 * kernel_index1), input_mask, eviction_policy='evict_last', other=0.0)
    zero_value = 0.0
    max_value = triton_helpers.maximum(loaded_value, zero_value)
    clamp_max = 1.0
    clamped_value = triton_helpers.minimum(max_value, clamp_max)
    broadcast_clamped = tl.broadcast_to(clamped_value, [XBLOCK, RBLOCK])
    clamped_or_neg_inf = tl.where(input_mask, broadcast_clamped, float("-inf"))
    max_across_reduction = triton_helpers.max2(clamped_or_neg_inf, 1)[:, None]
    shifted_values = clamped_value - max_across_reduction
    exp_values = tl.math.exp(shifted_values)
    broadcast_exp = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(input_mask, broadcast_exp, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    tl.store(out_ptr0 + (linear_index), max_across_reduction, input_mask)
    tl.store(out_ptr1 + (linear_index), sum_exp_values, input_mask)