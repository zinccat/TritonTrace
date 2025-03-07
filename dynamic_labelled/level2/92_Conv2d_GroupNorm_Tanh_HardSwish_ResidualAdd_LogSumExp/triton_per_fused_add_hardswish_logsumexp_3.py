# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_hardswish_logsumexp_3(input_ptr0, input_ptr1, output_ptr0, output_ptr1, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = (x_index % kernel_size0)
    x4 = x_index // kernel_size0
    x5 = x_index

    input_value0 = tl.load(input_ptr0 + (x3 + 4*r2 + 64*x4 + r2*kernel_size1*kernel_size1 + ((-64)*kernel_size1*x4) + ((-4)*kernel_size1*r2) + 16*x4*kernel_size1*kernel_size1), x_mask, eviction_policy='evict_last', other=0.0)
    input_value1 = tl.load(input_ptr1 + (x3 + 4*r2 + 64*x4 + r2*kernel_size1*kernel_size1 + ((-64)*kernel_size1*x4) + ((-4)*kernel_size1*r2) + 16*x4*kernel_size1*kernel_size1), x_mask, eviction_policy='evict_last', other=0.0)

    bias = 3.0
    input_plus_bias = input_value1 + bias

    lower_bound = 0.0
    upper_bound = 6.0

    clamped_value = triton_helpers.minimum(triton_helpers.maximum(input_plus_bias, lower_bound), upper_bound)
    scaled_value = clamped_value * input_value1 * 0.16666666666666666

    hardswish_output = input_value0 + scaled_value
    broadcasted_output = tl.broadcast_to(hardswish_output, [XBLOCK, RBLOCK])

    masked_output = tl.where(x_mask, broadcasted_output, float("-inf"))
    max_value = triton_helpers.max2(masked_output, 1)[:, None]
    abs_max_value = tl.math.abs(max_value)

    inf_value = float("inf")
    is_inf = abs_max_value == inf_value
    safe_max_value = tl.where(is_inf, lower_bound, max_value)

    shifted_output = hardswish_output - safe_max_value
    exp_output = tl.math.exp(shifted_output)
    broadcasted_exp = tl.broadcast_to(exp_output, [XBLOCK, RBLOCK])

    masked_exp = tl.where(x_mask, broadcasted_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]

    tl.store(output_ptr0 + (x5), safe_max_value, x_mask)
    tl.store(output_ptr1 + (x5), sum_exp, x_mask)