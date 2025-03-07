# From: 24_LogSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_2(in_ptr0, in_ptr1, out_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_even_index = x_index % 2
    x_odd_index = x_index // 2
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_full_index = x_index

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < reduction_num_elements
        r_full_index = r_index
        temp_index = r_full_index + x_even_index * ((1 + kernel_size) // 2)
        kernel_limit = kernel_size
        within_kernel = temp_index < kernel_limit
        value0 = tl.load(in_ptr0 + (r_full_index + kernel_size * x_odd_index + x_even_index * ((1 + kernel_size) // 2)), r_mask & within_kernel & x_mask, eviction_policy='evict_first', other=0.0)
        value1 = tl.load(in_ptr1 + (tl.broadcast_to(x_odd_index, [XBLOCK, RBLOCK])), r_mask & within_kernel & x_mask, eviction_policy='evict_last', other=0.0)
        diff = value0 - value1
        exp_diff = tl.math.exp(diff)
        zero_filled = tl.full(exp_diff.shape, 0, exp_diff.dtype)
        masked_exp = tl.where(within_kernel, exp_diff, zero_filled)
        broadcast_exp = tl.broadcast_to(masked_exp, [XBLOCK, RBLOCK])
        temp_sum += broadcast_exp
        temp_sum = tl.where(r_mask & x_mask, temp_sum, temp_sum)

    sum_exp = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x_full_index), sum_exp, x_mask)