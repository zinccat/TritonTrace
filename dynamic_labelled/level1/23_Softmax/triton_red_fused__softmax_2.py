# From: 23_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_2(in_ptr0, in_ptr1, out_ptr0, kernel_size, total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_even_indices = x_indices % 2
    x_odd_indices = x_indices // 2
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_full_indices = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_full_indices = r_indices
        temp_index0 = r_full_indices + x_even_indices * ((1 + kernel_size) // 2)
        temp_kernel_size = kernel_size
        temp_mask = temp_index0 < temp_kernel_size
        temp_load0 = tl.load(in_ptr0 + (r_full_indices + kernel_size * x_odd_indices + x_even_indices * ((1 + kernel_size) // 2)), r_mask & temp_mask & x_mask, eviction_policy='evict_first', other=0.0)
        temp_load1 = tl.load(in_ptr1 + (tl.broadcast_to(x_odd_indices, [XBLOCK, RBLOCK])), r_mask & temp_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp_diff = temp_load0 - temp_load1
        temp_exp = tl.math.exp(temp_diff)
        temp_zeros = tl.full(temp_exp.shape, 0, temp_exp.dtype)
        temp_selected = tl.where(temp_mask, temp_exp, temp_zeros)
        temp_broadcast = tl.broadcast_to(temp_selected, [XBLOCK, RBLOCK])
        temp_accumulated = temp_sum + temp_broadcast
        temp_sum = tl.where(r_mask & x_mask, temp_accumulated, temp_sum)

    temp_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x_full_indices), temp_result, x_mask)