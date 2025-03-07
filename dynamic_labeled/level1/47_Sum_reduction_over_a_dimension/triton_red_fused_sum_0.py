# From: 47_Sum_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, kernel_size0, kernel_size1, x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = ((x_index // kernel_size0) % 2)
    x0 = (x_index % kernel_size0)
    x2 = x_index // kernel_size1
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = x_index
    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r3 = r_index
        temp_index = r3 + x1 * ((1 + kernel_size0) // 2)
        kernel_size = kernel_size0
        index_mask = temp_index < kernel_size
        load_value = tl.load(in_ptr0 + (x0 + kernel_size0 * r3 + x2 * kernel_size0 * kernel_size0 + kernel_size0 * x1 * ((1 + kernel_size0) // 2)), r_mask & index_mask & x_mask, eviction_policy='evict_last', other=0.0)
        broadcast_value = tl.broadcast_to(load_value, [XBLOCK, RBLOCK])
        temp_sum += broadcast_value
        temp_sum = tl.where(r_mask & x_mask, temp_sum, temp_sum)
    final_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x4), final_sum, x_mask)