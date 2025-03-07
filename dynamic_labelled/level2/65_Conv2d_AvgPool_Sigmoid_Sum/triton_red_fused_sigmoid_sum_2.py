# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sigmoid_sum_2red_fused_sigmoid_sum_2(input_ptr, output_ptr, kernel_size, total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_indices_1 = r_indices
        temp_load = tl.load(
            input_ptr + (r_indices_1 + 16 * x_indices_0 + ((-32) * x_indices_0 * (kernel_size // 2)) + 16 * x_indices_0 * (kernel_size // 2) * (kernel_size // 2)),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        temp_sigmoid = tl.sigmoid(temp_load)
        temp_broadcast = tl.broadcast_to(temp_sigmoid, [XBLOCK, RBLOCK])
        temp_accumulate = temp_sum + temp_broadcast
        temp_sum = tl.where(r_mask & x_mask, temp_accumulate, temp_sum)
    
    temp_final_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr + (x_indices_0), temp_final_sum, x_mask)