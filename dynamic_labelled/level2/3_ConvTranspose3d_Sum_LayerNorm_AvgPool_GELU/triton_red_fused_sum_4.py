# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_4red_fused_sum_4(input_ptr, output_ptr, kernel_size_0, kernel_size_1, x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_mask_full = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, r_num_elements, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < r_num_elements
        r_indices_1 = r_indices
        temp_load = tl.load(
            input_ptr + (64 * (((r_indices_1 + 64 * kernel_size_0 * kernel_size_1 * x_indices_0) // 64) % (8192 * kernel_size_0 * kernel_size_1))) + ((r_indices_1 % 64)),
            r_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(r_mask, temp_sum, temp_accumulator)
    
    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (x_indices_0), temp_result, None)