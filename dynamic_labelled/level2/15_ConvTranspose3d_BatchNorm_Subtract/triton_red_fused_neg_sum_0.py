# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_neg_sum_0red_fused_neg_sum_0(input_ptr, output_ptr, kernel_size, total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    total_elements = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_indices_flat = r_indices
        loaded_values = tl.load(
            input_ptr + (r_indices_flat + 31 * x_indices_flat + ((-124) * kernel_size * x_indices_flat) + 124 * x_indices_flat * kernel_size * kernel_size),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        negated_values = -loaded_values
        broadcasted_values = tl.broadcast_to(negated_values, [XBLOCK, RBLOCK])
        temp_sum_update = temp_sum + broadcasted_values
        temp_sum = tl.where(r_mask & x_mask, temp_sum_update, temp_sum)
    
    summed_values = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr + (x_indices_flat), summed_values, x_mask)