# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_5(input_ptr, output_ptr, total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    reduction_elements = 8192
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    block_mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < reduction_elements
        r_indices_clamped = r_indices
        loaded_values = tl.load(input_ptr + (r_indices_clamped), r_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_sum_updated = temp_sum + broadcasted_values
        temp_sum = tl.where(r_mask, temp_sum_updated, temp_sum)
    
    reduced_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr + (tl.full([XBLOCK, 1], 0, tl.int32)), reduced_sum, None)