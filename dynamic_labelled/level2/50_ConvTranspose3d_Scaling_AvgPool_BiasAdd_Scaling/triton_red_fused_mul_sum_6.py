# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sum_6red_fused_mul_sum_6(input_ptr, output_ptr, total_elements, block_size, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    block_size = 1923
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    rmask_full = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, block_size, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < block_size
        r_indices_clamped = r_indices
        loaded_values = tl.load(input_ptr + (r_indices_clamped), r_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_accumulator = temp_accumulator + broadcasted_values
        temp_accumulator = tl.where(r_mask, temp_accumulator, temp_accumulator)
    
    summed_values = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (tl.full([XBLOCK, 1], 0, tl.int32)), summed_values, None)