# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_0red_fused_sum_0(input_ptr, output_ptr, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_elements_x = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_indices_for_output = x_indices

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < num_elements_r
        r_indices_for_load = r_indices
        loaded_values = tl.load(input_ptr + (r_indices_for_load), r_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_accumulator_updated = temp_accumulator + broadcasted_values
        temp_accumulator = tl.where(r_mask & x_mask, temp_accumulator_updated, temp_accumulator)

    summed_values = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (x_indices_for_output), summed_values, x_mask)