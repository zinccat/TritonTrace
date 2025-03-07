# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_2(in_out_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    input_values = tl.load(in_out_ptr0 + (indices), valid_mask)
    divisor = 0.5
    result_values = input_values * divisor
    tl.store(in_out_ptr0 + (indices), result_values, valid_mask)