# From: 20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_0poi_fused_mul_0(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    indices = block_indices
    input_value = tl.load(input_ptr + (indices), None)
    output_value = tl.load(output_ptr + (indices), None)
    result = input_value * output_value
    tl.store(output_ptr + (indices), result, None)