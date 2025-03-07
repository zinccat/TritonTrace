# From: 31_VisionAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_2poi_fused_mul_2(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    element_indices = block_indices
    input_value = tl.load(input_ptr + (element_indices), None)
    multiplier = 0.1767766952966369
    result = input_value * multiplier
    tl.store(output_ptr + (element_indices), result, None)