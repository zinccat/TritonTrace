# From: 50_ReLUSelfAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_eq_0poi_fused_eq_0(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    indices = block_indices
    loaded_values = tl.load(input_ptr + (indices), None)
    zero_value = 0.0
    comparison_result = loaded_values == zero_value
    tl.store(output_ptr + (indices), comparison_result, None)