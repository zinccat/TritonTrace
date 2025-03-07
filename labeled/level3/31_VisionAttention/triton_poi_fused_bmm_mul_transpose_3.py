# From: 31_VisionAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bmm_mul_transpose_3poi_fused_bmm_mul_transpose_3(input_ptr, output_ptr0, output_ptr1, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    index_within_block = block_indices
    temp_value = tl.load(input_ptr + (4194304 + index_within_block), None)
    tl.store(output_ptr0 + (index_within_block), temp_value, None)
    tl.store(output_ptr1 + (index_within_block), temp_value, None)