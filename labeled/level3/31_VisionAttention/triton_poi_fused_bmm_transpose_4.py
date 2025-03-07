# From: 31_VisionAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bmm_transpose_4poi_fused_bmm_transpose_4(input_ptr, output_ptr0, output_ptr1, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    index = block_indices
    temp_value = tl.load(input_ptr + (8388608 + index), None)
    tl.store(output_ptr0 + (index), temp_value, None)
    tl.store(output_ptr1 + (index), temp_value, None)