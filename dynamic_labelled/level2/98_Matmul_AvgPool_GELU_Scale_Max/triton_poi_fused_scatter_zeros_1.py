# From: 98_Matmul_AvgPool_GELU_Scale_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_scatter_zeros_1poi_fused_scatter_zeros_1(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    input_values0 = tl.load(input_ptr0 + (indices), valid_mask)
    input_values1 = tl.load(input_ptr1 + (indices), valid_mask)
    tl.device_assert(((0 <= input_values0) & (input_values0 < 64)) | ~valid_mask, "index out of bounds: 0 <= input_values0 < 64")
    tl.store(output_ptr0 + (input_values0 + 64*indices), input_values1, valid_mask)