# From: 25_Swish

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_mul_sigmoid_0(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    input_indices = block_indices
    input_values = tl.load(input_ptr + (input_indices), None)
    sigmoid_values = tl.sigmoid(input_values)
    fused_values = input_values * sigmoid_values
    tl.store(output_ptr + (input_indices), fused_values, None)