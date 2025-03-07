# From: 68_Matmul_Min_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_minimum_sub_0(input_ptr0, input_ptr1, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    num_elements = 640
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    input_values0 = tl.load(input_ptr0 + (indices), valid_mask)
    input_value1 = tl.load(input_ptr1 + (0))
    broadcasted_value1 = tl.broadcast_to(input_value1, [XBLOCK])
    min_values = triton_helpers.minimum(input_values0, broadcasted_value1)
    result_values = min_values - broadcasted_value1
    tl.store(output_ptr0 + (indices), result_values, valid_mask)