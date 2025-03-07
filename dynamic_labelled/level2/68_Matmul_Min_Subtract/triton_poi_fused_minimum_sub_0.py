# From: 68_Matmul_Min_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_minimum_sub_0poi_fused_minimum_sub_0(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    input_values = tl.load(input_ptr0 + (indices), valid_mask)
    scalar_value = tl.load(input_ptr1 + (0))
    broadcasted_value = tl.broadcast_to(scalar_value, [BLOCK_SIZE])
    min_values = triton_helpers.minimum(input_values, broadcasted_value)
    result_values = min_values - broadcasted_value
    tl.store(output_ptr0 + (indices), result_values, valid_mask)