# From: 68_Matmul_Min_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_minimum_sub_0(input_ptr0, input_ptr1, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    base_indices = indices
    input_values = tl.load(input_ptr0 + (base_indices), mask)
    constant_value = tl.load(input_ptr1 + (0))
    broadcasted_constant = tl.broadcast_to(constant_value, [XBLOCK])
    min_values = triton_helpers.minimum(input_values, broadcasted_constant)
    result_values = min_values - broadcasted_constant
    tl.store(output_ptr0 + (base_indices), result_values, mask)