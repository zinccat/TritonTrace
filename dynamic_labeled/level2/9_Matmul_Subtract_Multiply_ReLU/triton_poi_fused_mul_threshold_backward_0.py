# From: 9_Matmul_Subtract_Multiply_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_threshold_backward_0(input_ptr0, input_ptr1, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    index = indices
    input_value0 = tl.load(input_ptr0 + (index), mask).to(tl.int1)
    input_value1 = tl.load(input_ptr1 + (index), mask)
    threshold_value = 0.0
    selected_value = tl.where(input_value0, threshold_value, input_value1)
    multiplier = 1.5
    result_value = selected_value * multiplier
    tl.store(output_ptr0 + (index), result_value, mask)