# From: 63_Gemm_ReLU_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_threshold_backward_0(input_ptr0, input_ptr1, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    base_indices = indices
    input_values = tl.load(input_ptr0 + (base_indices), mask).to(tl.int1)
    divisor_values = tl.load(input_ptr1 + (base_indices), mask)
    half_value = 0.5
    half_divisor = divisor_values * half_value
    zero_value = 0.0
    result_values = tl.where(input_values, zero_value, half_divisor)
    tl.store(output_ptr0 + (base_indices), result_values, mask)