# From: 100_ConvTranspose3d_Clamp_Min_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_scalar_tensor_where_0(input_ptr0, input_ptr1, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    base_indices = indices
    input_values = tl.load(input_ptr0 + (base_indices), mask).to(tl.int1)
    scalar_values = tl.load(input_ptr1 + (base_indices), mask)
    scalar_multiplier = 0.5
    scaled_values = scalar_values * scalar_multiplier
    default_value = 0.0
    result_values = tl.where(input_values, scaled_values, default_value)
    tl.store(output_ptr0 + (base_indices), result_values, mask)