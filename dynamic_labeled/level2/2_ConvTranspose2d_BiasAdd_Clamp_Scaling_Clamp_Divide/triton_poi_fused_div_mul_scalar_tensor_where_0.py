# From: 2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_mul_scalar_tensor_where_0(input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    base_index = index
    input_data0 = tl.load(input_ptr0 + (base_index), mask).to(tl.int1)
    input_data1 = tl.load(input_ptr1 + (base_index), mask).to(tl.int1)
    input_data2 = tl.load(input_ptr2 + (base_index), mask)
    scalar_half = 0.5
    scaled_data = input_data2 * scalar_half
    zero_value = 0.0
    conditional_result = tl.where(input_data1, scaled_data, zero_value)
    scalar_two = 2.0
    multiplied_result = conditional_result * scalar_two
    final_result = tl.where(input_data0, multiplied_result, zero_value)
    tl.store(output_ptr0 + (base_index), final_result, mask)