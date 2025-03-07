# From: 2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_mul_scalar_tensor_where_0poi_fused_div_mul_scalar_tensor_where_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    indices = block_indices

    input_data0 = tl.load(input_ptr0 + (indices), mask).to(tl.int1)
    input_data1 = tl.load(input_ptr1 + (indices), mask).to(tl.int1)
    input_data2 = tl.load(input_ptr2 + (indices), mask)

    scale_factor = 0.5
    scaled_data = input_data2 * scale_factor

    zero_value = 0.0
    conditional_scaled_data = tl.where(input_data1, scaled_data, zero_value)

    multiply_factor = 2.0
    multiplied_data = conditional_scaled_data * multiply_factor

    conditional_output = tl.where(input_data0, multiplied_data, zero_value)

    tl.store(output_ptr0 + (indices), conditional_output, mask)