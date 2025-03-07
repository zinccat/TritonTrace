# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_ge_le_logical_and_mul_scalar_tensor_where_3(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements

    element_index = index
    channel_index = (index // kernel_size0) % 16
    depth_index = index // kernel_size1

    input_data0 = tl.load(input_ptr0 + (element_index), mask, eviction_policy='evict_last')
    input_data1 = tl.load(input_ptr1 + (channel_index), mask, eviction_policy='evict_last')
    input_data2 = tl.load(input_ptr2 + (depth_index), mask, eviction_policy='evict_last')
    input_data3 = tl.load(input_ptr3 + (depth_index), mask, eviction_policy='evict_last')
    input_data4 = tl.load(input_ptr4 + (element_index), mask, eviction_policy='evict_last')

    intermediate_result1 = input_data0 * input_data1
    intermediate_result2 = intermediate_result1 - input_data2
    intermediate_result3 = intermediate_result2 * input_data3

    lower_bound = -1.0
    upper_bound = 1.0

    is_within_bounds = (intermediate_result3 >= lower_bound) & (intermediate_result3 <= upper_bound)

    scaled_input4 = input_data4 * input_data1
    zero_value = 0.0

    final_result = tl.where(is_within_bounds, scaled_input4, zero_value)

    tl.store(output_ptr0 + (element_index), final_result, mask)