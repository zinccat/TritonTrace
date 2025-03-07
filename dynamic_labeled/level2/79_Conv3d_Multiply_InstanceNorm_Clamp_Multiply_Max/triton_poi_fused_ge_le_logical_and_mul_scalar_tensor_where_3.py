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
    linear_index = index
    batch_index = (index // kernel_size0) % 16
    depth_index = index // kernel_size1

    input0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input1 = tl.load(input_ptr1 + (batch_index), mask, eviction_policy='evict_last')
    input2 = tl.load(input_ptr2 + (depth_index), mask, eviction_policy='evict_last')
    input3 = tl.load(input_ptr3 + (depth_index), mask, eviction_policy='evict_last')
    input4 = tl.load(input_ptr4 + (linear_index), mask, eviction_policy='evict_last')

    multiply_result = input0 * input1
    subtract_result = multiply_result - input2
    multiply_result2 = subtract_result * input3

    lower_bound = -1.0
    upper_bound = 1.0

    is_greater_equal = multiply_result2 >= lower_bound
    is_less_equal = multiply_result2 <= upper_bound
    within_bounds = is_greater_equal & is_less_equal

    multiply_result3 = input4 * input1
    zero_value = 0.0

    result = tl.where(within_bounds, multiply_result3, zero_value)
    tl.store(output_ptr0 + (linear_index), result, mask)