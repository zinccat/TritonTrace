# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_exp_hardswish_mul_sub_5(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, 
    kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    col_index = index % kernel_size0
    row_index = index // kernel_size1
    linear_index = index
    depth_index = index // kernel_size0
    channel_index = ((index // kernel_size3) % 16)
    
    input_value0 = tl.load(in_ptr0 + (col_index + 4 * row_index + row_index * kernel_size2 * kernel_size2 + ((-4) * kernel_size2 * row_index)), mask, eviction_policy='evict_last')
    output_value = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    input_value2 = tl.load(in_ptr2 + (col_index + 4 * row_index + row_index * kernel_size2 * kernel_size2 + ((-4) * kernel_size2 * row_index)), mask, eviction_policy='evict_last')
    input_value3 = tl.load(in_ptr3 + (linear_index), mask, eviction_policy='evict_last')
    input_value4 = tl.load(in_ptr4 + (depth_index // 2), mask, eviction_policy='evict_last')
    input_value5 = tl.load(in_ptr5 + (channel_index), mask, eviction_policy='evict_last')
    input_value6 = tl.load(in_ptr6 + (depth_index // 2), mask, eviction_policy='evict_last')
    input_value7 = tl.load(in_ptr7 + (depth_index // 2), mask, eviction_policy='evict_last')
    
    constant3 = 3.0
    adjusted_input1 = input_value1 + constant3
    zero = 0.0
    max_value = triton_helpers.maximum(adjusted_input1, zero)
    constant6 = 6.0
    min_value = triton_helpers.minimum(max_value, constant6)
    scaled_input1 = input_value1 * min_value
    constant1_6 = 0.16666666666666666
    scaled_value = scaled_input1 * constant1_6
    updated_output = output_value + scaled_value
    difference = updated_output - input_value2
    exp_value = tl.math.exp(difference)
    product0 = input_value0 * exp_value
    
    product1 = input_value4 * input_value5
    product2 = input_value3 * product1
    product3 = output_value * input_value6
    sum_products = product2 + product3
    final_sum = sum_products + input_value7
    
    result = product0 + final_sum
    tl.store(in_out_ptr0 + (linear_index), result, mask)