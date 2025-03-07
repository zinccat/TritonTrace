# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_mul_sigmoid_backward_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, kernel_size0, kernel_size1, kernel_size2, 
    num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index = index % kernel_size0
    batch_index = index // kernel_size1
    
    input_value0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (linear_index), mask, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (kernel_index + batch_index + 4*kernel_size2*batch_index + 4*batch_index*kernel_size2*kernel_size2), mask, eviction_policy='evict_last')
    input_value4 = tl.load(input_ptr4 + (kernel_index + batch_index + 4*kernel_size2*batch_index + 4*batch_index*kernel_size2*kernel_size2), mask, eviction_policy='evict_last')
    
    one = 1.0
    one_minus_input_value1 = one - input_value1
    input_value1_times_one_minus_input_value1 = input_value1 * one_minus_input_value1
    input_value0_times_input_value1_times_one_minus_input_value1 = input_value0 * input_value1_times_one_minus_input_value1
    
    two = 2.0
    input_value0_times_input_value1_times_one_minus_input_value1_times_two = input_value0_times_input_value1_times_one_minus_input_value1 * two
    
    input_value2_minus_input_value3 = input_value2 - input_value3
    exp_input_value2_minus_input_value3 = tl.math.exp(input_value2_minus_input_value3)
    exp_input_value2_minus_input_value3_div_input_value4 = exp_input_value2_minus_input_value3 / input_value4
    
    result = input_value0_times_input_value1_times_one_minus_input_value1_times_two * exp_input_value2_minus_input_value3_div_input_value4
    tl.store(output_ptr0 + (linear_index), result, mask)