# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_mul_sigmoid_sigmoid_backward_4(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7, 
    output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    group_index = index // kernel_size0
    channel_index = ((index // kernel_size1) % 16)
    kernel_index = (index % kernel_size2)
    sub_channel_index = ((index // kernel_size2) % kernel_size2)
    batch_index = index // kernel_size1

    input0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input1 = tl.load(input_ptr1 + (group_index // 2), mask, eviction_policy='evict_last')
    input2 = tl.load(input_ptr2 + (channel_index), mask, eviction_policy='evict_last')
    input3 = tl.load(
        input_ptr3 + (
            kernel_index + 
            ((-2) * (((kernel_index + ((-2) * sub_channel_index) + kernel_size3 * sub_channel_index) // ((-2) + kernel_size3)) % ((-2) + kernel_size3))) + 
            4 * batch_index + 
            kernel_size3 * ((((kernel_index + ((-2) * sub_channel_index) + kernel_size3 * sub_channel_index) // ((-2) + kernel_size3)) % ((-2) + kernel_size3))) + 
            batch_index * kernel_size3 * kernel_size3 + 
            ((-4) * kernel_size3 * batch_index)
        ), 
        mask, 
        eviction_policy='evict_last'
    )
    input4 = tl.load(input_ptr4 + (channel_index), mask, eviction_policy='evict_last')
    input5 = tl.load(input_ptr5 + (channel_index), mask, eviction_policy='evict_last')
    input6 = tl.load(input_ptr6 + (group_index // 2), mask, eviction_policy='evict_last')
    input7 = tl.load(input_ptr7 + (group_index // 2), mask, eviction_policy='evict_last')
    input8 = tl.load(input_ptr3 + (linear_index), mask, eviction_policy='evict_last')

    intermediate1 = input1 * input2
    intermediate2 = input0 * intermediate1
    intermediate3 = input3 + input4
    intermediate4 = intermediate3 * input5
    sigmoid_output = tl.sigmoid(intermediate4)
    intermediate5 = sigmoid_output * input6
    intermediate6 = intermediate2 + intermediate5
    intermediate7 = intermediate6 + input7
    intermediate8 = input8 + input4
    intermediate9 = intermediate8 * input5
    sigmoid_backward = tl.sigmoid(intermediate9)
    one_minus_sigmoid = 1.0 - sigmoid_backward
    gradient = sigmoid_backward * one_minus_sigmoid
    final_output = intermediate7 * gradient

    tl.store(output_ptr0 + (linear_index), final_output, mask)