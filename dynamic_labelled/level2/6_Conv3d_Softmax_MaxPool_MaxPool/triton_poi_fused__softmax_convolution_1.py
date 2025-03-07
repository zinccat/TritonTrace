# From: 6_Conv3d_Softmax_MaxPool_MaxPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_convolution_1poi_fused__softmax_convolution_1(
    in_out_ptr, input_ptr0, input_ptr1, input_ptr2, kernel_size0, kernel_size1, kernel_size2, kernel_size3, kernel_size4, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    batch_index = (index // kernel_size0) % 16
    kernel_index1 = index % kernel_size1
    depth_index = index // kernel_size2

    loaded_output = tl.load(in_out_ptr + (linear_index), mask, eviction_policy='evict_last')
    loaded_input0 = tl.load(input_ptr0 + (batch_index), mask, eviction_policy='evict_last')
    loaded_input1 = tl.load(
        input_ptr1 + (
            kernel_index1 + 
            (-8) * depth_index + 
            (-2) * depth_index * kernel_size4 * kernel_size4 + 
            4 * kernel_size3 * depth_index + 
            8 * kernel_size4 * depth_index + 
            kernel_size3 * depth_index * kernel_size4 * kernel_size4 + 
            (-4) * kernel_size3 * kernel_size4 * depth_index
        ), 
        mask, 
        eviction_policy='evict_last'
    )
    loaded_input2 = tl.load(
        input_ptr2 + (
            kernel_index1 + 
            (-8) * depth_index + 
            (-2) * depth_index * kernel_size4 * kernel_size4 + 
            4 * kernel_size3 * depth_index + 
            8 * kernel_size4 * depth_index + 
            kernel_size3 * depth_index * kernel_size4 * kernel_size4 + 
            (-4) * kernel_size3 * kernel_size4 * depth_index
        ), 
        mask, 
        eviction_policy='evict_last'
    )

    sum_inputs = loaded_output + loaded_input0
    subtracted_input = sum_inputs - loaded_input1
    exp_result = tl.math.exp(subtracted_input)
    softmax_result = exp_result / loaded_input2

    tl.store(in_out_ptr + (linear_index), softmax_result, mask)