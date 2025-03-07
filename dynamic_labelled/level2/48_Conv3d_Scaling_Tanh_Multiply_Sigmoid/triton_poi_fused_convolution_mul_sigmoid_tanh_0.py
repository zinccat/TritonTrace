# From: 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_tanh_0(
    input_output_ptr, input_ptr0, input_ptr1, input_ptr2, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = (indices // kernel_size) % 16

    input_output_data = tl.load(input_output_ptr + (linear_index), mask, eviction_policy='evict_last')
    input_data0 = tl.load(input_ptr0 + (channel_index), mask, eviction_policy='evict_last')
    input_data1 = tl.load(input_ptr1 + (channel_index), mask, eviction_policy='evict_last')
    input_data2 = tl.load(input_ptr2 + (channel_index), mask, eviction_policy='evict_last')

    sum_data = input_output_data + input_data0
    product_data = sum_data * input_data1
    tanh_data = tl.extra.cuda.libdevice.tanh(product_data)
    multiplied_data = tanh_data * input_data2
    sigmoid_data = tl.sigmoid(multiplied_data)

    tl.store(input_output_ptr + (linear_index), sum_data, mask)
    tl.store(output_ptr + (linear_index), sigmoid_data, mask)