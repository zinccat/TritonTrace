# From: 25_Conv2d_Min_Tanh_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_scatter_tanh_backward_zeros_1poi_fused_scatter_tanh_backward_zeros_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index_x = index % kernel_size0
    kernel_index_y = index // kernel_size0

    input_value0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (linear_index), mask, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (linear_index), mask, eviction_policy='evict_last')

    tl.device_assert(((0 <= input_value0) & (input_value0 < 16)) | ~mask, "index out of bounds: 0 <= input_value0 < 16")

    squared_value2 = input_value2 * input_value2
    one_minus_squared_value2 = 1.0 - squared_value2
    scaled_value1 = input_value1 * one_minus_squared_value2
    result = scaled_value1 * input_value3

    output_index = (
        kernel_index_x + 4 * input_value0 + 64 * kernel_index_y +
        input_value0 * kernel_size1 * kernel_size1 +
        (-64) * kernel_size1 * kernel_index_y +
        (-4) * kernel_size1 * input_value0 +
        16 * kernel_index_y * kernel_size1 * kernel_size1
    )

    tl.store(output_ptr0 + output_index, result, mask)