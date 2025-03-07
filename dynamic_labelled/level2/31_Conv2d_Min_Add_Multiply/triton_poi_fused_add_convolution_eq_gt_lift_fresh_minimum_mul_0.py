# From: 31_Conv2d_Min_Add_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_convolution_eq_gt_lift_fresh_minimum_mul_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, output_ptr2, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    global_index = index
    channel_index = (index // kernel_size) % 16

    input_value0 = tl.load(input_ptr0 + global_index, mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + channel_index, mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + channel_index, mask, eviction_policy='evict_last')

    sum_values = input_value0 + input_value1
    threshold = 0.5
    min_value = triton_helpers.minimum(sum_values, threshold)
    result_value = min_value + input_value2
    scale_factor = 2.0
    scaled_result = result_value * scale_factor

    is_equal = sum_values == threshold
    is_greater = sum_values > threshold

    tl.store(output_ptr0 + global_index, scaled_result, mask)
    tl.store(output_ptr1 + global_index, is_equal, mask)
    tl.store(output_ptr2 + global_index, is_greater, mask)