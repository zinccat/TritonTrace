# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_clamp_mul_3poi_fused__softmax_clamp_mul_3(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index0 = index % kernel_size0
    kernel_index2 = index // kernel_size1

    input_value0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (kernel_index0 + kernel_size2 * kernel_size3 * kernel_index2), mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (kernel_index0 + kernel_size2 * kernel_size3 * kernel_index2), mask, eviction_policy='evict_last')

    clamp_min = 0.0
    max_value = triton_helpers.maximum(input_value0, clamp_min)
    clamp_max = 1.0
    clamped_value = triton_helpers.minimum(max_value, clamp_max)

    exp_input = clamped_value - input_value1
    exp_result = tl.math.exp(exp_input)
    softmax_result = exp_result / input_value2

    scaled_result = softmax_result * 2.0
    tl.store(output_ptr0 + (linear_index), scaled_result, mask)