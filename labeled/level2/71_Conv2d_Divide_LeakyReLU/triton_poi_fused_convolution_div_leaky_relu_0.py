# From: 71_Conv2d_Divide_LeakyReLU

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_div_leaky_relu_0(input_ptr0, input_ptr1, output_ptr0, output_ptr1, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    linear_index = index
    channel_index = (index // 900) % 16
    batch_index = (index // 14400)
    output_index = index % 14400

    input_value0 = tl.load(input_ptr0 + (linear_index), None)
    input_value1 = tl.load(input_ptr1 + (channel_index), None, eviction_policy='evict_last')
    sum_values = input_value0 + input_value1

    half = 0.5
    scaled_sum = sum_values * half

    zero = 0.0
    is_positive = scaled_sum > zero

    leaky_slope = 0.01
    leaky_value = scaled_sum * leaky_slope

    output_value = tl.where(is_positive, scaled_sum, leaky_value)

    tl.store(output_ptr0 + (output_index + (14464 * batch_index)), is_positive, None)
    tl.store(output_ptr1 + (linear_index), output_value, None)