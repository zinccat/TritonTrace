# From: 54_Conv2d_Multiply_LeakyReLU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_gelu_leaky_relu_mul_0(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    channel_index = (index // kernel_size) % 16

    # Load data with eviction policy
    in_out_data = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_data_0 = tl.load(in_ptr0 + (channel_index), mask, eviction_policy='evict_last')
    input_data_1 = tl.load(in_ptr1 + (channel_index), mask, eviction_policy='evict_last')

    # Perform operations
    add_result = in_out_data + input_data_0
    multiply_result = add_result * input_data_1

    # Leaky ReLU
    zero = 0.0
    greater_than_zero = multiply_result > zero
    leaky_relu_slope = 0.01
    leaky_relu_result = tl.where(greater_than_zero, multiply_result, multiply_result * leaky_relu_slope)

    # GELU
    half = 0.5
    sqrt_two_over_pi = 0.7071067811865476
    gelu_scaled_input = leaky_relu_result * sqrt_two_over_pi
    erf_result = tl.extra.cuda.libdevice.erf(gelu_scaled_input)
    one = 1.0
    gelu_result = half * (leaky_relu_result * (erf_result + one))

    # Store results
    tl.store(in_out_ptr0 + (linear_index), add_result, mask)
    tl.store(out_ptr0 + (linear_index), gelu_result, mask)