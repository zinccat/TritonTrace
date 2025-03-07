# From: 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_gelu_gelu_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_tanh_tanh_backward_0poi_fused_add_gelu_gelu_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_tanh_tanh_backward_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    index2 = index
    index0 = index % 512

    input_val0 = tl.load(input_ptr0 + index2, mask)
    input_val1 = tl.load(input_ptr1 + index0, mask, eviction_policy='evict_last')
    input_val2 = tl.load(input_ptr2 + index2, mask)

    add_result = input_val0 + input_val1
    sigmoid_result = tl.sigmoid(add_result)
    mul_result = sigmoid_result * add_result
    tanh_result = tl.extra.cuda.libdevice.tanh(mul_result)

    half = 0.5
    tanh_half = tanh_result * half
    sqrt_half = 0.7071067811865476
    tanh_sqrt_half = tanh_result * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(tanh_sqrt_half)
    one = 1.0
    erf_plus_one = erf_result + one
    gelu_result = tanh_half * erf_plus_one

    negative_one = -1.0
    gelu_leq_neg_one = gelu_result <= negative_one
    gelu_geq_one = gelu_result >= one
    gelu_out_of_bounds = gelu_leq_neg_one | gelu_geq_one

    zero = 0.0
    gelu_clipped = tl.where(gelu_out_of_bounds, zero, input_val2)

    erf_plus_one_half = erf_plus_one * half
    tanh_squared = tanh_result * tanh_result
    neg_half = -0.5
    exp_component = tanh_squared * neg_half
    exp_result = tl.math.exp(exp_component)
    sqrt_two_pi_inv = 0.3989422804014327
    exp_scaled = exp_result * sqrt_two_pi_inv
    tanh_exp_scaled = tanh_result * exp_scaled
    derivative = erf_plus_one_half + tanh_exp_scaled

    output_val0 = gelu_clipped * derivative
    one_minus_tanh_squared = one - tanh_squared
    output_val1_component = output_val0 * one_minus_tanh_squared
    one_minus_sigmoid = one - sigmoid_result
    sigmoid_one_minus_sigmoid = sigmoid_result * one_minus_sigmoid
    output_val1_component2 = output_val1_component * sigmoid_one_minus_sigmoid
    output_val1 = output_val1_component2 + (output_val1_component * add_result)

    tl.store(output_ptr0 + index2, output_val0, mask)
    tl.store(output_ptr1 + index2, output_val1, mask)