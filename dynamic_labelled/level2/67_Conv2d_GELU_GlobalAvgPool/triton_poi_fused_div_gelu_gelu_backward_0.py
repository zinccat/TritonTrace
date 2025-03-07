# From: 67_Conv2d_GELU_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_gelu_gelu_backward_0poi_fused_div_gelu_gelu_backward_0(in_out_ptr0, in_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    row_index = index // kernel_size0
    col_index = index
    input_value = tl.load(in_ptr0 + (row_index), mask, eviction_policy='evict_last')
    grad_output = tl.load(in_out_ptr0 + (col_index), mask, eviction_policy='evict_last')
    divisor = 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1)
    divisor_float = divisor.to(tl.float32)
    scaled_input = input_value / divisor_float
    sqrt_half = 0.7071067811865476
    scaled_grad_output = grad_output * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(scaled_grad_output)
    one = 1.0
    erf_plus_one = erf_result + one
    half = 0.5
    erf_half = erf_plus_one * half
    grad_output_squared = grad_output * grad_output
    neg_half = -0.5
    exp_argument = grad_output_squared * neg_half
    exp_result = tl.math.exp(exp_argument)
    sqrt_two_pi = 0.3989422804014327
    gaussian = exp_result * sqrt_two_pi
    gaussian_scaled = grad_output * gaussian
    gelu_grad = erf_half + gaussian_scaled
    final_result = scaled_input * gelu_grad
    tl.store(in_out_ptr0 + (col_index), final_result, mask)