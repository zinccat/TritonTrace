# From: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_convolution_gelu_leaky_relu_0poi_fused_add_clamp_convolution_gelu_leaky_relu_0(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size) % 16)
    
    input_output_value = tl.load(in_out_ptr0 + (x3), x_mask, eviction_policy='evict_last')
    input_value_0 = tl.load(in_ptr0 + (x1), x_mask, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr1 + (x1), x_mask, eviction_policy='evict_last')
    
    added_value = input_output_value + input_value_0
    zero = 0.0
    is_positive = added_value > zero
    leaky_relu_slope = 0.2
    leaky_relu_value = added_value * leaky_relu_slope
    leaky_relu_result = tl.where(is_positive, added_value, leaky_relu_value)
    
    sum_result = leaky_relu_result + input_value_1
    clamp_min = -1.0
    clamp_max = 1.0
    
    clamped_value = triton_helpers.maximum(sum_result, clamp_min)
    clamped_value = triton_helpers.minimum(clamped_value, clamp_max)
    
    gelu_coefficient = 0.5
    gelu_coefficient_sqrt2 = 0.7071067811865476
    
    scaled_clamped_value = clamped_value * gelu_coefficient
    scaled_sqrt2_clamped_value = clamped_value * gelu_coefficient_sqrt2
    
    erf_result = tl.extra.cuda.libdevice.erf(scaled_sqrt2_clamped_value)
    gelu_result = scaled_clamped_value * (erf_result + clamp_max)
    
    tl.store(in_out_ptr0 + (x3), added_value, x_mask)
    tl.store(out_ptr0 + (x3), gelu_result, x_mask)