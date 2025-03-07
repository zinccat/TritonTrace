# From: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_ge_gelu_gelu_backward_le_leaky_relu_logical_and_scalar_tensor_where_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size) % 16)
    
    input_value0 = tl.load(input_ptr0 + (x3), x_mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (x1), x_mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (x3), x_mask, eviction_policy='evict_last')
    
    zero = 0.0
    leaky_relu_slope = 0.2
    clamp_min = -1.0
    clamp_max = 1.0
    erf_coefficient = 0.7071067811865476
    erf_offset = 1.0
    erf_half = 0.5
    exp_coefficient = -0.5
    exp_base_coefficient = 0.3989422804014327
    
    leaky_relu = tl.where(input_value0 > zero, input_value0, input_value0 * leaky_relu_slope)
    add_result = leaky_relu + input_value1
    
    clamp_condition = (add_result >= clamp_min) & (add_result <= clamp_max)
    clamped_value = triton_helpers.maximum(add_result, clamp_min)
    clamped_value = triton_helpers.minimum(clamped_value, clamp_max)
    
    erf_input = clamped_value * erf_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    erf_result = erf_result + erf_offset
    erf_result = erf_result * erf_half
    
    squared_clamped = clamped_value * clamped_value
    exp_result = tl.math.exp(squared_clamped * exp_coefficient)
    exp_result = exp_result * exp_base_coefficient
    
    gelu_result = erf_result + (clamped_value * exp_result)
    final_result = input_value2 * gelu_result
    
    output_value = tl.where(clamp_condition, final_result, zero)
    tl.store(output_ptr0 + (x3), output_value, x_mask)