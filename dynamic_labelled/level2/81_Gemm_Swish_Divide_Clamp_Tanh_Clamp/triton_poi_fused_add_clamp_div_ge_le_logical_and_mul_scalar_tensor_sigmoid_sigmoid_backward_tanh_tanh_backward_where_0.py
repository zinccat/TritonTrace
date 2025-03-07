# From: 81_Gemm_Swish_Divide_Clamp_Tanh_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_div_ge_le_logical_and_mul_scalar_tensor_sigmoid_sigmoid_backward_tanh_tanh_backward_where_0poi_fused_add_clamp_div_ge_le_logical_and_mul_scalar_tensor_sigmoid_sigmoid_backward_tanh_tanh_backward_where_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    input_value = tl.load(in_out_ptr0 + (x0), xmask)
    input_tensor = tl.load(in_ptr0 + (x0), xmask)
    
    sigmoid_output = tl.sigmoid(input_value)
    sigmoid_derivative = input_value * sigmoid_output
    
    half = 0.5
    scaled_sigmoid = sigmoid_derivative * half
    
    lower_bound = -1.0
    upper_bound = 1.0
    
    is_ge_lower_bound = scaled_sigmoid >= lower_bound
    is_le_upper_bound = scaled_sigmoid <= upper_bound
    within_bounds = is_ge_lower_bound & is_le_upper_bound
    
    clamped_value = triton_helpers.maximum(scaled_sigmoid, lower_bound)
    clamped_value = triton_helpers.minimum(clamped_value, upper_bound)
    
    tanh_output = tl.extra.cuda.libdevice.tanh(clamped_value)
    
    tanh_ge_lower_bound = tanh_output >= lower_bound
    tanh_le_upper_bound = tanh_output <= upper_bound
    tanh_within_bounds = tanh_ge_lower_bound & tanh_le_upper_bound
    
    zero = 0.0
    selected_value = tl.where(tanh_within_bounds, input_tensor, zero)
    
    tanh_squared = tanh_output * tanh_output
    tanh_derivative = upper_bound - tanh_squared
    
    scaled_selected_value = selected_value * tanh_derivative
    final_value = tl.where(within_bounds, scaled_selected_value, zero)
    
    scaled_final_value = final_value * half
    scaled_sigmoid_output = scaled_final_value * sigmoid_output
    scaled_sigmoid_input = scaled_final_value * input_value
    
    one_minus_sigmoid = upper_bound - sigmoid_output
    sigmoid_derivative_product = sigmoid_derivative * one_minus_sigmoid
    
    scaled_sigmoid_derivative_product = scaled_sigmoid_input * sigmoid_derivative_product
    combined_result = scaled_sigmoid_output + scaled_sigmoid_derivative_product
    
    tl.store(in_out_ptr0 + (x0), combined_result, xmask)