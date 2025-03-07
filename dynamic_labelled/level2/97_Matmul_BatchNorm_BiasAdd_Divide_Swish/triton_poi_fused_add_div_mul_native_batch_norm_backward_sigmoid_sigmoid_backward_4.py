# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_4poi_fused_add_div_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_4(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    
    input_data = tl.load(in_ptr0 + (x2), xmask)
    grad_output = tl.load(in_out_ptr0 + (x2), xmask)
    mean = tl.load(in_ptr1 + (x2), xmask)
    variance = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    scale = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    running_mean = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    epsilon = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    
    sigmoid_grad_output = tl.sigmoid(grad_output)
    grad_input = input_data * sigmoid_grad_output
    grad_input_scaled = input_data * grad_output
    one = 1.0
    one_minus_sigmoid = one - sigmoid_grad_output
    grad_sigmoid = sigmoid_grad_output * one_minus_sigmoid
    grad_input_combined = grad_input + (grad_input_scaled * grad_sigmoid)
    grad_input_final = grad_input_combined * one
    
    mean_diff = mean - variance
    normalization_factor = (tl.full([], 1.00000000000000, tl.float64) / ((512 * ks0) / 512)).to(tl.float32)
    scale_normalized = scale * normalization_factor
    variance_squared = running_var * running_var
    scale_variance = scale_normalized * variance_squared
    mean_variance = mean_diff * scale_variance
    grad_input_adjusted = grad_input_final - mean_variance
    
    running_var_normalized = running_var * normalization_factor
    grad_input_adjusted_final = grad_input_adjusted - running_var_normalized
    epsilon_scaled = running_var * epsilon
    grad_input_final_result = grad_input_adjusted_final * epsilon_scaled
    
    tl.store(in_out_ptr0 + (x2), grad_input_final_result, xmask)