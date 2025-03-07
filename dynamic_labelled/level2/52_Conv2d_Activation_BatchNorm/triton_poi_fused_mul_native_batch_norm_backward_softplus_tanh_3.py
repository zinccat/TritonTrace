# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_native_batch_norm_backward_softplus_tanh_3(
    input_grad_ptr, input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, saved_mean_ptr, output_grad_ptr,
    kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size_0) % 16)
    
    input_grad = tl.load(input_grad_ptr + (x3), x_mask, eviction_policy='evict_last')
    input = tl.load(input_ptr + (x3), x_mask, eviction_policy='evict_last')
    running_mean = tl.load(running_mean_ptr + (x1), x_mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x1), x_mask, eviction_policy='evict_last')
    weight = tl.load(weight_ptr + (x1), x_mask, eviction_policy='evict_last')
    bias = tl.load(bias_ptr + (x1), x_mask, eviction_policy='evict_last')
    saved_mean = tl.load(saved_mean_ptr + (x1), x_mask, eviction_policy='evict_last')
    
    threshold = 20.0
    is_greater_than_threshold = input > threshold
    exp_input = tl.math.exp(input)
    log1p_exp_input = tl.extra.cuda.libdevice.log1p(exp_input)
    softplus_input = tl.where(is_greater_than_threshold, input, log1p_exp_input)
    tanh_softplus = tl.extra.cuda.libdevice.tanh(softplus_input)
    grad_activation = tanh_softplus * input
    
    delta = grad_activation - running_mean
    inv_std = (tl.full([], 1.0, tl.float64) / ((64 * kernel_size_1 + ((-64) * kernel_size_1 * kernel_size_2) + 16 * kernel_size_1 * kernel_size_2 * kernel_size_2) / 16))
    inv_std_float32 = inv_std.to(tl.float32)
    weight_scaled = weight * inv_std_float32
    var_scaled = running_var * running_var
    weight_var_scaled = weight_scaled * var_scaled
    delta_weight_var_scaled = delta * weight_var_scaled
    normalized_input_grad = input_grad - delta_weight_var_scaled
    bias_scaled = bias * inv_std_float32
    normalized_input_grad_bias = normalized_input_grad - bias_scaled
    saved_mean_scaled = running_var * saved_mean
    output_grad = normalized_input_grad_bias * saved_mean_scaled
    
    tl.store(output_grad_ptr + (x3), output_grad, x_mask)