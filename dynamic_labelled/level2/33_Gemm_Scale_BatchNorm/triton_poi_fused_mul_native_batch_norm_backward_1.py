# From: 33_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_native_batch_norm_backward_1poi_fused_mul_native_batch_norm_backward_1(
    input_grad_ptr, input_data_ptr, scale_ptr, bias_ptr, running_var_ptr, running_mean_ptr, weight_ptr, save_mean_ptr, 
    output_grad_ptr, output_data_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < num_elements
    x2 = xindex
    x0 = (xindex % 512)
    
    grad_input = tl.load(input_grad_ptr + (x2), xmask)
    input_data = tl.load(input_data_ptr + (x2), xmask)
    scale = tl.load(scale_ptr + (x0), xmask, eviction_policy='evict_last')
    bias = tl.load(bias_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    weight = tl.load(weight_ptr + (x0), xmask, eviction_policy='evict_last')
    save_mean = tl.load(save_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    
    scaled_input = input_data * scale
    centered_input = scaled_input - bias
    inv_std = (tl.full([], 1.00000000000000, tl.float64) / ((512 * kernel_size) / 512))
    inv_std_float32 = inv_std.to(tl.float32)
    weight_scaled = weight * inv_std_float32
    var_scaled = running_var * inv_std_float32
    var_scaled_squared = var_scaled * var_scaled
    normalized_grad = centered_input * var_scaled_squared
    grad_input_centered = grad_input - normalized_grad
    mean_grad = running_mean * inv_std_float32
    grad_input_centered_mean = grad_input_centered - mean_grad
    save_mean_scaled = running_var * save_mean
    grad_input_centered_mean_scaled = grad_input_centered_mean * save_mean_scaled
    output_grad = grad_input_centered_mean_scaled * scale
    
    tl.store(output_grad_ptr + (x2), grad_input_centered_mean_scaled, xmask)
    tl.store(output_data_ptr + (x2), output_grad, xmask)