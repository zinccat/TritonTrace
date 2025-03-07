# From: 33_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_native_batch_norm_backward_1(
    input_grad_ptr, scale_ptr, running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr, grad_output_ptr, save_mean_ptr, grad_input_ptr, grad_scale_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < num_elements
    x2 = xindex
    x0 = (xindex % 512)
    
    input_grad = tl.load(input_grad_ptr + (x2), xmask)
    scale = tl.load(scale_ptr + (x2), xmask)
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')
    bn_weight = tl.load(bn_weight_ptr + (x0), xmask, eviction_policy='evict_last')
    bn_bias = tl.load(bn_bias_ptr + (x0), xmask, eviction_policy='evict_last')
    grad_output = tl.load(grad_output_ptr + (x0), xmask, eviction_policy='evict_last')
    save_mean = tl.load(save_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    
    scale_running_mean = scale * running_mean
    mean_diff = scale_running_mean - running_var
    inv_std = (tl.full([], 1.00000000000000, tl.float64) / ((512 * kernel_size) / 512))
    inv_std_float32 = inv_std.to(tl.float32)
    std_weight = bn_weight * inv_std_float32
    var_weight = std_weight * (bn_bias * bn_bias)
    grad_input = input_grad - (mean_diff * var_weight)
    grad_bias = grad_output * inv_std_float32
    grad_input -= grad_bias
    save_mean_grad = grad_output * save_mean
    grad_input *= save_mean_grad
    grad_scale = grad_input * running_mean
    
    tl.store(grad_input_ptr + (x2), grad_input, xmask)
    tl.store(grad_scale_ptr + (x2), grad_scale, xmask)