# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_batch_norm_backward_4poi_fused_native_batch_norm_backward_4(
    in_out_ptr, input_data, mean_ptr, variance_ptr, scale_ptr, bias_ptr, running_var_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < num_elements
    x2 = xindex
    x0 = (xindex % 1024)
    
    input_grad = tl.load(in_out_ptr + (x2), xmask)
    input_data_val = tl.load(input_data + (x2), xmask)
    mean_val = tl.load(mean_ptr + (x0), xmask, eviction_policy='evict_last')
    variance_val = tl.load(variance_ptr + (x0), xmask, eviction_policy='evict_last')
    scale_val = tl.load(scale_ptr + (x0), xmask, eviction_policy='evict_last')
    bias_val = tl.load(bias_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var_val = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')
    
    centered_input = input_data_val - mean_val
    normalization_factor = (tl.full([], 1.00000000000000, tl.float64) / ((1024 * kernel_size) / 1024))
    normalization_factor = normalization_factor.to(tl.float32)
    scaled_variance = variance_val * normalization_factor
    variance_sqrt = variance_val * variance_val
    inv_std_dev = scaled_variance * variance_sqrt
    normalized_grad = centered_input * inv_std_dev
    grad_input = input_grad - normalized_grad
    scaled_bias = bias_val * normalization_factor
    grad_input -= scaled_bias
    running_var_grad = variance_val * running_var_val
    final_grad = grad_input * running_var_grad
    
    tl.store(in_out_ptr + (x2), final_grad, xmask)