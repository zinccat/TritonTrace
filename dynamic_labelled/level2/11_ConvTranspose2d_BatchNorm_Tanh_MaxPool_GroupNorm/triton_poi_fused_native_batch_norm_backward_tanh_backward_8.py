# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_batch_norm_backward_tanh_backward_8(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, 
    kernel_size, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    
    # Load inputs
    grad_output = tl.load(in_out_ptr0 + (x3), None)
    input_data = tl.load(in_ptr0 + (x3), None)
    grad_input = tl.load(in_ptr1 + (x3), None)
    running_mean = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    running_var = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    weight = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    bias = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    saved_mean = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    
    # Compute intermediate values
    input_squared = input_data * input_data
    epsilon = 1.0
    variance_term = epsilon - input_squared
    normalized_grad_output = grad_output * variance_term
    
    mean_diff = grad_input - running_mean
    normalization_factor = (tl.full([], 1.00000000000000, tl.float64) / ((262144 * kernel_size) / 64))
    normalization_factor_float32 = normalization_factor.to(tl.float32)
    normalized_weight = weight * normalization_factor_float32
    weight_squared = weight * weight
    weight_variance_term = normalized_weight * weight_squared
    mean_diff_weighted = mean_diff * weight_variance_term
    grad_input_normalized = normalized_grad_output - mean_diff_weighted
    
    normalized_bias = bias * normalization_factor_float32
    grad_input_normalized_bias = grad_input_normalized - normalized_bias
    saved_mean_weighted = weight * saved_mean
    final_grad_input = grad_input_normalized_bias * saved_mean_weighted
    
    # Store result
    tl.store(in_out_ptr0 + (x3), final_grad_input, None)