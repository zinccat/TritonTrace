# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Load input data
    input_data = tl.load(in_ptr0 + (xindex), None)
    scale_factor = tl.load(in_ptr1 + (xindex // 16384), None, eviction_policy='evict_last')
    shift_factor = tl.load(in_ptr2 + ((xindex // 1024) % 64), None, eviction_policy='evict_last')
    grad_output = tl.load(in_out_ptr0 + (xindex), None)
    mean = tl.load(in_ptr3 + (xindex // 16384), None, eviction_policy='evict_last')
    variance = tl.load(in_ptr4 + (xindex // 16384), None, eviction_policy='evict_last')
    saved_variance = tl.load(in_ptr5 + (xindex // 16384), None, eviction_policy='evict_last')
    
    # Compute intermediate values
    scale_shift_product = scale_factor * shift_factor
    normalized_input = input_data * scale_shift_product
    mean_variance_product = mean * variance
    variance_diff = mean_variance_product - saved_variance
    variance_diff_scaled = variance_diff * scale_factor
    variance_diff_scaled_cubed = variance_diff_scaled * variance_diff_scaled * variance_diff_scaled
    epsilon = 6.103515625e-05
    variance_diff_scaled_cubed_epsilon = variance_diff_scaled_cubed * epsilon
    grad_input = grad_output * variance_diff_scaled_cubed_epsilon
    grad_input_adjusted = normalized_input + grad_input
    
    # Compute final gradient
    shift_factor_scaled = -epsilon * shift_factor
    mean_scaled = mean * scale_factor
    mean_scaled_epsilon = mean_scaled * epsilon
    shift_factor_adjusted = shift_factor_scaled - mean_scaled_epsilon
    final_grad = grad_input_adjusted + shift_factor_adjusted
    
    # Store the result
    tl.store(in_out_ptr0 + (xindex), final_grad, None)