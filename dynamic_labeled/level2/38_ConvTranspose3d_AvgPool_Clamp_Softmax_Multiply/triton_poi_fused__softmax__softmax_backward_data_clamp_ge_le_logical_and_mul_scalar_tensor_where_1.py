# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_clamp_ge_le_logical_and_mul_scalar_tensor_where_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ks0, ks1, ks2, ks3, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % ks0)
    x2 = xindex // ks1

    # Load data from pointers
    current_value = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last')
    max_value = tl.load(in_ptr0 + (x0 + ks2 * x2 * ks3 * ks3), xmask, eviction_policy='evict_last')
    sum_exp = tl.load(in_ptr1 + (x0 + ks2 * x2 * ks3 * ks3), xmask, eviction_policy='evict_last')
    grad_output = tl.load(in_ptr2 + (x0 + ks2 * x2 * ks3 * ks3), xmask, eviction_policy='evict_last')
    scale_factor = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')

    # Initialize temporary variables
    lower_bound = 0.0
    upper_bound = 1.0

    # Clamp current value between lower and upper bounds
    is_ge_lower = current_value >= lower_bound
    is_le_upper = current_value <= upper_bound
    is_within_bounds = is_ge_lower & is_le_upper

    # Apply clamping
    clamped_value = triton_helpers.maximum(current_value, lower_bound)
    clamped_value = triton_helpers.minimum(clamped_value, upper_bound)

    # Compute softmax backward pass
    adjusted_value = clamped_value - max_value
    exp_value = tl.math.exp(adjusted_value)
    softmax_grad = exp_value / sum_exp
    neg_softmax_grad = -softmax_grad

    # Compute final gradient
    scale_multiplier = 2.0
    scaled_grad = scale_factor * scale_multiplier * softmax_grad
    final_grad = tl.extra.cuda.libdevice.fma(neg_softmax_grad, grad_output, scaled_grad)

    # Store the result
    result_value = tl.where(is_within_bounds, final_grad, lower_bound)
    tl.store(in_out_ptr0 + (x3), result_value, xmask)