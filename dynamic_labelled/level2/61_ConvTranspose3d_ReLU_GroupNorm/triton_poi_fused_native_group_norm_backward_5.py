# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_5poi_fused_native_group_norm_backward_5(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, kernel_size0, kernel_size1, kernel_size2, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // kernel_size0

    grad_output = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_last')
    input_data = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    running_mean = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    running_var = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    weight = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')

    mean_weight_product = input_data * running_mean
    variance_diff = mean_weight_product - running_var
    normalized_grad = variance_diff * weight
    grad_weight = normalized_grad * weight
    grad_weight_cubed = grad_weight * weight
    grad_weight_cubed = grad_weight_cubed * weight

    epsilon = 2.0
    kernel_size1_float = kernel_size1.to(tl.float32)
    normalization_factor = epsilon + kernel_size1_float
    normalization_factor_power = tl.extra.cuda.libdevice.pow(normalization_factor, epsilon)

    scale_factor = 16.0
    scale_with_power = scale_factor * normalization_factor_power
    kernel_size2_float = kernel_size2.to(tl.float32)
    normalization_factor_with_epsilon = epsilon + kernel_size2_float
    final_scale = scale_with_power * normalization_factor_with_epsilon

    final_scale_double = final_scale.to(tl.float64)
    one_double = tl.full([1], 1.0, tl.float64)
    inv_final_scale = one_double / final_scale_double
    inv_final_scale_float = inv_final_scale.to(tl.float32)

    grad_input = grad_weight_cubed * inv_final_scale_float
    neg_grad_input = -grad_input
    grad_input_weighted = neg_grad_input * running_mean
    input_weighted = input_data * weight
    input_weighted_scaled = input_weighted * inv_final_scale_float
    grad_input_adjusted = grad_input_weighted - input_weighted_scaled

    updated_grad_output = grad_output + grad_input_adjusted
    tl.store(in_out_ptr0 + (x2), updated_grad_output, xmask)