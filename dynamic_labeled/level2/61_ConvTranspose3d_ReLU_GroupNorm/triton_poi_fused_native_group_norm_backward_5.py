# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, kernel_size0, kernel_size1, kernel_size2, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // kernel_size0

    # Load inputs
    grad_output = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_last')
    input_data = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    mean = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    inv_std = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')

    # Compute intermediate values
    mean_scaled = input_data * inv_std
    mean_variance_diff = mean_scaled - variance
    variance_scaled = mean_variance_diff * inv_std
    variance_scaled_cubed = variance_scaled * variance_scaled * inv_std

    # Compute normalization factor
    factor_2 = 2.0
    kernel_size1_float = kernel_size1.to(tl.float32)
    normalization_base = factor_2 + kernel_size1_float
    normalization_power = tl.extra.cuda.libdevice.pow(normalization_base, factor_2)
    factor_16 = 16.0
    normalization_factor = factor_16 * normalization_power

    kernel_size2_float = kernel_size2.to(tl.float32)
    normalization_base_2 = factor_2 + kernel_size2_float
    normalization_factor *= normalization_base_2

    normalization_factor_double = normalization_factor.to(tl.float64)
    one_double = tl.full([1], 1.0, tl.float64)
    normalization_factor_inv = one_double / normalization_factor_double
    normalization_factor_inv_float = normalization_factor_inv.to(tl.float32)

    # Compute gradient
    grad_input_scaled = variance_scaled_cubed * normalization_factor_inv_float
    grad_input_scaled_neg = -grad_input_scaled
    grad_input_scaled_neg_mean = grad_input_scaled_neg * inv_std
    input_data_scaled = input_data * inv_std
    input_data_scaled_norm = input_data_scaled * normalization_factor_inv_float
    grad_input = grad_input_scaled_neg_mean - input_data_scaled_norm

    # Update gradient output
    updated_grad_output = grad_output + grad_input
    tl.store(in_out_ptr0 + (x2), updated_grad_output, xmask)