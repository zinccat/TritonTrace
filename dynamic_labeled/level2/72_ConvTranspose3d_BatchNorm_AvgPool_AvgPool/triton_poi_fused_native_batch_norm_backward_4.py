# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_batch_norm_backward_4(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ks0, ks1, ks2, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // ks0) % 16)

    # Load inputs
    grad_output = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last')
    input_data = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    mean = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    inv_std = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    var = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    saved_mean = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    saved_var = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')

    # Compute intermediate values
    input_centered = input_data - mean
    normalization_factor = (
        tl.full([], 1.0, tl.float64) /
        ((((-16) * ks1) + ((-192) * ks1 * ks2 * ks2) + 96 * ks1 * ks2 + 128 * ks1 * ks2 * ks2 * ks2) / 16)
    )
    normalization_factor = normalization_factor.to(tl.float32)
    inv_std_scaled = inv_std * normalization_factor
    var_scaled = var * var
    inv_std_var_scaled = inv_std_scaled * var_scaled
    grad_input_centered = input_centered * inv_std_var_scaled
    grad_input = grad_output - grad_input_centered
    mean_grad = saved_mean * normalization_factor
    grad_input_mean = grad_input - mean_grad
    var_grad = grad_input_mean * saved_var

    # Store result
    tl.store(in_out_ptr0 + (x3), var_grad, xmask)