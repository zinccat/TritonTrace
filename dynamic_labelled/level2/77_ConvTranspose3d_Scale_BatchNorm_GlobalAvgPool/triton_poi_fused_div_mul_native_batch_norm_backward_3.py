# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_mul_native_batch_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, 
    kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices
    batch_index = xindex // kernel_size0
    element_index = xindex
    channel_index = ((xindex // kernel_size1) % 32)

    # Load data
    input_data = tl.load(in_ptr0 + (batch_index), xmask, eviction_policy='evict_last')
    grad_output = tl.load(in_out_ptr0 + (element_index), xmask, eviction_policy='evict_last')
    running_mean = tl.load(in_ptr1 + (channel_index), xmask, eviction_policy='evict_last')
    running_var = tl.load(in_ptr2 + (channel_index), xmask, eviction_policy='evict_last')
    weight = tl.load(in_ptr3 + (channel_index), xmask, eviction_policy='evict_last')
    bias = tl.load(in_ptr4 + (channel_index), xmask, eviction_policy='evict_last')
    grad_input = tl.load(in_ptr5 + (channel_index), xmask, eviction_policy='evict_last')

    # Compute normalization
    input_normalized = input_data / kernel_size0.to(tl.float32)
    grad_output_scaled = grad_output * 2.0
    delta = grad_output_scaled - running_mean

    # Compute variance scaling factor
    variance_scale = (
        tl.full([], 1.0, tl.float64) / 
        ((128 * kernel_size2 * kernel_size2 + 256 * kernel_size2 + 
          32 * kernel_size2 * kernel_size2 * kernel_size3 * kernel_size3 + 
          64 * kernel_size2 * kernel_size3 * kernel_size3 + 
          128 * kernel_size3 * kernel_size2 * kernel_size2 + 
          256 * kernel_size2 * kernel_size3) / 32)
    ).to(tl.float32)

    # Compute gradient
    inv_std = variance_scale * (running_var * running_var)
    grad_normalized = delta * inv_std
    grad_input_adjusted = input_normalized - grad_normalized - bias * variance_scale
    grad_weight = grad_input_adjusted * grad_input

    # Store result
    grad_weight_scaled = grad_weight * weight * 2.0
    tl.store(in_out_ptr0 + (element_index), grad_weight_scaled, xmask)