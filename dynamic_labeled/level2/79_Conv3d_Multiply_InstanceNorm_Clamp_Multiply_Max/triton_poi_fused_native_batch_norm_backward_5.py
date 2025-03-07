# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_batch_norm_backward_5(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, 
    kernel_size0, kernel_size1, kernel_size2, kernel_size3, kernel_size4, 
    xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // kernel_size1

    # Load data with eviction policy
    grad_output = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_last')
    input_data = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    scale_factor = tl.load(in_ptr1 + (((x2 // kernel_size0) % 16)), xmask, eviction_policy='evict_last')
    mean = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    variance = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    running_mean = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    running_var = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')

    # Compute intermediate values
    scaled_input = input_data * scale_factor
    centered_input = scaled_input - mean
    normalization_factor = (
        tl.full([], 1.00000000000000, tl.float64) / 
        ((((-128) * kernel_size2) + ((-32) * kernel_size2 * kernel_size4 * kernel_size4) + 
         64 * kernel_size2 * kernel_size3 + 128 * kernel_size2 * kernel_size4 + 
         ((-64) * kernel_size2 * kernel_size3 * kernel_size4) + 
         16 * kernel_size2 * kernel_size3 * kernel_size4 * kernel_size4) / 
         (16 * kernel_size2))
    )
    normalization_factor = normalization_factor.to(tl.float32)
    inv_std_dev = variance * normalization_factor
    std_dev_squared = variance * variance
    grad_input = centered_input * (inv_std_dev * std_dev_squared)
    grad_scale = grad_output - grad_input
    grad_bias = running_var * normalization_factor
    grad_output_adjusted = grad_output - grad_bias

    # Store the result
    grad_input_adjusted = grad_output_adjusted * input_data
    tl.store(in_out_ptr0 + (x2), grad_input_adjusted, xmask)