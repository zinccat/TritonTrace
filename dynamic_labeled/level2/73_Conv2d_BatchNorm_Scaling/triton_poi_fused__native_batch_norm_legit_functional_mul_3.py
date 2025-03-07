# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_mul_3(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_out, 
    output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, 
    XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size_0) % 16)
    
    mean = tl.load(input_ptr_mean + (x3), x_mask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (x1), x_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (x1), x_mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (x1), x_mask, eviction_policy='evict_last')
    output = tl.load(input_ptr_out + (x1), x_mask, eviction_policy='evict_last')
    
    normalized_input = mean - variance
    variance_adjustment = 4 * kernel_size_1 + kernel_size_1 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_1 * kernel_size_2)
    variance_adjustment_float = variance_adjustment.to(tl.float32)
    variance_normalized = variance / variance_adjustment_float
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    scaled_input = normalized_input * inv_sqrt_variance
    scaled_and_shifted = scaled_input * scale
    final_output = scaled_and_shifted + shift
    output_scaled = final_output * 2.0
    
    tl.store(output_ptr + (x3), output_scaled, x_mask)