# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_2(input_ptr_mean, input_ptr_var, input_ptr_scale, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    element_indices = indices
    group_indices = indices // kernel_size
    
    mean_values = tl.load(input_ptr_mean + (element_indices), mask, eviction_policy='evict_last')
    variance_values = tl.load(input_ptr_var + (group_indices), mask, eviction_policy='evict_last')
    scale_values = tl.load(input_ptr_scale + (group_indices), mask, eviction_policy='evict_last')
    
    centered_values = mean_values - scale_values
    kernel_size_float = kernel_size.to(tl.float32)
    normalized_variance = variance_values / kernel_size_float
    
    epsilon = 1e-05
    variance_with_epsilon = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    
    normalized_output = centered_values * inv_sqrt_variance
    tl.store(output_ptr + (element_indices), normalized_output, mask)