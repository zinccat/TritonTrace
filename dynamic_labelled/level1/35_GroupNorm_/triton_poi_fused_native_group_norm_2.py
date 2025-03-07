# From: 35_GroupNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    group_index = index // kernel_size0
    channel_index = (index // kernel_size0) % 64
    spatial_index = index % kernel_size1
    batch_index = index // kernel_size1

    input_data = tl.load(in_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    mean_data = tl.load(in_ptr1 + (group_index // 8), mask, eviction_policy='evict_last')
    variance_data = tl.load(in_ptr2 + (group_index // 8), mask, eviction_policy='evict_last')
    gamma_data = tl.load(in_ptr3 + (channel_index), mask, eviction_policy='evict_last')
    beta_data = tl.load(in_ptr4 + (channel_index), mask, eviction_policy='evict_last')

    normalized_data = input_data - mean_data
    variance_scale = 8 * kernel_size0
    variance_scale_float = variance_scale.to(tl.float32)
    variance_adjusted = variance_data / variance_scale_float
    epsilon = 1e-05
    variance_stabilized = variance_adjusted + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    scaled_data = normalized_data * inv_stddev
    gamma_scaled_data = scaled_data * gamma_data
    output_data = gamma_scaled_data + beta_data

    tl.store(out_ptr0 + (spatial_index + batch_index * (kernel_size0 // kernel_size1)), output_data, mask)