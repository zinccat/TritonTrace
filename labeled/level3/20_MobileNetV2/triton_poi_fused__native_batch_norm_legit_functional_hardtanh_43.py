# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_43poi_fused__native_batch_norm_legit_functional_hardtanh_43(input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 752640
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    channel_indices = block_indices % 384
    
    input_data = tl.load(input_ptr0 + (global_indices), valid_mask)
    mean_data = tl.load(input_ptr1 + (channel_indices), valid_mask, eviction_policy='evict_last')
    variance_data = tl.load(input_ptr2 + (channel_indices), valid_mask, eviction_policy='evict_last')
    gamma_data = tl.load(input_ptr3 + (channel_indices), valid_mask, eviction_policy='evict_last')
    beta_data = tl.load(input_ptr4 + (channel_indices), valid_mask, eviction_policy='evict_last')
    
    normalized_data = input_data - mean_data
    scaled_data = normalized_data * variance_data
    scaled_gamma = scaled_data * gamma_data
    shifted_data = scaled_gamma + beta_data
    
    clamped_min = 0.0
    clamped_max = 6.0
    clamped_data = triton_helpers.minimum(triton_helpers.maximum(shifted_data, clamped_min), clamped_max)
    
    tl.store(output_ptr0 + (global_indices), clamped_data, valid_mask)