# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_13poi_fused__native_batch_norm_legit_functional_hardtanh_13(input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    channel_indices = block_indices % 96
    input_value = tl.load(input_ptr0 + (global_indices), None)
    mean_value = tl.load(input_ptr1 + (channel_indices), None, eviction_policy='evict_last')
    variance_value = tl.load(input_ptr2 + (channel_indices), None, eviction_policy='evict_last')
    gamma_value = tl.load(input_ptr3 + (channel_indices), None, eviction_policy='evict_last')
    beta_value = tl.load(input_ptr4 + (channel_indices), None, eviction_policy='evict_last')
    
    normalized_value = (input_value - mean_value) * variance_value
    scaled_value = normalized_value * gamma_value
    shifted_value = scaled_value + beta_value
    
    clamped_value = triton_helpers.maximum(shifted_value, 0.0)
    output_value = triton_helpers.minimum(clamped_value, 6.0)
    
    tl.store(output_ptr0 + (global_indices), output_value, None)