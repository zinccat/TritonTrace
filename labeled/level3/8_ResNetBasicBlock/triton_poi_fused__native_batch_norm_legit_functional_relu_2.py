# From: 8_ResNetBasicBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_2poi_fused__native_batch_norm_legit_functional_relu_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    channel_index = (block_indices // 50176) % 64
    
    input_value = tl.load(input_ptr0 + (index), None)
    mean_value = tl.load(input_ptr1 + (channel_index), None, eviction_policy='evict_last')
    variance_value = tl.load(input_ptr2 + (channel_index), None, eviction_policy='evict_last')
    gamma_value = tl.load(input_ptr3 + (channel_index), None, eviction_policy='evict_last')
    beta_value = tl.load(input_ptr4 + (channel_index), None, eviction_policy='evict_last')
    
    normalized_value = input_value - mean_value
    variance_scale = 501760.0
    epsilon = 1e-05
    
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_value / variance_scale + epsilon)
    scaled_value = normalized_value * inv_stddev
    gamma_scaled_value = scaled_value * gamma_value
    output_value = gamma_scaled_value + beta_value
    
    relu_min = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(relu_min, output_value)
    
    tl.store(output_ptr0 + (index), relu_output, None)