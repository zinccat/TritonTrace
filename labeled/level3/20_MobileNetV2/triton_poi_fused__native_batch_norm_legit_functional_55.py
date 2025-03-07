# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_55poi_fused__native_batch_norm_legit_functional_55(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 78400
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = indices % 160
    input_data = tl.load(input_ptr_input + (linear_index), mask)
    mean_data = tl.load(input_ptr_mean + (channel_index), mask, eviction_policy='evict_last')
    var_data = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale_data = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    bias_data = tl.load(input_ptr_bias + (channel_index), mask, eviction_policy='evict_last')
    
    normalized_data = input_data - mean_data
    variance_epsilon = 1e-05
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(var_data / 490.0 + variance_epsilon)
    scaled_data = normalized_data * inv_std_dev * scale_data
    output_data = scaled_data + bias_data
    
    tl.store(output_ptr + (linear_index), output_data, mask)