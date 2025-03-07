# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_7poi_fused__native_batch_norm_legit_functional_relu_7(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    input_indices = block_indices
    channel_indices = (block_indices // 3136) % 64
    
    input_data = tl.load(input_ptr_input + (input_indices), None)
    mean_data = tl.load(input_ptr_mean + (channel_indices), None, eviction_policy='evict_last')
    var_data = tl.load(input_ptr_var + (channel_indices), None, eviction_policy='evict_last')
    scale_data = tl.load(input_ptr_scale + (channel_indices), None, eviction_policy='evict_last')
    shift_data = tl.load(input_ptr_shift + (channel_indices), None, eviction_policy='evict_last')
    
    normalized_data = input_data - mean_data
    variance_scale = 31360.0
    epsilon = 1e-05
    
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(var_data / variance_scale + epsilon)
    scaled_data = normalized_data * inv_std_dev * scale_data
    shifted_data = scaled_data + shift_data
    
    relu_output = tl.full([1], 0, tl.int32)
    relu_applied = triton_helpers.maximum(relu_output, shifted_data)
    
    tl.store(output_ptr + (input_indices), relu_applied, None)