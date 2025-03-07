# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_2poi_fused__native_batch_norm_legit_functional_relu_2(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    input_index = block_indices
    channel_index = (block_indices // 50176) % 32
    
    input_value = tl.load(input_ptr_input + (input_index), None)
    mean_value = tl.load(input_ptr_mean + (channel_index), None, eviction_policy='evict_last')
    var_value = tl.load(input_ptr_var + (channel_index), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (channel_index), None, eviction_policy='evict_last')
    shift_value = tl.load(input_ptr_shift + (channel_index), None, eviction_policy='evict_last')
    
    normalized_value = input_value - mean_value
    variance_scale = 501760.0
    epsilon = 1e-05
    
    inv_stddev = tl.extra.cuda.libdevice.rsqrt((var_value / variance_scale) + epsilon)
    scaled_value = normalized_value * inv_stddev
    scaled_and_shifted_value = scaled_value * scale_value + shift_value
    
    relu_min = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(relu_min, scaled_and_shifted_value)
    
    tl.store(output_ptr + (input_index), relu_output, None)