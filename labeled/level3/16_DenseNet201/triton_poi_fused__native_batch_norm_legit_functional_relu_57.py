# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_57poi_fused__native_batch_norm_legit_functional_relu_57(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    input_index = block_indices
    batch_index = (block_indices // 784) % 384
    
    input_value = tl.load(input_ptr_input + (input_index), None)
    mean_value = tl.load(input_ptr_mean + (batch_index), None, eviction_policy='evict_last')
    var_value = tl.load(input_ptr_var + (batch_index), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (batch_index), None, eviction_policy='evict_last')
    shift_value = tl.load(input_ptr_shift + (batch_index), None, eviction_policy='evict_last')
    
    normalized_value = input_value - mean_value
    variance_scale = 7840.0
    epsilon = 1e-05
    
    inv_stddev = tl.extra.cuda.libdevice.rsqrt((var_value / variance_scale) + epsilon)
    scaled_value = normalized_value * inv_stddev
    scaled_and_shifted_value = scaled_value * scale_value + shift_value
    
    relu_output = tl.full([1], 0, tl.int32)
    relu_result = triton_helpers.maximum(relu_output, scaled_and_shifted_value)
    
    tl.store(output_ptr + (input_index), relu_result, None)