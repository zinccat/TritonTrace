# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_7poi_fused__native_batch_norm_legit_functional_hardtanh_7(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_out, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):

    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    element_indices = block_indices
    element_index_mod = (block_indices % 32)
    
    input_mean = tl.load(input_ptr_mean + (element_indices), None)
    input_var = tl.load(input_ptr_var + (element_index_mod), None, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (element_index_mod), None, eviction_policy='evict_last')
    input_shift = tl.load(input_ptr_shift + (element_index_mod), None, eviction_policy='evict_last')
    input_out = tl.load(input_ptr_out + (element_index_mod), None, eviction_policy='evict_last')
    
    normalized_input = (input_mean - input_var) * input_var * input_scale
    batch_norm_output = normalized_input + input_shift
    
    clamped_output = triton_helpers.maximum(batch_norm_output, 0.0)
    clamped_output = triton_helpers.minimum(clamped_output, 6.0)
    
    tl.store(output_ptr + (element_indices), clamped_output, None)