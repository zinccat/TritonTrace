# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_54poi_fused__native_batch_norm_legit_functional_hardtanh_54(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_out, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    local_indices = block_indices % 672
    input_mean = tl.load(input_ptr_mean + (global_indices), None)
    input_var = tl.load(input_ptr_var + (local_indices), None, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (local_indices), None, eviction_policy='evict_last')
    input_shift = tl.load(input_ptr_shift + (local_indices), None, eviction_policy='evict_last')
    output_min = tl.load(input_ptr_out + (local_indices), None, eviction_policy='evict_last')
    
    normalized_input = (input_mean - input_var) * input_var * input_scale
    activated_output = normalized_input + output_min
    clamped_output = triton_helpers.maximum(activated_output, 0.0)
    final_output = triton_helpers.minimum(clamped_output, 6.0)
    
    tl.store(output_ptr + (global_indices), final_output, None)