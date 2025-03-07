# From: 33_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_mul_1poi_fused__native_batch_norm_legit_functional_mul_1(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, input_ptr_output, 
    output_ptr, scale_factor, num_elements, BLOCK_SIZE : tl.constexpr):

    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    block_indices_mod = block_indices % 512

    mean = tl.load(input_ptr_mean + (global_indices), valid_mask)
    variance = tl.load(input_ptr_var + (block_indices_mod), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (block_indices_mod), valid_mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (block_indices_mod), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (global_indices), valid_mask, eviction_policy='evict_last')
    output_data = tl.load(input_ptr_output + (block_indices_mod), valid_mask, eviction_policy='evict_last')

    normalized_input = (input_data - mean) * scale
    epsilon = 1e-05
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(normalized_input + epsilon)
    scaled_input = normalized_input * inv_std_dev
    batch_norm_output = scaled_input * input_data + shift

    tl.store(output_ptr + (global_indices), batch_norm_output, valid_mask)