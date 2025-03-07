# From: 36_RMSNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_mean_pow_sqrt_1poi_fused_add_div_mean_pow_sqrt_1(input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    index_mod_k0 = block_indices % kernel_size0
    index_div_k1 = block_indices // kernel_size1
    
    input_value0 = tl.load(input_ptr0 + (linear_index), valid_mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (index_mod_k0 + kernel_size0 * index_div_k1), valid_mask, eviction_policy='evict_last')
    
    divisor = kernel_size2
    divisor_float = divisor.to(tl.float32)
    
    normalized_value = input_value1 / divisor_float
    epsilon = 1e-05
    adjusted_value = normalized_value + epsilon
    
    sqrt_value = tl.extra.cuda.libdevice.sqrt(adjusted_value)
    result_value = input_value0 / sqrt_value
    
    tl.store(output_ptr0 + (linear_index), result_value, valid_mask)