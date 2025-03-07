# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_div_31poi_fused_clone_div_31(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 1505280
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    
    index_mod_32 = block_indices % 32
    index_div_32_mod_49 = (block_indices // 32) % 49
    index_div_1568_mod_6 = (block_indices // 1568) % 6
    index_div_9408 = block_indices // 9408
    index_div_32 = block_indices // 32
    full_index = block_indices
    
    input_value0 = tl.load(input_ptr0 + (index_mod_32 + 32 * index_div_1568_mod_6 + 576 * index_div_32_mod_49 + 28224 * index_div_9408), valid_mask)
    input_value1 = tl.load(input_ptr1 + (index_div_32), valid_mask, eviction_policy='evict_last')
    
    epsilon = 1e-12
    max_value = triton_helpers.maximum(input_value1, epsilon)
    
    result_value = input_value0 / max_value
    tl.store(output_ptr0 + (full_index), result_value, valid_mask)