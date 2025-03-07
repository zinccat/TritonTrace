# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_div_54poi_fused_clone_div_54(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 752640
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    
    index_mod_32 = block_indices % 32
    index_div_32_mod_49 = (block_indices // 32) % 49
    index_div_1568_mod_12 = (block_indices // 1568) % 12
    index_div_18816 = block_indices // 18816
    index_div_32 = block_indices // 32
    full_index = block_indices
    
    temp0 = tl.load(input_ptr0 + (index_mod_32 + 32 * index_div_1568_mod_12 + 1152 * index_div_32_mod_49 + 56448 * index_div_18816), valid_mask)
    temp1 = tl.load(input_ptr1 + (index_div_32), valid_mask, eviction_policy='evict_last')
    epsilon = 1e-12
    max_value = triton_helpers.maximum(temp1, epsilon)
    result = temp0 / max_value
    
    tl.store(output_ptr0 + (full_index), result, valid_mask)