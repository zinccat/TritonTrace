# From: 28_VisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_3poi_fused_clone_3(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 605184
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    index_mod_512 = block_indices % 512
    index_div_512_mod_394 = (block_indices // 512) % 394
    index_div_201728 = block_indices // 201728
    linear_index = block_indices
    temp0 = tl.load(input_ptr0 + (index_mod_512 + 512 * index_div_201728 + 1536 * index_div_512_mod_394), valid_mask)
    temp1 = tl.load(input_ptr1 + (index_mod_512 + 512 * index_div_201728), valid_mask, eviction_policy='evict_last')
    temp2 = temp0 + temp1
    tl.store(output_ptr0 + (linear_index), temp2, valid_mask)