# From: 91_cumsum_reverse

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_flip_1poi_fused_flip_1(in_ptr0, out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < num_elements
    
    index_mod_kernel = block_indices % kernel_size
    index_div_kernel = block_indices // kernel_size
    original_index = block_indices
    
    temp_value = tl.load(
        in_ptr0 + ((-1) + kernel_size + ((-1) * index_mod_kernel) + kernel_size * index_div_kernel),
        valid_mask,
        eviction_policy='evict_last'
    )
    
    tl.store(out_ptr0 + (original_index), temp_value, valid_mask)