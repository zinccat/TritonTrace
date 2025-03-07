# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_41poi_fused_cat_41(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 188160
    block_id = tl.program_id(0) * XBLOCK
    index_within_block = tl.arange(0, XBLOCK)[:]
    valid_indices_mask = index_within_block < xnumel
    
    element_within_group = index_within_block % 384
    group_index_within_batch = (index_within_block // 384) % 7
    batch_index = index_within_block // 2688
    group_index = index_within_block // 384
    
    input_offset = 384 + element_within_group + 768 * group_index_within_batch + 10752 * batch_index
    output_offset = element_within_group + 1536 * group_index
    
    temp_data = tl.load(in_ptr0 + input_offset, valid_indices_mask)
    tl.store(out_ptr0 + output_offset, temp_data, valid_indices_mask)