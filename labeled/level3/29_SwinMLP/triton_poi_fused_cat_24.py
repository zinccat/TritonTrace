# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_24poi_fused_cat_24(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 376320
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < xnumel
    
    element_within_group = index_within_block % 192
    group_index_within_row = (index_within_block // 192) % 14
    row_index = index_within_block // 2688
    group_index = index_within_block // 192
    
    input_offset = 5376 + element_within_group + 384 * group_index_within_row + 10752 * row_index
    output_offset = element_within_group + 768 * group_index
    
    tmp0 = tl.load(in_ptr0 + input_offset, valid_mask)
    tl.store(out_ptr0 + output_offset, tmp0, valid_mask)