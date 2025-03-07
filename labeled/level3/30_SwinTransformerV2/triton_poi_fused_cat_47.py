# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_47poi_fused_cat_47(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 376320
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = tl.arange(0, XBLOCK)[:]
    valid_indices_mask = index_within_block < xnumel
    
    index_within_192 = index_within_block % 192
    index_within_14 = (index_within_block // 192) % 14
    index_within_2688 = index_within_block // 2688
    index_within_192_full = index_within_block // 192
    
    input_offset = 5568 + index_within_192 + 384 * index_within_14 + 10752 * index_within_2688
    output_offset = index_within_192 + 768 * index_within_192_full
    
    tmp0 = tl.load(in_ptr0 + input_offset, valid_indices_mask)
    tl.store(out_ptr0 + output_offset, tmp0, valid_indices_mask)