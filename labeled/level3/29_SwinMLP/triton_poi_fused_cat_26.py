# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_26poi_fused_cat_26(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 376320
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel

    index_mod_192 = block_indices % 192
    index_div_192_mod_14 = (block_indices // 192) % 14
    index_div_2688 = block_indices // 2688
    index_div_192 = block_indices // 192

    input_offset = 5568 + index_mod_192 + 384 * index_div_192_mod_14 + 10752 * index_div_2688
    output_offset = index_mod_192 + 768 * index_div_192

    tmp0 = tl.load(in_ptr0 + input_offset, valid_mask)
    tl.store(out_ptr0 + output_offset, tmp0, valid_mask)