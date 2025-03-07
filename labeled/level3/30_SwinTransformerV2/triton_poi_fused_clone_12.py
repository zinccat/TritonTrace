# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_12poi_fused_clone_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_32 = block_indices % 32
    index_div_32_mod_3 = (block_indices // 32) % 3
    index_div_96_mod_49 = (block_indices // 96) % 49
    index_div_4704 = block_indices // 4704
    original_index = block_indices
    
    tmp0 = tl.load(in_ptr0 + (index_mod_32 + 32 * index_div_96_mod_49 + 1568 * index_div_32_mod_3 + 4704 * index_div_4704), None)
    tl.store(out_ptr0 + (original_index), tmp0, None)