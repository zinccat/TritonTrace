# From: 44_MiniGPTBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_6poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_96 = block_indices % 96
    index_div_96_mod_8 = (block_indices // 96) % 8
    index_div_768_mod_512 = (block_indices // 768) % 512
    index_div_393216 = block_indices // 393216
    original_index = block_indices
    
    tmp0 = tl.load(in_ptr0 + (index_mod_96 + 96 * index_div_768_mod_512 + 49152 * index_div_96_mod_8 + 393216 * index_div_393216), None)
    tl.store(out_ptr0 + (original_index), tmp0, None)