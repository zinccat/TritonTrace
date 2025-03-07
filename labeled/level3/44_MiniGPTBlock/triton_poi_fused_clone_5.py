# From: 44_MiniGPTBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_5poi_fused_clone_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_96 = block_indices % 96
    index_div_96_mod_512 = (block_indices // 96) % 512
    index_div_49152_mod_8 = (block_indices // 49152) % 8
    index_div_393216 = block_indices // 393216
    full_index = block_indices
    
    load_address0 = 1536 + index_mod_96 + 96 * index_div_49152_mod_8 + 2304 * index_div_96_mod_512 + 1179648 * index_div_393216
    load_address1 = 1536 + index_mod_96 + 96 * index_div_49152_mod_8
    
    tmp0 = tl.load(in_ptr0 + load_address0, None)
    tmp1 = tl.load(in_ptr1 + load_address1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    
    tl.store(out_ptr0 + full_index, tmp2, None)