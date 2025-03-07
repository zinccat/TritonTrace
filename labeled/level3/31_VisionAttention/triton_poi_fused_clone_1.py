# From: 31_VisionAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_1poi_fused_clone_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_128 = block_indices % 128
    index_div_128_mod_32768 = (block_indices // 128) % 32768
    index_div_4194304 = block_indices // 4194304
    full_index = block_indices
    
    load_address0 = in_ptr0 + (index_mod_128 + 128 * index_div_4194304 + 384 * index_div_128_mod_32768)
    load_address1 = in_ptr1 + (index_mod_128 + 128 * index_div_4194304)
    
    tmp0 = tl.load(load_address0, None)
    tmp1 = tl.load(load_address1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    
    tl.store(out_ptr0 + (full_index), tmp2, None)