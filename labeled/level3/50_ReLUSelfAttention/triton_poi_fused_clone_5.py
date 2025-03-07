# From: 50_ReLUSelfAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_5poi_fused_clone_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_64 = block_indices % 64
    index_div_64_mod_12 = (block_indices // 64) % 12
    index_div_768_mod_1024 = (block_indices // 768) % 1024
    index_div_786432 = block_indices // 786432
    linear_index = block_indices
    
    tmp0 = tl.load(in_ptr0 + (index_mod_64 + 64 * index_div_768_mod_1024 + 65536 * index_div_64_mod_12 + 786432 * index_div_786432), None)
    tl.store(out_ptr0 + (linear_index), tmp0, None)