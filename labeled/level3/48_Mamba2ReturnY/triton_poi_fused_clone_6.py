# From: 48_Mamba2ReturnY

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_6poi_fused_clone_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    block_index_mod_64 = block_indices % 64
    block_index_div_64_mod_64 = (block_indices // 64) % 64
    block_index_div_4096_mod_8 = (block_indices // 4096) % 8
    block_index_div_32768 = block_indices // 32768
    linear_index = block_indices
    
    tmp0 = tl.load(in_ptr0 + (block_index_mod_64 + 64 * block_index_div_4096_mod_8 + 512 * block_index_div_64_mod_64 + 32768 * block_index_div_32768), None)
    tl.store(out_ptr0 + (linear_index), tmp0, None)
    tl.store(out_ptr1 + (linear_index), tmp0, None)