# From: 49_Mamba2ReturnFinalState

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_2poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_64 = block_indices % 64
    index_div_64_mod_64 = (block_indices // 64) % 64
    index_div_4096_mod_8 = (block_indices // 4096) % 8
    index_div_32768 = block_indices // 32768
    original_index = block_indices
    
    temp_value = tl.load(in_ptr0 + (index_mod_64 + 64 * index_div_4096_mod_8 + 512 * index_div_64_mod_64 + 32768 * index_div_32768), None)
    tl.store(out_ptr0 + (original_index), temp_value, None)