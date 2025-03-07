# From: 48_Mamba2ReturnY

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bmm_5poi_fused_bmm_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_64 = block_indices % 64
    index_div_64_mod_256 = (block_indices // 64) % 256
    index_div_16384 = block_indices // 16384
    full_index = block_indices
    
    tmp0 = tl.load(
        in_ptr0 + (index_mod_64 + 64 * (index_div_64_mod_256 % 8) + 512 * index_div_16384 + 32768 * (index_div_64_mod_256 // 8)),
        None
    )
    tl.store(out_ptr0 + (full_index), tmp0, None)