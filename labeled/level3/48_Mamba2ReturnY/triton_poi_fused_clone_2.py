# From: 48_Mamba2ReturnY

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_2poi_fused_clone_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_index = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    block_row = block_index % 16
    block_col = (block_index // 16) % 64
    block_depth = (block_index // 1024) % 8
    block_slice = block_index // 8192
    linear_index = block_index
    
    input_offset = block_row + 16 * block_depth + 128 * block_col + 8192 * block_slice
    tmp0 = tl.load(in_ptr0 + input_offset, None)
    
    tl.store(out_ptr0 + linear_index, tmp0, None)
    tl.store(out_ptr1 + linear_index, tmp0, None)