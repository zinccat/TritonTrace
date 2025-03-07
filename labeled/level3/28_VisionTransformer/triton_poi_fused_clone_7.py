# From: 28_VisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_7poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 201728
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel

    index_mod_512 = block_indices % 512
    index_div_512_mod_197 = (block_indices // 512) % 197
    index_div_100864 = block_indices // 100864
    original_index = block_indices

    temp_data = tl.load(in_ptr0 + (index_mod_512 + 512 * index_div_100864 + 1024 * index_div_512_mod_197), valid_mask)
    tl.store(out_ptr0 + (original_index), temp_data, valid_mask)