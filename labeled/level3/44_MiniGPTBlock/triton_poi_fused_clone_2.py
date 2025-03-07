# From: 44_MiniGPTBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_2poi_fused_clone_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 98304
    xnumel = 512
    
    # Calculate the y offset and index
    y_offset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_index < ynumel
    
    # Calculate the x offset and index
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    
    # Intermediate variables for indexing
    x_index_2 = x_index
    y_index_0 = y_index % 768
    y_index_1 = y_index // 768
    y_index_3 = y_index
    
    # Load data from input pointers with masking
    tmp0 = tl.load(in_ptr0 + (768 + y_index_0 + 2304 * x_index_2 + 1179648 * y_index_1), x_mask & y_mask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (768 + y_index_0), y_mask, eviction_policy='evict_last')
    
    # Perform computation
    tmp2 = tmp0 + tmp1
    
    # Store the result to the output pointer
    tl.store(out_ptr0 + (x_index_2 + 512 * y_index_3), tmp2, x_mask & y_mask)