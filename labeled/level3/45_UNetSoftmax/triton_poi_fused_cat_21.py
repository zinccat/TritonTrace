# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_21poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    block_row = (x_index // 512) % 1024
    block_col = x_index % 512
    block_depth = x_index // 524288
    linear_index = x_index
    
    block_row_check = block_row
    tl.full([1], 0, tl.int64)
    max_block_row = tl.full([1], 512, tl.int64)
    is_within_block = block_row_check < max_block_row
    
    load0 = tl.load(in_ptr0 + (block_col + 512 * block_row + 262144 * block_depth), is_within_block, other=0.0)
    load1 = tl.load(in_ptr1 + (block_row), is_within_block, eviction_policy='evict_last', other=0.0)
    sum_loads = load0 + load1
    
    zero_tensor = tl.full(sum_loads.shape, 0.0, sum_loads.dtype)
    conditional_sum = tl.where(is_within_block, sum_loads, zero_tensor)
    
    is_out_of_block = block_row_check >= max_block_row
    tl.full([1], 1024, tl.int64)
    
    load2 = tl.load(in_ptr2 + (block_col + 512 * ((-512) + block_row) + 262144 * block_depth), is_out_of_block, other=0.0)
    final_result = tl.where(is_within_block, conditional_sum, load2)
    
    tl.store(out_ptr0 + (linear_index), final_result, None)