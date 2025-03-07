# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_16poi_fused_clone_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    
    # Calculate indices for accessing elements
    x_within_block = xindex % 32
    x_block_row = (xindex // 32) % 49
    x_depth = (xindex // 1568) % 6
    x_channel = xindex // 9408
    x_global_index = xindex
    
    # Load data from input pointers with calculated indices
    input0_index = (x_within_block + 32 * x_depth + 192 * (x_block_row % 7) + 1344 * (x_channel % 4) + 5376 * (x_block_row // 7) + 37632 * (x_channel // 4))
    input1_index = (7 * (x_channel % 4) + 28 * (x_block_row // 7) + 196 * (x_channel // 4) + (x_block_row % 7))
    
    tmp0 = tl.load(in_ptr0 + input0_index, xmask)
    tmp1 = tl.load(in_ptr1 + input1_index, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + input1_index, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x_within_block + 32 * x_depth), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x_within_block + 32 * x_depth), xmask, eviction_policy='evict_last')
    
    # Perform computations
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    
    # Store the result
    tl.store(out_ptr0 + x_global_index, tmp8, xmask)