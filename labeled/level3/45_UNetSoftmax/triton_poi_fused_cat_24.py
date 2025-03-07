# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_24poi_fused_cat_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x_channel = (x_index // 32768) % 128
    x_within_channel = x_index % 32768
    x_batch = x_index // 4194304
    x_flat_index = x_index
    
    channel_mask = x_channel
    tl.full([1], 0, tl.int64)
    max_channel = tl.full([1], 64, tl.int64)
    is_within_first_half = channel_mask < max_channel
    
    load_first_half = tl.load(in_ptr0 + (x_within_channel + 32768 * (x_channel) + 2097152 * x_batch), is_within_first_half, other=0.0)
    load_second_half = tl.load(in_ptr1 + (x_channel), is_within_first_half, eviction_policy='evict_last', other=0.0)
    sum_loads = load_first_half + load_second_half
    
    zero_tensor = tl.full(sum_loads.shape, 0.0, sum_loads.dtype)
    result_first_half = tl.where(is_within_first_half, sum_loads, zero_tensor)
    
    is_within_second_half = channel_mask >= max_channel
    max_channel_second_half = tl.full([1], 128, tl.int64)
    
    load_second_half_offset = tl.load(in_ptr2 + (x_within_channel + 32768 * ((-64) + x_channel) + 2097152 * x_batch), is_within_second_half, other=0.0)
    final_result = tl.where(is_within_first_half, result_first_half, load_second_half_offset)
    
    tl.store(out_ptr0 + (x_flat_index), final_result, None)