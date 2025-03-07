# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_23poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x_channel = (x_index // 8192) % 256
    x_within_channel = x_index % 8192
    x_batch = x_index // 2097152
    x_flat_index = x_index
    
    channel_mask = x_channel < 128
    load_value_0 = tl.load(in_ptr0 + (x_within_channel + 8192 * x_channel + 1048576 * x_batch), channel_mask, other=0.0)
    load_value_1 = tl.load(in_ptr1 + (x_channel), channel_mask, eviction_policy='evict_last', other=0.0)
    sum_values = load_value_0 + load_value_1
    
    zero_filled = tl.full(sum_values.shape, 0.0, sum_values.dtype)
    selected_sum = tl.where(channel_mask, sum_values, zero_filled)
    
    channel_mask_not = x_channel >= 128
    load_value_2 = tl.load(in_ptr2 + (x_within_channel + 8192 * ((-128) + x_channel) + 1048576 * x_batch), channel_mask_not, other=0.0)
    final_value = tl.where(channel_mask, selected_sum, load_value_2)
    
    tl.store(out_ptr0 + (x_flat_index), final_value, None)