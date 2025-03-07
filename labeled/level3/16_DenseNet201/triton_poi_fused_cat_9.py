# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_9poi_fused_cat_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    channel_index = (x_index // 3136) % 96
    spatial_index = x_index % 3136
    batch_index = x_index // 301056
    linear_index = x_index
    
    channel_temp = channel_index
    tl.full([1], 0, tl.int64)
    channel_threshold = tl.full([1], 64, tl.int64)
    is_below_threshold = channel_temp < channel_threshold
    
    load_from_in_ptr0 = tl.load(
        in_ptr0 + (spatial_index + 3136 * channel_index + 200704 * batch_index),
        is_below_threshold,
        other=0.0
    )
    
    is_above_threshold = channel_temp >= channel_threshold
    tl.full([1], 96, tl.int64)
    
    load_from_in_ptr1 = tl.load(
        in_ptr1 + (spatial_index + 3136 * ((-64) + channel_index) + 100352 * batch_index),
        is_above_threshold,
        other=0.0
    )
    
    selected_value = tl.where(is_below_threshold, load_from_in_ptr0, load_from_in_ptr1)
    tl.store(out_ptr0 + (linear_index), selected_value, None)