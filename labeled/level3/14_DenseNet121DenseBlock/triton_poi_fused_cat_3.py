# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_3poi_fused_cat_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    channel_index = (block_indices // 50176) % 64
    spatial_index = block_indices % 50176
    batch_index = block_indices // 3211264
    linear_index = block_indices
    
    channel = channel_index
    is_within_first_input = tl.full([1], 0, tl.int64)
    channel_threshold = tl.full([1], 32, tl.int64)
    is_within_first_input = channel < channel_threshold
    
    value_from_first_input = tl.load(
        in_ptr0 + (spatial_index + 50176 * channel_index + 1605632 * batch_index), 
        is_within_first_input, 
        other=0.0
    )
    
    is_within_second_input = channel >= channel_threshold
    channel_offset = tl.full([1], 64, tl.int64)
    
    value_from_second_input = tl.load(
        in_ptr1 + (spatial_index + 50176 * ((-32) + channel_index) + 1605632 * batch_index), 
        is_within_second_input, 
        other=0.0
    )
    
    selected_value = tl.where(is_within_first_input, value_from_first_input, value_from_second_input)
    
    tl.store(out_ptr0 + (linear_index), selected_value, None)