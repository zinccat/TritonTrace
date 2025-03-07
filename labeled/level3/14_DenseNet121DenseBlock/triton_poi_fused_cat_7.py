# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_7poi_fused_cat_7(input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    channel_index = (block_indices // 50176) % 96
    spatial_index = block_indices % 50176
    batch_index = block_indices // 4816896
    linear_index = block_indices
    
    channel = channel_index
    zero = tl.full([1], 0, tl.int64)
    threshold_32 = tl.full([1], 32, tl.int64)
    condition_32 = channel < threshold_32
    value_32 = tl.load(input_ptr0 + (spatial_index + 50176 * channel + 1605632 * batch_index), condition_32, other=0.0)
    
    threshold_64 = tl.full([1], 64, tl.int64)
    condition_64 = channel < threshold_64
    condition_32_and_64 = (channel >= threshold_32) & condition_64
    value_64 = tl.load(input_ptr1 + (spatial_index + 50176 * ((-32) + channel) + 1605632 * batch_index), condition_32_and_64, other=0.0)
    
    condition_above_64 = channel >= threshold_64
    threshold_96 = tl.full([1], 96, tl.int64)
    value_96 = tl.load(input_ptr2 + (spatial_index + 50176 * ((-64) + channel) + 1605632 * batch_index), condition_above_64, other=0.0)
    
    selected_value_64_or_96 = tl.where(condition_32_and_64, value_64, value_96)
    selected_value = tl.where(condition_32, value_32, selected_value_64_or_96)
    
    tl.store(output_ptr0 + (linear_index), selected_value, None)