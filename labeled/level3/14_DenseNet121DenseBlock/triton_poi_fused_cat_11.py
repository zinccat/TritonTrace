# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_11poi_fused_cat_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    channel_index = (block_indices // 50176) % 128
    spatial_index = block_indices % 50176
    batch_index = block_indices // 6422528
    linear_index = block_indices
    
    channel = channel_index
    condition_32 = channel < 32
    value_32 = tl.load(in_ptr0 + (spatial_index + 50176 * channel + 1605632 * batch_index), condition_32, other=0.0)
    
    condition_64 = (channel >= 32) & (channel < 64)
    value_64 = tl.load(in_ptr1 + (spatial_index + 50176 * (channel - 32) + 1605632 * batch_index), condition_64, other=0.0)
    
    condition_96 = (channel >= 64) & (channel < 96)
    value_96 = tl.load(in_ptr2 + (spatial_index + 50176 * (channel - 64) + 1605632 * batch_index), condition_96, other=0.0)
    
    condition_128 = channel >= 96
    value_128 = tl.load(in_ptr3 + (spatial_index + 50176 * (channel - 96) + 1605632 * batch_index), condition_128, other=0.0)
    
    result_96_or_128 = tl.where(condition_96, value_96, value_128)
    result_64_or_96_or_128 = tl.where(condition_64, value_64, result_96_or_128)
    final_result = tl.where(condition_32, value_32, result_64_or_96_or_128)
    
    tl.store(out_ptr0 + (linear_index), final_result, None)