# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_15poi_fused_cat_15(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    channel_index = (x_index // 50176) % 160
    spatial_index = x_index % 50176
    batch_index = x_index // 8028160
    linear_index = x_index
    
    channel_value = channel_index
    zero_value = tl.full([1], 0, tl.int64)
    threshold_32 = tl.full([1], 32, tl.int64)
    threshold_64 = tl.full([1], 64, tl.int64)
    threshold_96 = tl.full([1], 96, tl.int64)
    threshold_128 = tl.full([1], 128, tl.int64)
    
    load_condition_32 = channel_value < threshold_32
    load_condition_64 = (channel_value >= threshold_32) & (channel_value < threshold_64)
    load_condition_96 = (channel_value >= threshold_64) & (channel_value < threshold_96)
    load_condition_128 = (channel_value >= threshold_96) & (channel_value < threshold_128)
    load_condition_160 = channel_value >= threshold_128
    
    value_32 = tl.load(
        input_ptr0 + (spatial_index + 50176 * channel_value + 1605632 * batch_index), 
        load_condition_32, 
        other=0.0
    )
    value_64 = tl.load(
        input_ptr1 + (spatial_index + 50176 * (channel_value - 32) + 1605632 * batch_index), 
        load_condition_64, 
        other=0.0
    )
    value_96 = tl.load(
        input_ptr2 + (spatial_index + 50176 * (channel_value - 64) + 1605632 * batch_index), 
        load_condition_96, 
        other=0.0
    )
    value_128 = tl.load(
        input_ptr3 + (spatial_index + 50176 * (channel_value - 96) + 1605632 * batch_index), 
        load_condition_128, 
        other=0.0
    )
    value_160 = tl.load(
        input_ptr4 + (spatial_index + 50176 * (channel_value - 128) + 1605632 * batch_index), 
        load_condition_160, 
        other=0.0
    )
    
    result_128 = tl.where(load_condition_128, value_128, value_160)
    result_96 = tl.where(load_condition_96, value_96, result_128)
    result_64 = tl.where(load_condition_64, value_64, result_96)
    result_32 = tl.where(load_condition_32, value_32, result_64)
    
    final_result = result_32
    tl.store(output_ptr0 + (linear_index), final_result, None)