# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_22poi_fused_cat_22(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    channel_index = (block_indices // 50176) % 224
    spatial_index = block_indices % 50176
    batch_index = block_indices // 11239424
    linear_index = block_indices
    
    channel = channel_index
    
    zero = tl.full([1], 0, tl.int64)
    threshold_32 = tl.full([1], 32, tl.int64)
    threshold_64 = tl.full([1], 64, tl.int64)
    threshold_96 = tl.full([1], 96, tl.int64)
    threshold_128 = tl.full([1], 128, tl.int64)
    threshold_160 = tl.full([1], 160, tl.int64)
    threshold_192 = tl.full([1], 192, tl.int64)
    threshold_224 = tl.full([1], 224, tl.int64)
    
    load_0 = tl.load(input_ptr0 + (spatial_index + 50176 * channel_index + 1605632 * batch_index), channel_index < threshold_32, other=0.0)
    load_1 = tl.load(input_ptr1 + (spatial_index + 50176 * (channel_index - 32) + 1605632 * batch_index), (channel_index >= threshold_32) & (channel_index < threshold_64), other=0.0)
    load_2 = tl.load(input_ptr2 + (spatial_index + 50176 * (channel_index - 64) + 1605632 * batch_index), (channel_index >= threshold_64) & (channel_index < threshold_96), other=0.0)
    load_3 = tl.load(input_ptr3 + (spatial_index + 50176 * (channel_index - 96) + 1605632 * batch_index), (channel_index >= threshold_96) & (channel_index < threshold_128), other=0.0)
    load_4 = tl.load(input_ptr4 + (spatial_index + 50176 * (channel_index - 128) + 1605632 * batch_index), (channel_index >= threshold_128) & (channel_index < threshold_160), other=0.0)
    load_5 = tl.load(input_ptr5 + (spatial_index + 50176 * (channel_index - 160) + 1605632 * batch_index), (channel_index >= threshold_160) & (channel_index < threshold_192), other=0.0)
    load_6 = tl.load(input_ptr6 + (spatial_index + 50176 * (channel_index - 192) + 1605632 * batch_index), channel_index >= threshold_192, other=0.0)
    
    result = tl.where((channel_index >= threshold_192), load_6, tl.where(
        (channel_index >= threshold_160), load_5, tl.where(
            (channel_index >= threshold_128), load_4, tl.where(
                (channel_index >= threshold_96), load_3, tl.where(
                    (channel_index >= threshold_64), load_2, tl.where(
                        (channel_index >= threshold_32), load_1, load_0
                    )
                )
            )
        )
    ))
    
    tl.store(output_ptr0 + linear_index, result, None)