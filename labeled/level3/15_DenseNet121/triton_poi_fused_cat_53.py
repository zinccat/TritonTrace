# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_53poi_fused_cat_53(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 2759680
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements

    channel_index = (indices // 784) % 352
    pixel_index = indices % 784
    batch_index = indices // 275968
    flat_index = indices

    channel_128 = channel_index
    channel_160 = tl.full([1], 160, tl.int64)
    channel_192 = tl.full([1], 192, tl.int64)
    channel_224 = tl.full([1], 224, tl.int64)
    channel_256 = tl.full([1], 256, tl.int64)
    channel_288 = tl.full([1], 288, tl.int64)
    channel_320 = tl.full([1], 320, tl.int64)
    channel_352 = tl.full([1], 352, tl.int64)

    mask_128 = channel_128 < channel_160
    mask_160 = (channel_128 >= channel_160) & (channel_128 < channel_192)
    mask_192 = (channel_128 >= channel_192) & (channel_128 < channel_224)
    mask_224 = (channel_128 >= channel_224) & (channel_128 < channel_256)
    mask_256 = (channel_128 >= channel_256) & (channel_128 < channel_288)
    mask_288 = (channel_128 >= channel_288) & (channel_128 < channel_320)
    mask_320 = (channel_128 >= channel_320) & (channel_128 < channel_352)

    value_128 = tl.load(input_ptr0 + (pixel_index + 784 * channel_128 + 100352 * batch_index), mask_128 & mask, other=0.0)
    value_160 = tl.load(input_ptr1 + (pixel_index + 784 * (channel_128 - 128) + 25088 * batch_index), mask_160 & mask, other=0.0)
    value_192 = tl.load(input_ptr2 + (pixel_index + 784 * (channel_128 - 160) + 25088 * batch_index), mask_192 & mask, other=0.0)
    value_224 = tl.load(input_ptr3 + (pixel_index + 784 * (channel_128 - 192) + 25088 * batch_index), mask_224 & mask, other=0.0)
    value_256 = tl.load(input_ptr4 + (pixel_index + 784 * (channel_128 - 224) + 25088 * batch_index), mask_256 & mask, other=0.0)
    value_288 = tl.load(input_ptr5 + (pixel_index + 784 * (channel_128 - 256) + 25088 * batch_index), mask_288 & mask, other=0.0)
    value_320 = tl.load(input_ptr6 + (pixel_index + 784 * (channel_128 - 288) + 25088 * batch_index), mask_320 & mask, other=0.0)
    value_352 = tl.load(input_ptr7 + (pixel_index + 784 * (channel_128 - 320) + 25088 * batch_index), (channel_128 >= channel_352) & mask, other=0.0)

    result = tl.where(mask_320, value_320, value_352)
    result = tl.where(mask_288, value_288, result)
    result = tl.where(mask_256, value_256, result)
    result = tl.where(mask_224, value_224, result)
    result = tl.where(mask_192, value_192, result)
    result = tl.where(mask_160, value_160, result)
    result = tl.where(mask_128, value_128, result)

    tl.store(output_ptr0 + (flat_index), result, mask)