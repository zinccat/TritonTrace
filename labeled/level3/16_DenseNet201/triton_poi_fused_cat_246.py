# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_246poi_fused_cat_246(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    num_elements = 548800
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements

    channel_index = (indices // 49) % 1120
    row_index = indices % 49
    batch_index = indices // 54880
    flat_index = indices

    channel_base = channel_index

    zero_value = tl.full([1], 0, tl.int64)
    threshold_896 = tl.full([1], 896, tl.int64)
    condition_896 = channel_base < threshold_896
    value_0 = tl.load(input_ptr0 + (row_index + 49 * channel_index + 43904 * batch_index), condition_896 & mask, other=0.0)

    threshold_928 = tl.full([1], 928, tl.int64)
    condition_928 = (channel_base >= threshold_896) & (channel_base < threshold_928)
    value_1 = tl.load(input_ptr1 + (row_index + 49 * (channel_index - 896) + 1568 * batch_index), condition_928 & mask, other=0.0)

    threshold_960 = tl.full([1], 960, tl.int64)
    condition_960 = (channel_base >= threshold_928) & (channel_base < threshold_960)
    value_2 = tl.load(input_ptr2 + (row_index + 49 * (channel_index - 928) + 1568 * batch_index), condition_960 & mask, other=0.0)

    threshold_992 = tl.full([1], 992, tl.int64)
    condition_992 = (channel_base >= threshold_960) & (channel_base < threshold_992)
    value_3 = tl.load(input_ptr3 + (row_index + 49 * (channel_index - 960) + 1568 * batch_index), condition_992 & mask, other=0.0)

    threshold_1024 = tl.full([1], 1024, tl.int64)
    condition_1024 = (channel_base >= threshold_992) & (channel_base < threshold_1024)
    value_4 = tl.load(input_ptr4 + (row_index + 49 * (channel_index - 992) + 1568 * batch_index), condition_1024 & mask, other=0.0)

    threshold_1056 = tl.full([1], 1056, tl.int64)
    condition_1056 = (channel_base >= threshold_1024) & (channel_base < threshold_1056)
    value_5 = tl.load(input_ptr5 + (row_index + 49 * (channel_index - 1024) + 1568 * batch_index), condition_1056 & mask, other=0.0)

    threshold_1088 = tl.full([1], 1088, tl.int64)
    condition_1088 = (channel_base >= threshold_1056) & (channel_base < threshold_1088)
    value_6 = tl.load(input_ptr6 + (row_index + 49 * (channel_index - 1056) + 1568 * batch_index), condition_1088 & mask, other=0.0)

    threshold_1120 = tl.full([1], 1120, tl.int64)
    condition_1120 = channel_base >= threshold_1088
    value_7 = tl.load(input_ptr7 + (row_index + 49 * (channel_index - 1088) + 1568 * batch_index), condition_1120 & mask, other=0.0)

    result = tl.where(condition_1088, value_6, value_7)
    result = tl.where(condition_1056, value_5, result)
    result = tl.where(condition_1024, value_4, result)
    result = tl.where(condition_992, value_3, result)
    result = tl.where(condition_960, value_2, result)
    result = tl.where(condition_928, value_1, result)
    result = tl.where(condition_896, value_0, result)

    tl.store(output_ptr0 + (flat_index), result, mask)